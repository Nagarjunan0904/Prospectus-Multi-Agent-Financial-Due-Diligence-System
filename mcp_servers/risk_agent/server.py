"""
Risk-Agent MCP server — red-flag detection layer.

Transports
----------
HTTP (default)
    Streamable-HTTP on :9004/mcp, protected by MCP_RISK_AGENT_TOKEN.
    Start with::

        python -m mcp_servers.risk_agent.server

stdio
    Local Claude Desktop testing::

        python -m mcp_servers.risk_agent.server --transport stdio

Tools
-----
run_all_checks(cik, ratio_history, mdna_text, risk_factors_text)
    Primary tool for the due-diligence pipeline.  Runs all three
    detectors and returns a flat list matching
    DueDiligenceState["risk_flags"].

detect_debt_spike(ratio_history)
    Compares D/E across the last N annual periods; flags >25 % and
    >50 % increases at 'medium' / 'high' severity.

detect_insider_selling_cluster(cik, days=30)
    Calls the Data Agent for the full insider-transaction list; flags
    3+ distinct sellers with zero offsetting purchases.

flag_audit_language(mdna_text, risk_factors_text)
    Word-boundary regex scan for going-concern, material-weakness,
    and restatement language in MD&A and Risk Factors sections.
"""
import argparse
import asyncio
import contextlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount

from shared.audit_log import make_params_hash, record as _audit_record
from mcp_servers.risk_agent.tools.red_flags import (
    detect_debt_spike,
    detect_insider_selling_cluster,
    flag_audit_language,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = Server("risk-agent")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_all_checks",
            description=(
                "Run all risk detectors and return a flat list of flags matching "
                "DueDiligenceState['risk_flags']. "
                "Each detector is fault-tolerant: an error in one never silences the others. "
                "Flags: [{flag, severity, evidence, source_tool}, ...]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cik": {
                        "type": "string",
                        "description": "10-digit zero-padded CIK for insider-transaction lookups.",
                    },
                    "ratio_history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Output of quant_agent.get_ratio_history: "
                            "[{period_end, debt_to_equity}, ...]."
                        ),
                    },
                    "mdna_text": {
                        "type": "string",
                        "description": "MD&A section text from get_filing_sections.",
                    },
                    "risk_factors_text": {
                        "type": "string",
                        "description": "Risk Factors section text from get_filing_sections.",
                    },
                    "days": {
                        "type": "integer",
                        "default": 90,
                        "description": (
                            "Look-back window in days for the insider-selling check "
                            "(default 90 — one full quarter; verified against real "
                            "Form 4 data where 30-day windows miss genuine clusters)."
                        ),
                    },
                },
                "required": ["cik", "ratio_history", "mdna_text", "risk_factors_text"],
            },
        ),
        Tool(
            name="detect_debt_spike",
            description=(
                "Detect anomalous D/E increases across annual 10-K periods. "
                "Thresholds: >25 % increase → 'medium', >50 % → 'high'. "
                "Two comparisons: debt_spike_recent (adjacent annual filings) and "
                "debt_spike_multi_year (~4 filings back). "
                "Returns [] when fewer than 2 non-None data points are available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ratio_history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "[{period_end: YYYY-MM-DD, debt_to_equity: float|null}, ...] "
                            "from quant_agent.get_ratio_history."
                        ),
                    },
                },
                "required": ["ratio_history"],
            },
        ),
        Tool(
            name="detect_insider_selling_cluster",
            description=(
                "Flag coordinated insider selling with no offsetting purchases. "
                "Fetches the full transaction list from the Data Agent. "
                "Severity: 'medium' for 3–4 distinct sellers, 'high' for 5+. "
                "Returns [] when fewer than 3 sellers or any purchase exists."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cik": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Look-back window in days (default 30).",
                    },
                },
                "required": ["cik"],
            },
        ),
        Tool(
            name="flag_audit_language",
            description=(
                "Scan MD&A and Risk Factors text for high-risk disclosure language "
                "using word-boundary regex. "
                "Detects: going concern (high), material weakness (high), "
                "restatement (medium). "
                "One flag per matched category. Returns [] if both texts are empty."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mdna_text":          {"type": "string"},
                    "risk_factors_text":  {"type": "string"},
                },
                "required": ["mdna_text", "risk_factors_text"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Per-tool implementations
# ---------------------------------------------------------------------------

async def _impl_run_all_checks(
    cik: str,
    ratio_history: list[dict],
    mdna_text: str,
    risk_factors_text: str,
    days: int = 90,
) -> Any:
    return await run_all_checks(cik, ratio_history, mdna_text, risk_factors_text, days=days)


async def _impl_detect_debt_spike(ratio_history: list[dict]) -> Any:
    return detect_debt_spike(ratio_history)


async def _impl_detect_insider_selling_cluster(
    cik: str,
    days: int = 30,
) -> Any:
    return await detect_insider_selling_cluster(cik, days)


async def _impl_flag_audit_language(
    mdna_text: str,
    risk_factors_text: str,
) -> Any:
    return flag_audit_language(mdna_text, risk_factors_text)


_TOOLS: dict[str, Any] = {
    "run_all_checks":                    _impl_run_all_checks,
    "detect_debt_spike":                 _impl_detect_debt_spike,
    "detect_insider_selling_cluster":    _impl_detect_insider_selling_cluster,
    "flag_audit_language":               _impl_flag_audit_language,
}


# ---------------------------------------------------------------------------
# call_tool with audit logging
# ---------------------------------------------------------------------------

async def _audit(
    tool: str,
    ph: str,
    started_at: datetime,
    latency_ms: float,
    status: str,
) -> None:
    await asyncio.to_thread(
        _audit_record, "risk", tool, ph, started_at, latency_ms, status
    )


@mcp.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    arguments = arguments or {}
    ph = make_params_hash(arguments)
    started_at = datetime.now(timezone.utc)

    fn = _TOOLS.get(name)
    if fn is None:
        latency_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        await _audit(name, ph, started_at, latency_ms, "error")
        raise ValueError(f"Unknown tool: {name!r}")

    try:
        result = await fn(**arguments)
        latency_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        await _audit(name, ph, started_at, latency_ms, "success")
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    except Exception:
        latency_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
        await _audit(name, ph, started_at, latency_ms, "error")
        raise


# ---------------------------------------------------------------------------
# Bearer-token auth middleware (HTTP transport only)
# ---------------------------------------------------------------------------

class _BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != self._token:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)


# ---------------------------------------------------------------------------
# HTTP app factory
# ---------------------------------------------------------------------------

def _build_http_app(token: str) -> Starlette:
    session_manager = StreamableHTTPSessionManager(mcp, stateless=True)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):  # type: ignore[type-arg]
        async with session_manager.run():
            yield

    async def _asgi(scope: Any, receive: Any, send: Any) -> None:
        await session_manager.handle_request(scope, receive, send)

    app = Starlette(
        routes=[Mount("/mcp", app=_asgi)],
        lifespan=lifespan,
    )
    if token:
        app.add_middleware(_BearerAuthMiddleware, token=token)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-Agent MCP server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="'http' (default, port --port) or 'stdio' (Claude Desktop).",
    )
    parser.add_argument("--port", type=int, default=9004)
    args = parser.parse_args()

    if args.transport == "stdio":
        async def _run() -> None:
            async with stdio_server() as (r, w):
                await mcp.run(r, w, mcp.create_initialization_options())
        asyncio.run(_run())
    else:
        token = os.environ.get("MCP_RISK_AGENT_TOKEN", "")
        app = _build_http_app(token)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
