"""
Data-Agent MCP server — SEC EDGAR data layer.

Transports
----------
HTTP (default)
    Streamable-HTTP on :9001/mcp, protected by a static Bearer token read
    from MCP_DATA_AGENT_TOKEN.  Start with::

        python -m mcp_servers.data_agent.server

stdio
    Local Claude Desktop testing — no network, no auth.  Start with::

        python -m mcp_servers.data_agent.server --transport stdio

Auth design note (portfolio scope — raise this in interviews)
-------------------------------------------------------------
The current bearer-token check is an intentional portfolio-scope decision,
not an oversight.  Production deployment replaces it with OAuth 2.1:
  • RFC 8414 discovery endpoint so clients can auto-discover the auth server
  • Short-lived, scoped access tokens (e.g. read:edgar) per MCP spec §4.3
  • Token introspection / JWKS verification on every request

Adding full OIDC plumbing is ~300 lines of boilerplate that obscures the
agent architecture being demonstrated here, so it was deferred deliberately.
The middleware injection point (``_BearerAuthMiddleware``) is already in the
right place for a production swap.
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
from mcp_servers.data_agent.tools.edgar_client import (
    get_company_facts as _edgar_company_facts,
    get_recent_filings as _edgar_recent_filings,
    resolve_cik,
)
from mcp_servers.data_agent.tools.filing_sections import (
    get_mdna,
    get_risk_factors,
)
from mcp_servers.data_agent.tools.insider_filings import (
    get_insider_transactions as _get_insider_transactions,
)


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = Server("data-agent")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="resolve_cik",
            description=(
                "Resolve a ticker symbol to its 10-digit SEC EDGAR CIK string. "
                "Call this once per session to obtain the CIK before calling "
                "tools that accept a 'cik' parameter directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'NVDA'.",
                    },
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_company_facts",
            description=(
                "Return all us-gaap XBRL concepts for a ticker with historical "
                "values by fiscal period.  Use this for quantitative analysis: "
                "revenue, net income, EPS, debt levels, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL'.",
                    },
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_recent_filings",
            description=(
                "List recent SEC filings, optionally filtered by form type. "
                "Accepts either 'ticker' (resolved internally) or 'cik' (10-digit "
                "zero-padded string, skips an extra resolve step). "
                "Returns accession numbers, dates, and primary document paths."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker, e.g. 'AAPL'. Use this or 'cik', not both.",
                    },
                    "cik": {
                        "type": "string",
                        "description": "10-digit zero-padded CIK, e.g. '0001045810'. Use this or 'ticker'.",
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "e.g. ['10-K', '10-Q'].  Omit for all forms.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of filings to return.",
                    },
                },
            },
        ),
        Tool(
            name="get_filing_sections",
            description=(
                "Extract narrative sections from a company's most recent filing "
                "of a given form type.  Returns the raw text (≤ ~8 000 tokens per "
                "section) suitable for LLM analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "form_type": {
                        "type": "string",
                        "default": "10-K",
                        "description": "SEC form type, e.g. '10-K'.",
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["1A", "7"],
                        },
                        "description": (
                            "'1A' = Item 1A Risk Factors, "
                            "'7' = Item 7 MD&A.  Both may be requested together."
                        ),
                    },
                },
                "required": ["ticker", "sections"],
            },
        ),
        Tool(
            name="get_insider_transactions",
            description=(
                "Return Form 4 insider transactions filed within the last N days, "
                "plus a buy/sell summary the Risk Agent consumes directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "default": 90,
                        "description": "Lookback window in days (default 90).",
                    },
                },
                "required": ["ticker"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Per-tool implementations
# ---------------------------------------------------------------------------

async def _resolve(ticker: str) -> str:
    """Wrapper so unknown-ticker ValueError bubbles up clearly."""
    return await resolve_cik(ticker.upper().strip())


async def _impl_resolve_cik(ticker: str) -> str:
    return await _resolve(ticker)


async def _impl_get_company_facts(ticker: str) -> Any:
    cik = await _resolve(ticker)
    return await _edgar_company_facts(cik)


async def _impl_get_recent_filings(
    ticker: str | None = None,
    cik: str | None = None,
    form_types: list[str] | None = None,
    limit: int = 10,
) -> Any:
    if cik is None and ticker is None:
        raise ValueError("Either 'ticker' or 'cik' must be provided.")
    if cik is None:
        cik = await _resolve(ticker)  # type: ignore[arg-type]
    return await _edgar_recent_filings(cik, form_types, limit)


async def _impl_get_filing_sections(
    ticker: str,
    form_type: str = "10-K",
    sections: list[str] | None = None,
) -> Any:
    cik = await _resolve(ticker)

    filings = await _edgar_recent_filings(cik, [form_type], limit=1)
    if not filings:
        raise ValueError(f"No {form_type!r} filings found for {ticker!r}")

    accession = filings[0]["accessionNumber"]
    result: dict[str, str] = {}

    for sec in sections or []:
        if sec == "1A":
            result["1A"] = await get_risk_factors(cik, accession)
        elif sec == "7":
            result["7"] = await get_mdna(cik, accession)
        else:
            raise ValueError(
                f"Section {sec!r} is not implemented. "
                "Supported values: '1A' (Item 1A Risk Factors), '7' (Item 7 MD&A)."
            )

    return result


async def _impl_get_insider_transactions(ticker: str, days: int = 90) -> Any:
    cik = await _resolve(ticker)
    return await _get_insider_transactions(cik, days)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_TOOLS: dict[str, Any] = {
    "resolve_cik": _impl_resolve_cik,
    "get_company_facts": _impl_get_company_facts,
    "get_recent_filings": _impl_get_recent_filings,
    "get_filing_sections": _impl_get_filing_sections,
    "get_insider_transactions": _impl_get_insider_transactions,
}


# ---------------------------------------------------------------------------
# call_tool handler with audit logging
# ---------------------------------------------------------------------------

async def _audit(
    tool: str,
    ph: str,
    started_at: datetime,
    latency_ms: float,
    status: str,
) -> None:
    await asyncio.to_thread(
        _audit_record, "data", tool, ph, started_at, latency_ms, status
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
        raise  # mcp SDK (raise_exceptions=False default) returns isError=True to client


# ---------------------------------------------------------------------------
# Bearer-token auth middleware  (HTTP transport only)
# ---------------------------------------------------------------------------

class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Static bearer-token gate for local/dev deployments.

    Production swap: replace with OAuth 2.1 token introspection.
    See module docstring for the full rationale.
    """

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
    parser = argparse.ArgumentParser(description="Data-Agent MCP server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help=(
            "'http' (default): streamable-HTTP on --port, requires "
            "MCP_DATA_AGENT_TOKEN env var.  "
            "'stdio': for local Claude Desktop — no auth, no network."
        ),
    )
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    if args.transport == "stdio":
        async def _run() -> None:
            async with stdio_server() as (r, w):
                await mcp.run(r, w, mcp.create_initialization_options())

        asyncio.run(_run())
    else:
        token = os.environ.get("MCP_DATA_AGENT_TOKEN", "")
        app = _build_http_app(token)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
