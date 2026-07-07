"""
Quant-Agent MCP server — financial ratio computation layer.

Transports
----------
HTTP (default)
    Streamable-HTTP on :9002/mcp, protected by MCP_QUANT_AGENT_TOKEN.
    Start with::

        python -m mcp_servers.quant_agent.server

stdio
    Local Claude Desktop testing.  Start with::

        python -m mcp_servers.quant_agent.server --transport stdio

Tools
-----
compute_ratios(ticker, cik=None)
    Fetches company_facts from the Data Agent (24 h-cached), fetches the
    latest close price from Stooq (15 min-cached), computes P/E, D/E,
    current ratio, margins, and revenue growth.  Every ratio is
    independently fault-tolerant: a missing XBRL concept → None + warning.

compare_peers(ticker, peer_tickers)
    Runs compute_ratios for every ticker concurrently, adds market_cap,
    returns a table sorted by market_cap descending.  Per-peer failures
    are soft: excluded from the table, never crash the call.

Auth note: same static bearer-token decision as data_agent; see that
module's docstring for the production OAuth 2.1 upgrade path.
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
from mcp_clients.data_client import get_company_facts as _get_company_facts
from mcp_clients.data_client import resolve_cik as _resolve_cik
from mcp_servers.quant_agent.tools.ratios import compare_peers, compute_ratios, get_ratio_history


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = Server("quant-agent")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="compute_ratios",
            description=(
                "Compute financial ratios for one company from EDGAR XBRL data. "
                "Fetches company_facts from the Data Agent (24 h-cached) and the "
                "latest close price from Stooq (15 min-cached). "
                "Returns P/E, debt/equity, current ratio, gross/operating/net margin, "
                "and revenue growth YoY and QoQ. "
                "Each ratio is computed independently — missing XBRL concepts → None "
                "with an explanation in 'warnings', never crashes the whole call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (used for Stooq price lookup).",
                    },
                    "cik": {
                        "type": "string",
                        "description": (
                            "10-digit zero-padded CIK if already known. "
                            "Skips a resolve_cik round-trip to the Data Agent."
                        ),
                    },
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="compare_peers",
            description=(
                "Compare a ticker against a peer group across the same ratio set. "
                "Fetches company_facts from the Data Agent for every ticker concurrently. "
                "Returns a table sorted by market_cap descending. "
                "Per-peer fetch failures are soft: the peer is excluded with a warning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "peer_tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of peer ticker symbols to compare against.",
                    },
                },
                "required": ["ticker", "peer_tickers"],
            },
        ),
        Tool(
            name="get_ratio_history",
            description=(
                "Return debt-to-equity ratio for each of the last N annual 10-K "
                "filing periods, sorted oldest-to-newest. "
                "Used by the Risk Agent to detect multi-period debt spikes. "
                "Returns [{period_end: YYYY-MM-DD, debt_to_equity: float|null}, ...]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol.",
                    },
                    "cik": {
                        "type": "string",
                        "description": "10-digit CIK if already known (skips resolve_cik).",
                    },
                    "periods": {
                        "type": "integer",
                        "default": 4,
                        "description": "Number of annual periods to return (default 4).",
                    },
                },
                "required": ["ticker"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Per-tool implementations
# ---------------------------------------------------------------------------

async def _impl_compute_ratios(ticker: str, cik: str | None = None) -> Any:
    """Resolve ticker → CIK (if needed), fetch company_facts, compute ratios."""
    resolved_cik = cik or await _resolve_cik(ticker)
    company_facts = await _get_company_facts(resolved_cik)
    return await compute_ratios(company_facts, ticker=ticker)


async def _impl_compare_peers(ticker: str, peer_tickers: list[str]) -> Any:
    return await compare_peers(ticker, peer_tickers)


async def _impl_get_ratio_history(
    ticker: str,
    cik: str | None = None,
    periods: int = 4,
) -> Any:
    resolved_cik = cik or await _resolve_cik(ticker)
    company_facts = await _get_company_facts(resolved_cik)
    return get_ratio_history(company_facts, periods=periods)


_TOOLS: dict[str, Any] = {
    "compute_ratios": _impl_compute_ratios,
    "compare_peers": _impl_compare_peers,
    "get_ratio_history": _impl_get_ratio_history,
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
        _audit_record, "quant", tool, ph, started_at, latency_ms, status
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
    parser = argparse.ArgumentParser(description="Quant-Agent MCP server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="'http' (default, port --port) or 'stdio' (Claude Desktop).",
    )
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    if args.transport == "stdio":
        async def _run() -> None:
            async with stdio_server() as (r, w):
                await mcp.run(r, w, mcp.create_initialization_options())
        asyncio.run(_run())
    else:
        token = os.environ.get("MCP_QUANT_AGENT_TOKEN", "")
        app = _build_http_app(token)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
