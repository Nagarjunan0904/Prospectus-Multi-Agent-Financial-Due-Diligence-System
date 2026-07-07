"""
Sentiment-Agent MCP server — FinBERT news sentiment layer.

Transports
----------
HTTP (default, strongly preferred)
    Streamable-HTTP on :9003/mcp, protected by MCP_SENTIMENT_AGENT_TOKEN.
    Start with::

        python -m mcp_servers.sentiment_agent.server

    The FinBERT model is loaded once at process start and stays warm
    in memory for the lifetime of the server.

stdio
    For local Claude Desktop testing::

        python -m mcp_servers.sentiment_agent.server --transport stdio

    IMPORTANT: stdio spawns a new process per tool call, which means the
    FinBERT model is reloaded each time (~1–2 s GPU / ~4–8 s CPU).
    Use HTTP transport for any production or latency-sensitive workflow.

Tools (exposed via MCP)
-----------------------
fetch_headlines(ticker, days=14)
    Raw Alpha Vantage headlines, 6-h cached.  Useful for inspection /
    debugging; the scoring pipeline is not run here.

get_sentiment_summary(ticker, days=14)
    FinBERT-scored headlines aggregated to a dict matching
    DueDiligenceState["sentiment"].  This is the tool that
    backend/agents/sentiment_agent.py will call in Phase 4.

get_sentiment_trend(ticker, days=30)
    Day-bucketed compound-score time series for the SentimentGauge chart.

score_sentiment is intentionally NOT exposed — raw per-text scores have
no value to an external caller without the aggregation context.
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
from mcp_servers.sentiment_agent.tools.finbert_scorer import (
    get_sentiment_summary,
    get_sentiment_trend,
)
from mcp_servers.sentiment_agent.tools.news_fetcher import fetch_headlines


# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = Server("sentiment-agent")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="fetch_headlines",
            description=(
                "Fetch recent news headlines for a ticker from Alpha Vantage "
                "(NEWS_SENTIMENT endpoint), 6-h cached to stay within the free-tier "
                "25-req/day budget. Returns raw title+summary text; Alpha Vantage's "
                "own sentiment scores are intentionally ignored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol."},
                    "days": {
                        "type": "integer",
                        "default": 14,
                        "description": "Days of history to return (default 14).",
                    },
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_sentiment_summary",
            description=(
                "Score recent headlines with ProsusAI/finbert and return an "
                "aggregated sentiment summary matching DueDiligenceState['sentiment']: "
                "{positive_pct, neutral_pct, negative_pct, headline_count, "
                "trend (daily compound scores oldest→newest)}. "
                "This is the primary tool for the due-diligence pipeline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "default": 14,
                        "description": "Days of news history to score (default 14).",
                    },
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_sentiment_trend",
            description=(
                "Return a day-by-day FinBERT compound-score time series "
                "for the SentimentGauge chart: "
                "[{date, compound_score, headline_count}, ...] sorted oldest→newest. "
                "Days with no headlines are omitted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "default": 30,
                        "description": "Days of history to bucket (default 30).",
                    },
                },
                "required": ["ticker"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Per-tool implementations
# ---------------------------------------------------------------------------

async def _impl_fetch_headlines(ticker: str, days: int = 14) -> Any:
    return await fetch_headlines(ticker, days)


async def _impl_get_sentiment_summary(ticker: str, days: int = 14) -> Any:
    return await get_sentiment_summary(ticker, days)


async def _impl_get_sentiment_trend(ticker: str, days: int = 30) -> Any:
    return await get_sentiment_trend(ticker, days)


_TOOLS: dict[str, Any] = {
    "fetch_headlines": _impl_fetch_headlines,
    "get_sentiment_summary": _impl_get_sentiment_summary,
    "get_sentiment_trend": _impl_get_sentiment_trend,
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
        _audit_record, "sentiment", tool, ph, started_at, latency_ms, status
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
    parser = argparse.ArgumentParser(description="Sentiment-Agent MCP server")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help=(
            "'http' (default, port --port) or 'stdio' (Claude Desktop). "
            "Note: stdio reloads FinBERT on every tool call — use http in production."
        ),
    )
    parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args()

    if args.transport == "stdio":
        async def _run() -> None:
            async with stdio_server() as (r, w):
                await mcp.run(r, w, mcp.create_initialization_options())
        asyncio.run(_run())
    else:
        token = os.environ.get("MCP_SENTIMENT_AGENT_TOKEN", "")
        app = _build_http_app(token)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
