"""
Thin MCP-protocol client for the Sentiment Agent server.

All communication goes through the MCP streamable-HTTP transport — this
module does NOT import from news_fetcher.py or finbert_scorer.py.

Public API
----------
get_sentiment_summary(ticker, days)   -> dict
get_sentiment_trend(ticker, days)     -> list[dict]

Exceptions
----------
SentimentClientError  – any sentiment-agent tool error
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SentimentClientError(Exception):
    """Generic sentiment-agent tool failure."""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_SENTIMENT_AGENT_URL   = os.environ.get("MCP_SENTIMENT_AGENT_URL",   "http://localhost:9003/mcp")
_SENTIMENT_AGENT_TOKEN = os.environ.get("MCP_SENTIMENT_AGENT_TOKEN", "")


@contextlib.asynccontextmanager
async def _open_session():
    headers: dict[str, str] = {}
    if _SENTIMENT_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {_SENTIMENT_AGENT_TOKEN}"
    async with streamablehttp_client(_SENTIMENT_AGENT_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


def _check(result: CallToolResult) -> Any:
    if result.isError:
        msg = result.content[0].text if result.content else "Sentiment Agent returned an error"
        raise SentimentClientError(msg)
    if not result.content:
        raise SentimentClientError("Sentiment Agent returned an empty response")
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def get_sentiment_summary(ticker: str, days: int = 14) -> dict[str, Any]:
    """Return FinBERT sentiment summary for recent headlines about *ticker*.

    Returns {positive_pct, neutral_pct, negative_pct, headline_count, trend}.
    """
    async with _open_session() as session:
        result = await session.call_tool(
            "get_sentiment_summary", {"ticker": ticker, "days": days}
        )
    return _check(result)


async def get_sentiment_trend(ticker: str, days: int = 30) -> list[dict[str, Any]]:
    """Return daily compound sentiment scores for *ticker* over *days* days.

    Returns [{date: YYYY-MM-DD, compound_score: float, headline_count: int}, ...]
    sorted oldest-to-newest.  Days with zero headlines are omitted.
    """
    async with _open_session() as session:
        result = await session.call_tool(
            "get_sentiment_trend", {"ticker": ticker, "days": days}
        )
    return _check(result)
