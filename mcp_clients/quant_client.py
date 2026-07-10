"""
Thin MCP-protocol client for the Quant Agent server.

All communication goes through the MCP streamable-HTTP transport — this
module does NOT import from ratios.py or any other quant-agent internal.
The orchestrator should only touch this file when it needs ratio data.

Public API
----------
compute_ratios(ticker)                         -> dict
get_ratio_history(ticker, periods)             -> list[dict]
compare_peers(ticker, peer_tickers)            -> dict

Exceptions
----------
QuantClientError  – any quant-agent tool error
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

load_dotenv()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuantClientError(Exception):
    """Generic quant-agent tool failure."""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_QUANT_AGENT_URL   = os.environ.get("MCP_QUANT_AGENT_URL",   "http://localhost:9002/mcp")
_QUANT_AGENT_TOKEN = os.environ.get("MCP_QUANT_AGENT_TOKEN", "")


@contextlib.asynccontextmanager
async def _open_session():
    headers: dict[str, str] = {}
    if _QUANT_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {_QUANT_AGENT_TOKEN}"
    async with streamablehttp_client(_QUANT_AGENT_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


def _check(result: CallToolResult) -> Any:
    if result.isError:
        msg = result.content[0].text if result.content else "Quant Agent returned an error"
        raise QuantClientError(msg)
    if not result.content:
        raise QuantClientError("Quant Agent returned an empty response")
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def compute_ratios(ticker: str) -> dict[str, Any]:
    """Compute financial ratios for *ticker* via the Quant Agent."""
    async with _open_session() as session:
        result = await session.call_tool("compute_ratios", {"ticker": ticker})
    return _check(result)


async def get_ratio_history(ticker: str, periods: int = 4) -> list[dict[str, Any]]:
    """Return D/E ratio for each of the last *periods* annual 10-K filings.

    Returns [{period_end: YYYY-MM-DD, debt_to_equity: float|None}, ...]
    sorted oldest-to-newest.
    """
    async with _open_session() as session:
        result = await session.call_tool(
            "get_ratio_history", {"ticker": ticker, "periods": periods}
        )
    return _check(result)


async def compare_peers(ticker: str, peer_tickers: list[str]) -> dict[str, Any]:
    """Compare *ticker* against *peer_tickers* on the same ratio set."""
    async with _open_session() as session:
        result = await session.call_tool(
            "compare_peers", {"ticker": ticker, "peer_tickers": peer_tickers}
        )
    return _check(result)
