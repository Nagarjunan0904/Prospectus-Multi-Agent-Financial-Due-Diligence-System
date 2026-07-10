"""
Thin MCP-protocol client for the Data Agent server.

All communication goes through the MCP streamable-HTTP transport — this
module does NOT import from edgar_client.py or any other data-agent
internal.  The orchestrator (supervisor, etc.) should only touch this file
when it needs SEC data.

Public API
----------
resolve_cik(ticker)                            -> str
get_recent_filings(cik, form_types, limit)     -> list[dict]
get_company_facts(cik)                         -> dict
get_filing_sections(cik, form_type, sections)  -> dict
get_insider_transactions(cik, days)            -> dict

Exceptions
----------
ValueError      – unknown ticker (mirrors the server-side behaviour)
EdgarAPIError   – SEC API failure (HTTP 4xx/5xx from EDGAR)
DataClientError – any other data-agent tool error not covered above
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
# Exceptions (re-exported so callers never touch edgar_client directly)
# ---------------------------------------------------------------------------

class DataClientError(Exception):
    """Generic data-agent tool failure."""


class EdgarAPIError(DataClientError):
    """SEC EDGAR returned a non-200 response (rate limit, outage, bad CIK, …)."""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_DATA_AGENT_URL = os.environ.get("MCP_DATA_AGENT_URL", "http://localhost:9001/mcp")
_DATA_AGENT_TOKEN = os.environ.get("MCP_DATA_AGENT_TOKEN", "")


@contextlib.asynccontextmanager
async def _open_session():
    """Open an MCP client session to the Data Agent server."""
    headers: dict[str, str] = {}
    if _DATA_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {_DATA_AGENT_TOKEN}"

    async with streamablehttp_client(_DATA_AGENT_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


def _check(result: CallToolResult, error_cls: type[Exception] = DataClientError) -> Any:
    """Raise *error_cls* if the MCP result carries isError=True; else return parsed JSON."""
    if result.isError:
        msg = result.content[0].text if result.content else "Data Agent returned an error"
        raise error_cls(msg)
    if not result.content:
        raise DataClientError("Data Agent returned an empty response")
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def resolve_cik(ticker: str) -> str:
    """Return the 10-digit zero-padded SEC EDGAR CIK for *ticker*.

    Raises
    ------
    ValueError
        If the ticker is not found in SEC EDGAR's company_tickers.json.
    """
    async with _open_session() as session:
        result = await session.call_tool("resolve_cik", {"ticker": ticker})

    if result.isError:
        msg = result.content[0].text if result.content else f"Unknown ticker: {ticker!r}"
        raise ValueError(msg)

    return json.loads(result.content[0].text)  # server returns a JSON string e.g. '"0001045810"'


async def get_recent_filings(
    cik: str,
    form_types: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the most recent *limit* SEC filings for *cik*.

    Raises
    ------
    EdgarAPIError
        If the Data Agent cannot reach EDGAR (HTTP errors, timeouts, etc.).
    """
    args: dict[str, Any] = {"cik": cik, "limit": limit}
    if form_types is not None:
        args["form_types"] = form_types

    async with _open_session() as session:
        result = await session.call_tool("get_recent_filings", args)

    return _check(result, error_cls=EdgarAPIError)


async def get_company_facts(cik: str) -> dict[str, Any]:
    """Return all XBRL us-gaap facts for *cik*."""
    async with _open_session() as session:
        result = await session.call_tool("get_company_facts", {"cik": cik})
    return _check(result, error_cls=EdgarAPIError)


async def get_filing_sections(
    cik: str,
    form_type: str = "10-K",
    sections: list[str] | None = None,
) -> dict[str, str]:
    """Return narrative sections from the most recent *form_type* filing for *cik*."""
    args: dict[str, Any] = {"cik": cik, "form_type": form_type}
    if sections is not None:
        args["sections"] = sections
    async with _open_session() as session:
        result = await session.call_tool("get_filing_sections", args)
    return _check(result, error_cls=EdgarAPIError)


async def get_insider_transactions(
    cik: str,
    days: int = 90,
) -> dict[str, Any]:
    """Return Form 4 insider transactions and buy/sell summary for *cik*."""
    async with _open_session() as session:
        result = await session.call_tool(
            "get_insider_transactions", {"cik": cik, "days": days}
        )
    return _check(result, error_cls=EdgarAPIError)
