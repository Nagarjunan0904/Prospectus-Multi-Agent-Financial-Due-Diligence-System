"""
Thin MCP-protocol client for the Risk Agent server.

All communication goes through the MCP streamable-HTTP transport — this
module does NOT import from red_flags.py or any other risk-agent internal.

Public API
----------
run_all_checks(cik, ratio_history, mdna_text, risk_factors_text, days)
    -> list[dict]

Exceptions
----------
RiskClientError  – any risk-agent tool error
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

class RiskClientError(Exception):
    """Generic risk-agent tool failure."""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_RISK_AGENT_URL   = os.environ.get("MCP_RISK_AGENT_URL",   "http://localhost:9004/mcp")
_RISK_AGENT_TOKEN = os.environ.get("MCP_RISK_AGENT_TOKEN", "")


@contextlib.asynccontextmanager
async def _open_session():
    headers: dict[str, str] = {}
    if _RISK_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {_RISK_AGENT_TOKEN}"
    async with streamablehttp_client(_RISK_AGENT_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


def _check(result: CallToolResult) -> Any:
    if result.isError:
        msg = result.content[0].text if result.content else "Risk Agent returned an error"
        raise RiskClientError(msg)
    if not result.content:
        raise RiskClientError("Risk Agent returned an empty response")
    return json.loads(result.content[0].text)


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def run_all_checks(
    cik: str,
    ratio_history: list[dict[str, Any]],
    mdna_text: str,
    risk_factors_text: str,
    days: int = 90,
) -> list[dict[str, Any]]:
    """Run all risk detectors for *cik* and return a flat list of flags.

    Parameters
    ----------
    cik:
        10-digit zero-padded SEC EDGAR CIK.
    ratio_history:
        Output of quant_client.get_ratio_history — used for debt-spike detection.
    mdna_text:
        Item 7 (MD&A) text from the most recent 10-K filing.
    risk_factors_text:
        Item 1A (Risk Factors) text from the most recent 10-K filing.
    days:
        Look-back window for insider-selling detection (default 90).

    Returns
    -------
    list[dict]
        [{flag, severity, evidence, source_tool}, ...] — empty list if no flags.
    """
    async with _open_session() as session:
        result = await session.call_tool(
            "run_all_checks",
            {
                "cik":               cik,
                "ratio_history":     ratio_history,
                "mdna_text":         mdna_text,
                "risk_factors_text": risk_factors_text,
                "days":              days,
            },
        )
    return _check(result)
