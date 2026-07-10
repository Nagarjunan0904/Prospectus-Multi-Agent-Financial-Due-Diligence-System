"""
Sentiment Agent node — runs FinBERT over recent news headlines via the
Sentiment Agent MCP server.

Makes one MCP tool call per run (happy path):
  get_sentiment_summary -> state['sentiment']

Runs in parallel with quant_agent after data_agent completes.  The
operator.add reducer on errors / agent_trace handles the state merge.

Return contract
---------------
Returns ONLY the keys this node sets, with ONLY NEW items in
``errors`` / ``agent_trace``.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mcp_clients.sentiment_client import SentimentClientError, get_sentiment_summary
from backend.state import AgentTraceEntry, DueDiligenceState

_log = logging.getLogger(__name__)


def _append_trace(
    agent_trace: list[AgentTraceEntry],
    tool: str | None,
    status: str,
    latency_ms: float | None = None,
) -> None:
    entry: AgentTraceEntry = {"node": "sentiment_agent", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> dict[str, Any]:
    errors: list[str] = []                    # only NEW errors from this node
    agent_trace: list[AgentTraceEntry] = []   # only NEW trace entries from this node

    if "sentiment_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {"agent_trace": agent_trace}

    ticker: str = state["ticker"]

    # ── get_sentiment_summary ────────────────────────────────────────────────
    sentiment: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        sentiment = await get_sentiment_summary(ticker, days=14)
        _append_trace(agent_trace, "get_sentiment_summary", "success",
                      (time.monotonic() - t0) * 1000)
        _log.info(
            "sentiment_agent: get_sentiment_summary OK for %s "
            "(%d headlines, %.0f ms)",
            ticker,
            sentiment.get("headline_count", 0),
            (time.monotonic() - t0) * 1000,
        )
    except (SentimentClientError, Exception) as exc:
        _append_trace(agent_trace, "get_sentiment_summary", "error",
                      (time.monotonic() - t0) * 1000)
        errors.append(f"get_sentiment_summary: {exc}")
        _log.warning("sentiment_agent: get_sentiment_summary failed (%s) — %s", ticker, exc)

    return {
        "sentiment":   sentiment,
        "errors":      errors,
        "agent_trace": agent_trace,
    }
