"""
Sentiment Agent node — stub implementation (Phase 2).

Real implementation (Phase 3) will run FinBERT over the 10-K Risk Factors
and MD&A text from state['filing_sections'] plus recent news headlines.
The stub returns a representative dict matching that shape.
"""
from __future__ import annotations

import asyncio
import logging
import time

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


async def run(state: DueDiligenceState) -> DueDiligenceState:
    agent_trace: list[AgentTraceEntry] = list(state.get("agent_trace") or [])

    if "sentiment_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {**state, "agent_trace": agent_trace}

    t0 = time.monotonic()
    await asyncio.sleep(0.2)  # placeholder for real FinBERT inference
    sentiment = {
        "positive_pct": 0.72,
        "neutral_pct": 0.19,
        "negative_pct": 0.09,
        "headline_count": 18,
        # trend: per-quarter sentiment score, newest last
        "trend": [0.65, 0.70, 0.68, 0.72],
        "dominant_themes": ["AI infrastructure", "data-centre demand", "margin expansion"],
    }
    _append_trace(agent_trace, "stub", "success", (time.monotonic() - t0) * 1000)
    _log.info("sentiment_agent: stub sentiment populated for %s", state.get("ticker"))

    return {**state, "sentiment": sentiment, "agent_trace": agent_trace}
