"""
Quant Agent node — stub implementation (Phase 2).

Real implementation (Phase 3) will compute pe_ratio, debt_to_equity, etc.
from state['company_facts'] using XBRL tag lookups.  The stub returns a
representative dict matching that eventual shape so the trace UI and
risk_agent receive the correct structure immediately.

Excluded automatically when the supervisor finds no 10-K/10-Q filings
(no XBRL-backed data → no ratio computation possible).
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
    entry: AgentTraceEntry = {"node": "quant_agent", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> DueDiligenceState:
    agent_trace: list[AgentTraceEntry] = list(state.get("agent_trace") or [])

    if "quant_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {**state, "agent_trace": agent_trace}

    t0 = time.monotonic()
    await asyncio.sleep(0.2)  # placeholder for real XBRL computation
    ratios = {
        "pe_ratio": 45.2,
        "price_to_sales": 18.7,
        "debt_to_equity": 0.18,
        "current_ratio": 4.17,
        "gross_margin": 0.57,
        "operating_margin": 0.34,
        "revenue_growth_yoy": 0.94,
        "operating_cash_flow_ratio": 1.21,
    }
    _append_trace(agent_trace, "stub", "success", (time.monotonic() - t0) * 1000)
    _log.info("quant_agent: stub ratios populated for %s", state.get("ticker"))

    return {**state, "ratios": ratios, "agent_trace": agent_trace}
