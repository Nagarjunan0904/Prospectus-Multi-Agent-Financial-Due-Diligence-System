"""
Risk Agent node — stub implementation (Phase 2).

Real implementation (Phase 3) will synthesise state['ratios'],
state['sentiment'], and state['insider_summary'] into a structured list
of risk flags with severity scores and source attribution.  The stub
returns one representative flag matching that schema.
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
    entry: AgentTraceEntry = {"node": "risk_agent", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> DueDiligenceState:
    agent_trace: list[AgentTraceEntry] = list(state.get("agent_trace") or [])

    if "risk_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {**state, "agent_trace": agent_trace}

    t0 = time.monotonic()
    await asyncio.sleep(0.2)  # placeholder for real LLM synthesis
    risk_flags = [
        {
            "flag": "sample_flag",
            "severity": "low",
            "evidence": "stub data — no real analysis performed",
            "source_tool": "stub",
        }
    ]
    _append_trace(agent_trace, "stub", "success", (time.monotonic() - t0) * 1000)
    _log.info("risk_agent: stub risk_flags populated for %s", state.get("ticker"))

    return {**state, "risk_flags": risk_flags, "agent_trace": agent_trace}
