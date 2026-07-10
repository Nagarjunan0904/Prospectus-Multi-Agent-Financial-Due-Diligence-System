"""
Quant Agent node — computes financial ratios via the Quant Agent MCP server.

Makes two MCP tool calls per run (happy path):
  compute_ratios    -> state['ratios']
  get_ratio_history -> state['ratio_history']  (consumed by risk_agent)

Each call is traced individually.  Failures are soft: they append to
state['errors'] and leave the corresponding state field empty, so
downstream agents receive a defined (if empty) value.

Return contract
---------------
Returns ONLY the keys this node sets, with ONLY NEW items in
``errors`` / ``agent_trace``.  The graph's operator.add reducer merges
these with updates from the parallel sentinel_agent branch.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mcp_clients.quant_client import QuantClientError, compute_ratios, get_ratio_history
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


async def run(state: DueDiligenceState) -> dict[str, Any]:
    errors: list[str] = []                    # only NEW errors from this node
    agent_trace: list[AgentTraceEntry] = []   # only NEW trace entries from this node

    if "quant_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {"agent_trace": agent_trace}

    ticker: str = state["ticker"]

    # ── compute_ratios ───────────────────────────────────────────────────────
    ratios: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        ratios = await compute_ratios(ticker)
        _append_trace(agent_trace, "compute_ratios", "success",
                      (time.monotonic() - t0) * 1000)
        _log.info("quant_agent: compute_ratios OK for %s", ticker)
    except (QuantClientError, Exception) as exc:
        _append_trace(agent_trace, "compute_ratios", "error",
                      (time.monotonic() - t0) * 1000)
        errors.append(f"compute_ratios: {exc}")
        _log.warning("quant_agent: compute_ratios failed (%s) — %s", ticker, exc)

    # ── get_ratio_history ────────────────────────────────────────────────────
    ratio_history: list[dict[str, Any]] = []
    t1 = time.monotonic()
    try:
        ratio_history = await get_ratio_history(ticker, periods=4)
        _append_trace(agent_trace, "get_ratio_history", "success",
                      (time.monotonic() - t1) * 1000)
        _log.info(
            "quant_agent: get_ratio_history OK for %s (%d periods)",
            ticker, len(ratio_history),
        )
    except (QuantClientError, Exception) as exc:
        _append_trace(agent_trace, "get_ratio_history", "error",
                      (time.monotonic() - t1) * 1000)
        errors.append(f"get_ratio_history: {exc}")
        _log.warning("quant_agent: get_ratio_history failed (%s) — %s", ticker, exc)

    return {
        "ratios":        ratios,
        "ratio_history": ratio_history,
        "errors":        errors,
        "agent_trace":   agent_trace,
    }
