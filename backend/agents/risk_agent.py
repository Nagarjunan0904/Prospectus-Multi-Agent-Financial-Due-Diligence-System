"""
Risk Agent node — synthesises quant and text data into structured risk flags
via the Risk Agent MCP server.

Makes one MCP tool call per run (happy path):
  run_all_checks -> state['risk_flags']

Requires:
  state['cik']           — set by supervisor
  state['ratio_history'] — set by quant_agent (debt-spike detection)
  state['filing_sections']['7']  — MD&A text (audit-language scan)
  state['filing_sections']['1A'] — Risk Factors text (audit-language scan)

Return contract
---------------
Returns ONLY the keys this node sets, with ONLY NEW items in
``errors`` / ``agent_trace``.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mcp_clients.risk_client import RiskClientError, run_all_checks
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


async def run(state: DueDiligenceState) -> dict[str, Any]:
    errors: list[str] = []                    # only NEW errors from this node
    agent_trace: list[AgentTraceEntry] = []   # only NEW trace entries from this node

    if "risk_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {"agent_trace": agent_trace}

    cik: str = state["cik"]  # type: ignore[assignment]
    ratio_history: list[dict[str, Any]] = state.get("ratio_history") or []
    filing_sections: dict[str, str] = state.get("filing_sections") or {}
    mdna_text: str        = filing_sections.get("7",  "")
    risk_factors_text: str = filing_sections.get("1A", "")

    # ── run_all_checks ───────────────────────────────────────────────────────
    risk_flags: list[dict[str, Any]] = []
    t0 = time.monotonic()
    try:
        risk_flags = await run_all_checks(
            cik=cik,
            ratio_history=ratio_history,
            mdna_text=mdna_text,
            risk_factors_text=risk_factors_text,
            days=90,
        )
        _append_trace(agent_trace, "run_all_checks", "success",
                      (time.monotonic() - t0) * 1000)
        _log.info(
            "risk_agent: run_all_checks OK for CIK %s — %d flag(s) (%.0f ms)",
            cik, len(risk_flags), (time.monotonic() - t0) * 1000,
        )
    except (RiskClientError, Exception) as exc:
        _append_trace(agent_trace, "run_all_checks", "error",
                      (time.monotonic() - t0) * 1000)
        errors.append(f"run_all_checks: {exc}")
        _log.warning("risk_agent: run_all_checks failed (CIK %s) — %s", cik, exc)

    return {
        "risk_flags":  risk_flags,
        "errors":      errors,
        "agent_trace": agent_trace,
    }
