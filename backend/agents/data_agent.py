"""
Data Agent node — fetches all SEC EDGAR data for a given company.

Makes three MCP tool calls per run (happy path):
  get_company_facts     -> state['company_facts']
  get_filing_sections   -> state['filing_sections']
  get_insider_transactions -> state['insider_summary']  (summary only)

Each call is traced individually.  Failures are soft: they append to
state['errors'] and leave the corresponding state field as an empty
dict, so downstream agents receive a defined (if empty) value.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mcp_clients.data_client import (
    EdgarAPIError,
    get_company_facts,
    get_filing_sections,
    get_insider_transactions,
    get_recent_filings,
)
from backend.state import AgentTraceEntry, DueDiligenceState

_log = logging.getLogger(__name__)


def _append_trace(
    agent_trace: list[AgentTraceEntry],
    tool: str | None,
    status: str,
    latency_ms: float | None = None,
) -> None:
    entry: AgentTraceEntry = {"node": "data_agent", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> DueDiligenceState:
    agent_trace: list[AgentTraceEntry] = list(state.get("agent_trace") or [])
    errors: list[str] = list(state.get("errors") or [])

    # ── Step 1: skip check ───────────────────────────────────────────────────
    if "data_agent" not in (state.get("required_agents") or []):
        _append_trace(agent_trace, None, "skipped")
        return {**state, "agent_trace": agent_trace}

    cik: str = state["cik"]  # type: ignore[assignment]  # guaranteed by supervisor

    # ── Step 2: company facts ────────────────────────────────────────────────
    company_facts: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        company_facts = await get_company_facts(cik)
        _append_trace(agent_trace, "get_company_facts", "success",
                      (time.monotonic() - t0) * 1000)
        _log.info("data_agent: get_company_facts OK for CIK %s", cik)
    except Exception as exc:
        _append_trace(agent_trace, "get_company_facts", "error",
                      (time.monotonic() - t0) * 1000)
        errors.append(f"get_company_facts: {exc}")
        _log.warning("data_agent: get_company_facts failed (CIK %s) — %s", cik, exc)

    # ── Step 3: determine which form type is available ───────────────────────
    form_type: str | None = None
    try:
        recent = await get_recent_filings(cik, form_types=["10-K", "10-Q"], limit=5)
        if any(f["form"] == "10-K" for f in recent):
            form_type = "10-K"
        elif any(f["form"] == "10-Q" for f in recent):
            form_type = "10-Q"
    except EdgarAPIError as exc:
        errors.append(f"get_recent_filings (form_type detection): {exc}")
        _log.warning("data_agent: form_type detection failed (CIK %s) — %s", cik, exc)
        # form_type stays None → step 4 skipped

    # ── Step 4: filing sections ──────────────────────────────────────────────
    filing_sections: dict[str, str] = {}
    if form_type is None:
        _append_trace(agent_trace, "get_filing_sections", "skipped")
        _log.info("data_agent: skipping get_filing_sections — no 10-K/10-Q for CIK %s", cik)
    else:
        t1 = time.monotonic()
        try:
            filing_sections = await get_filing_sections(
                cik, form_type, sections=["1A", "7"]
            )
            _append_trace(agent_trace, "get_filing_sections", "success",
                          (time.monotonic() - t1) * 1000)
            _log.info("data_agent: get_filing_sections OK (%s, CIK %s)", form_type, cik)
        except Exception as exc:
            _append_trace(agent_trace, "get_filing_sections", "error",
                          (time.monotonic() - t1) * 1000)
            errors.append(f"get_filing_sections: {exc}")
            _log.warning("data_agent: get_filing_sections failed — %s", exc)

    # ── Step 5: insider transactions (summary only) ──────────────────────────
    insider_summary: dict[str, Any] = {}
    t2 = time.monotonic()
    try:
        insider_data = await get_insider_transactions(cik, days=90)
        insider_summary = insider_data.get("summary", {})
        _append_trace(agent_trace, "get_insider_transactions", "success",
                      (time.monotonic() - t2) * 1000)
        _log.info("data_agent: get_insider_transactions OK for CIK %s", cik)
    except Exception as exc:
        _append_trace(agent_trace, "get_insider_transactions", "error",
                      (time.monotonic() - t2) * 1000)
        errors.append(f"get_insider_transactions: {exc}")
        _log.warning("data_agent: get_insider_transactions failed (CIK %s) — %s", cik, exc)

    return {
        **state,
        "company_facts": company_facts,
        "filing_sections": filing_sections,
        "insider_summary": insider_summary,
        "errors": errors,
        "agent_trace": agent_trace,
    }
