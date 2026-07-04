"""
Supervisor node for the due-diligence LangGraph graph.

Responsibilities
----------------
1. Resolve ticker → CIK via the Data Agent MCP client.
   Failure (unknown ticker) → populate state['errors'], stop early.
2. Fetch recent 10-K / 10-Q filings to decide which specialist agents
   are meaningful to run (quant analysis requires XBRL-backed filings).
3. Populate state['cik'] and state['required_agents'].
4. Record every tool invocation (success and failure) as AgentTraceEntry
   in state['agent_trace'].

This node intentionally makes NO direct imports from edgar_client.py or any
other data-agent internal — all SEC data goes through the MCP client.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from mcp_clients.data_client import EdgarAPIError, get_recent_filings, resolve_cik
from backend.state import AgentTraceEntry, DueDiligenceState

_log = logging.getLogger(__name__)

_ALL_AGENTS = ["data_agent", "quant_agent", "sentiment_agent", "risk_agent"]


def _append_trace(
    agent_trace: list[AgentTraceEntry],
    node: str,
    tool: str,
    status: str,
    latency_ms: float | None = None,
) -> None:
    entry: AgentTraceEntry = {"node": node, "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> DueDiligenceState:
    """LangGraph node: resolve ticker, gate quant agent, populate required_agents."""
    ticker: str = state["ticker"]
    errors: list[str] = list(state.get("errors") or [])
    agent_trace: list[AgentTraceEntry] = list(state.get("agent_trace") or [])

    # ── Step 1: resolve ticker → CIK ────────────────────────────────────────
    t0 = time.monotonic()
    try:
        cik: str | None = await resolve_cik(ticker)
        latency_ms = (time.monotonic() - t0) * 1000
        _append_trace(agent_trace, "supervisor", "resolve_cik", "success", latency_ms)
        _log.info("supervisor: resolved %s → CIK %s (%.0f ms)", ticker, cik, latency_ms)
    except ValueError as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        _append_trace(agent_trace, "supervisor", "resolve_cik", "error", latency_ms)
        errors.append(str(exc))
        _log.warning("supervisor: unknown ticker %r — %s", ticker, exc)
        return {
            **state,
            "cik": None,
            "required_agents": [],
            "errors": errors,
            "agent_trace": agent_trace,
        }

    # ── Step 2 / 3: fetch recent filings to decide required_agents ───────────
    required_agents: list[str] = list(_ALL_AGENTS)

    t1 = time.monotonic()
    filings: list[dict[str, Any]] = []
    filings_ok = True
    try:
        filings = await get_recent_filings(
            cik, form_types=["10-K", "10-Q"], limit=5
        )
        latency_ms = (time.monotonic() - t1) * 1000
        _append_trace(agent_trace, "supervisor", "get_recent_filings", "success", latency_ms)
        _log.info(
            "supervisor: %s has %d 10-K/10-Q filings (%.0f ms)",
            ticker,
            len(filings),
            latency_ms,
        )
    except EdgarAPIError as exc:
        latency_ms = (time.monotonic() - t1) * 1000
        _append_trace(agent_trace, "supervisor", "get_recent_filings", "error", latency_ms)
        errors.append(f"get_recent_filings: {exc}")
        filings_ok = False
        _log.warning(
            "supervisor: get_recent_filings failed for %s (CIK %s) — %s",
            ticker,
            cik,
            exc,
        )

    # ── Step 4: gate quant_agent ─────────────────────────────────────────────
    # Quant analysis requires at least one 10-K or 10-Q with XBRL-backed data.
    # If the filing fetch errored OR returned nothing, exclude quant_agent so
    # the graph doesn't run a node that will inevitably produce empty output.
    if not filings_ok or not filings:
        required_agents = [a for a in required_agents if a != "quant_agent"]
        _log.info(
            "supervisor: quant_agent excluded — no eligible filings for %s", ticker
        )

    return {
        **state,
        "cik": cik,
        "required_agents": required_agents,
        "errors": errors,
        "agent_trace": agent_trace,
    }
