"""
Shared LangGraph state types for the due-diligence graph.

DueDiligenceState is a flat TypedDict — LangGraph nodes return a dict of
the keys they modified and the graph merges them in.  List fields (errors,
agent_trace) accumulate across nodes; the graph schema uses operator.add
reducers (configure when building the StateGraph, not here).
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentTraceEntry(TypedDict, total=False):
    node: str
    tool: str
    status: str          # 'success' | 'error'
    latency_ms: float    # absent when unknown (e.g. early error path)


class DueDiligenceState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    ticker: str

    # ── Set by supervisor ───────────────────────────────────────────────────
    cik: str | None
    required_agents: list[str]

    # ── Accumulated by every node ───────────────────────────────────────────
    errors: list[str]
    agent_trace: list[AgentTraceEntry]

    # ── Populated by each specialist agent ─────────────────────────────────
    recent_filings: list[dict[str, Any]]
    company_facts: dict[str, Any]
    filing_sections: dict[str, str]
    insider_transactions: dict[str, Any]
    quant_analysis: dict[str, Any]
    sentiment_analysis: dict[str, Any]
    risk_assessment: dict[str, Any]
