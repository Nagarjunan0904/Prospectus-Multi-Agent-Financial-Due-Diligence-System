"""
Shared LangGraph state types for the due-diligence graph.

DueDiligenceState is a flat TypedDict — LangGraph nodes return a partial
dict of only the keys they modify and the graph merges them in.

Accumulator fields
------------------
``errors`` and ``agent_trace`` use ``Annotated[list, operator.add]`` so
that LangGraph applies list-concatenation when merging updates from
parallel branches (quant_agent and sentiment_agent run in parallel after
data_agent).

Node return contract
--------------------
Every node MUST return only the keys it modified, and MUST include only
NEW items in ``errors`` / ``agent_trace`` — not the full accumulated list.
The Annotated reducer handles concatenation; returning the full list
would double earlier entries.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentTraceEntry(TypedDict, total=False):
    node: str
    tool: str | None     # None on 'skipped' entries where no tool was called
    status: str          # 'success' | 'error' | 'skipped'
    latency_ms: float    # absent when unknown (e.g. early error path)


class DueDiligenceState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    ticker: str

    # ── Set by supervisor ───────────────────────────────────────────────────
    cik: str | None
    required_agents: list[str]

    # ── Accumulated by every node (operator.add merges parallel updates) ────
    errors: Annotated[list[str], operator.add]
    agent_trace: Annotated[list[AgentTraceEntry], operator.add]

    # ── Populated by data_agent ─────────────────────────────────────────────
    company_facts: dict[str, Any]
    filing_sections: dict[str, str]        # {'1A': <text>, '7': <text>}
    insider_summary: dict[str, Any]        # summary sub-object only (not full txn list)

    # ── Populated by quant_agent ────────────────────────────────────────────
    ratios: dict[str, Any]                 # compute_ratios output
    ratio_history: list[dict[str, Any]]    # get_ratio_history output; consumed by risk_agent

    # ── Populated by sentiment_agent ────────────────────────────────────────
    sentiment: dict[str, Any]

    # ── Populated by risk_agent ─────────────────────────────────────────────
    risk_flags: list[dict[str, Any]]
