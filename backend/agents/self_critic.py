"""
Self-Critic node — evaluates the InvestmentMemo produced by memo_writer
for two quality dimensions:

  Groundedness
    For each Claim, does ``source_field`` resolve to an existing value in
    the pipeline state?  A claim is grounded if the path resolves without
    raising KeyError/AttributeError — even if the resolved value is an
    empty container (``[]``, ``{}``) or ``""``.  An empty container is
    real pipeline data (e.g. ``risk_flags = []`` means "no flags found",
    which is accurate and citable), not a missing field.
    Only a missing key or ``None`` constitutes ungrounded.
    Ungrounded claims are tracked per-agent so the retry logic can re-run
    whichever upstream agent produced the most fictitious data.

  Completeness
    All four required section headings must be present with ≥ 1 claim each.
    Completeness = (present headings with ≥ 1 claim) / 4.

Retry routing
-------------
``retry_agent`` is None when the memo is acceptable.  When a retry is
needed, it is set to:
  • the upstream agent with the most ungrounded claims
    (ties broken by dict iteration order = insertion order in Python 3.7+);
  • "memo_writer" when completeness < 1.0 but ALL claims are grounded
    (the problem is the LLM's output structure, not the upstream data).

Skip condition
--------------
If ``memo`` is None AND ``company_facts`` is populated, memo_writer
must have errored — send it back for a retry immediately.
If ``memo`` is None AND ``company_facts`` is also absent, the pipeline
upstream failed entirely; skip and set retry_agent = None (no point
retrying a memo when there is no data).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from backend.state import AgentTraceEntry, DueDiligenceState

_log = logging.getLogger(__name__)

# Maps the first path segment of a Claim.source_field to the responsible agent.
_STATE_FIELD_TO_AGENT: dict[str, str] = {
    "ratios":          "quant_agent",
    "ratio_history":   "quant_agent",
    "sentiment":       "sentiment_agent",
    "risk_flags":      "risk_agent",
    "company_facts":   "data_agent",
    "filing_sections": "data_agent",
    "insider_summary": "data_agent",
}

_REQUIRED_HEADINGS: frozenset[str] = frozenset(
    {"Financial Snapshot", "Sentiment", "Risk Factors", "Recommendation"}
)

_COVERAGE_THRESHOLD = 1.0   # all four headings must be present with ≥1 claim


# ---------------------------------------------------------------------------
# Grounding helpers
# ---------------------------------------------------------------------------

def _claim_is_grounded(claim: dict[str, Any], state: DueDiligenceState) -> bool:
    """Return True iff the claim's source_field path exists in state and is not None.

    Walk logic:
    - Split source_field on '.'
    - First segment: state[segment] (dict key access)
    - Subsequent segments: try obj[segment] first; fall back to getattr(obj, segment)
    - Return False on any KeyError, AttributeError, TypeError, IndexError, ValueError
    - Return False if ANY intermediate or final value is None

    Empty containers ([], {}, "") are considered grounded — they are real,
    existing pipeline values that a claim can accurately cite (e.g. "no risk
    flags found" citing risk_flags=[]).  Only a missing key or a None value
    means the claim references something that doesn't exist.
    """
    source_field: str = claim.get("source_field", "")
    if not source_field:
        return False

    parts = source_field.split(".")
    try:
        obj: Any = state[parts[0]]  # type: ignore[literal-required]
        if obj is None:
            return False
        for part in parts[1:]:
            try:
                obj = obj[part]
            except (KeyError, TypeError, IndexError):
                obj = getattr(obj, part)
            if obj is None:
                return False
    except (KeyError, AttributeError, TypeError, IndexError, ValueError):
        return False

    return True


def _agent_for_field(source_field: str) -> str:
    """Return the agent name for the first path segment, or 'unknown'."""
    first = source_field.split(".")[0] if source_field else ""
    return _STATE_FIELD_TO_AGENT.get(first, "unknown")


# ---------------------------------------------------------------------------
# Completeness helper
# ---------------------------------------------------------------------------

def _section_completeness(memo: dict[str, Any]) -> float:
    """Fraction of the four required headings present with ≥ 1 claim each."""
    sections: list[dict[str, Any]] = memo.get("sections") or []
    present: set[str] = {
        s["heading"]
        for s in sections
        if s.get("heading") in _REQUIRED_HEADINGS and len(s.get("claims") or []) >= 1
    }
    return len(present) / 4.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _append_trace(
    agent_trace: list[AgentTraceEntry],
    tool: str | None,
    status: str,
    latency_ms: float | None = None,
) -> None:
    entry: AgentTraceEntry = {"node": "self_critic", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> dict[str, Any]:
    """LangGraph node: evaluate memo groundedness and completeness."""
    errors: list[str] = []
    agent_trace: list[AgentTraceEntry] = []

    memo = state.get("memo")

    # ── Handle missing memo ──────────────────────────────────────────────────
    if memo is None:
        if state.get("company_facts"):
            # Data was available but memo_writer errored — send it back.
            # Increment retry_count so _route_self_critic can enforce the cap
            # even on this fast path (memo never produced, not just low quality).
            new_count = state.get("retry_count", 0) + 1
            _append_trace(agent_trace, None, "retry-memo_writer")
            _log.info("self_critic: memo is None but company_facts present — retrying memo_writer")
            return {
                "retry_agent": "memo_writer",
                "retry_count": new_count,
                "agent_trace": agent_trace,
                "errors":      errors,
            }
        else:
            # Upstream data entirely absent — nothing to evaluate
            _append_trace(agent_trace, None, "skipped")
            _log.info("self_critic: skipping — memo and company_facts both absent")
            return {
                "retry_agent": None,
                "agent_trace": agent_trace,
            }

    t0 = time.monotonic()

    # ── Groundedness check ───────────────────────────────────────────────────
    ungrounded_by_agent: dict[str, int] = {}

    sections: list[dict[str, Any]] = memo.get("sections") or []
    total_claims = 0
    ungrounded_total = 0

    for section in sections:
        for claim in section.get("claims") or []:
            total_claims += 1
            if not _claim_is_grounded(claim, state):
                ungrounded_total += 1
                agent = _agent_for_field(claim.get("source_field", ""))
                ungrounded_by_agent[agent] = ungrounded_by_agent.get(agent, 0) + 1

    # ── Completeness check ───────────────────────────────────────────────────
    completeness = _section_completeness(memo)

    latency_ms = (time.monotonic() - t0) * 1000

    _log.info(
        "self_critic: %d/%d claims grounded, completeness %.2f, ungrounded_by_agent=%s",
        total_claims - ungrounded_total, total_claims, completeness, ungrounded_by_agent,
    )

    # ── Routing decision ─────────────────────────────────────────────────────
    needs_retry = completeness < _COVERAGE_THRESHOLD or bool(ungrounded_by_agent)

    if not needs_retry:
        _append_trace(agent_trace, None, "success", latency_ms)
        return {
            "retry_agent": None,
            "agent_trace": agent_trace,
            "errors":      errors,
        }

    # Determine which node to retry
    if ungrounded_by_agent:
        # Re-run whichever upstream agent produced the most ungrounded claims.
        # max() preserves insertion order for ties (Python 3.7+ dict guarantee).
        weakest_agent: str = max(
            ungrounded_by_agent, key=lambda a: ungrounded_by_agent[a]
        )
        retry_agent: str = weakest_agent
    else:
        # All claims grounded but memo structure is incomplete — re-run memo_writer
        retry_agent = "memo_writer"

    new_count = state.get("retry_count", 0) + 1
    errors.append(
        f"self_critic: retry={retry_agent} #{new_count} "
        f"(completeness={completeness:.2f}, ungrounded={ungrounded_total}/{total_claims}, "
        f"by_agent={ungrounded_by_agent})"
    )
    _append_trace(agent_trace, None, f"retry-{retry_agent}", latency_ms)

    return {
        "retry_agent": retry_agent,
        "retry_count": new_count,
        "agent_trace": agent_trace,
        "errors":      errors,
    }
