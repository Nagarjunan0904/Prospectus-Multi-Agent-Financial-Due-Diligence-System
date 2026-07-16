"""
Deterministic unit tests for the self_critic retry cap.

These tests do NOT require a running Postgres instance, LLM API keys, or
MCP servers — they exercise _route_self_critic and self_critic.run()
in isolation using synthetic state dicts.

Run
---
    python test_retry_cap.py

What it tests
-------------
1. Cap enforced: _route_self_critic returns END when retry_count >= MAX_RETRIES,
   even when retry_agent names a valid retryable node.

2. Cap not yet reached: _route_self_critic returns the retry node name when
   retry_count < MAX_RETRIES.

3. No retry needed: _route_self_critic returns END when retry_agent is None
   regardless of retry_count.

4. self_critic.run() increments retry_count correctly on a memo with an
   ungrounded claim (source_field resolves to nothing in state).

5. self_critic.run() does NOT set retry_agent when the memo is fully
   grounded and complete.
"""
from __future__ import annotations

import asyncio
import sys

from backend._platform import apply_windows_event_loop_fix
apply_windows_event_loop_fix()

# ---------------------------------------------------------------------------
# Test 1-3: _route_self_critic cap enforcement (pure function, no I/O)
# ---------------------------------------------------------------------------

from langgraph.graph import END
from backend.graph import MAX_RETRIES, _route_self_critic


def test_cap_enforced() -> None:
    """At retry_count == MAX_RETRIES, must return END even with a valid retry node."""
    state = {
        "retry_agent": "quant_agent",
        "retry_count": MAX_RETRIES,   # already at limit
        "memo": {
            "ticker": "NVDA",
            "sections": [
                {"heading": "Financial Snapshot", "claims": [
                    {"text": "P/E is high.", "source_field": "ratios.pe_ratio"},
                ]},
            ],
        },
    }
    result = _route_self_critic(state)  # type: ignore[arg-type]
    assert result == END, (
        f"FAIL: expected END when retry_count={MAX_RETRIES} "
        f"(cap={MAX_RETRIES}), got {result!r}"
    )
    print(f"  [PASS] cap enforced: _route_self_critic returned END "
          f"(retry_count={MAX_RETRIES}, retry_agent='quant_agent')")


def test_cap_not_yet_reached() -> None:
    """Below the cap, must return the retry node name."""
    state = {
        "retry_agent": "quant_agent",
        "retry_count": MAX_RETRIES - 1,
    }
    result = _route_self_critic(state)  # type: ignore[arg-type]
    assert result == "quant_agent", (
        f"FAIL: expected 'quant_agent' when retry_count={MAX_RETRIES - 1}, got {result!r}"
    )
    print(f"  [PASS] below cap: _route_self_critic returned 'quant_agent' "
          f"(retry_count={MAX_RETRIES - 1})")


def test_no_retry_needed() -> None:
    """None retry_agent always returns END, regardless of retry_count."""
    for count in (0, 1, MAX_RETRIES, MAX_RETRIES + 5):
        state = {"retry_agent": None, "retry_count": count}
        result = _route_self_critic(state)  # type: ignore[arg-type]
        assert result == END, (
            f"FAIL: expected END for retry_agent=None, retry_count={count}, got {result!r}"
        )
    print(f"  [PASS] retry_agent=None always returns END")


def test_unknown_node_name() -> None:
    """An unknown retry_agent that isn't in _RETRYABLE_NODES falls back to END."""
    state = {"retry_agent": "nonexistent_node", "retry_count": 0}
    result = _route_self_critic(state)  # type: ignore[arg-type]
    assert result == END, (
        f"FAIL: expected END for unknown node, got {result!r}"
    )
    print(f"  [PASS] unknown node name returns END (not in _RETRYABLE_NODES)")


# ---------------------------------------------------------------------------
# Tests 4-5: self_critic.run() — async, but still no I/O
# ---------------------------------------------------------------------------

from backend.agents.self_critic import run as self_critic_run


_FOUR_HEADING_MEMO = {
    "ticker": "NVDA",
    "sections": [
        {
            "heading": "Financial Snapshot",
            "claims": [{"text": "P/E is 82.", "source_field": "ratios.pe_ratio"}],
        },
        {
            "heading": "Sentiment",
            "claims": [{"text": "Positive sentiment.", "source_field": "sentiment.positive_pct"}],
        },
        {
            "heading": "Risk Factors",
            "claims": [{"text": "No flags found.", "source_field": "risk_flags"}],
        },
        {
            "heading": "Recommendation",
            "claims": [{"text": "HOLD — mixed signals.", "source_field": "ratios.pe_ratio"}],
        },
    ],
}


async def test_empty_risk_flags_grounded_routes_to_memo_writer() -> None:
    """
    Core regression test for the empty-container grounding fix.

    Scenario (matches what NVDA legitimately produces):
      - risk_flags = [] is present in state (risk_agent ran, found nothing)
      - Sentiment section in the memo is empty (LLM left it blank)
      - Risk Factors section correctly cites risk_flags=[] as source_field

    Before fix: risk_flags=[] was treated as ungrounded → retry_agent='risk_agent'
                (wrong: re-running risk_agent won't fill the Sentiment section)
    After fix:  risk_flags=[] is grounded → ungrounded_by_agent is empty →
                completeness=0.75 triggers retry_agent='memo_writer'
                (correct: the gap is in the memo structure, not upstream data)
    """
    state = {
        "ticker": "NVDA",
        "company_facts": {"foo": "bar"},
        "ratios": {"pe_ratio": 82.0},
        "risk_flags": [],       # present but empty — legitimately "no flags found"
        # "sentiment" deliberately absent from state
        "memo": {
            "ticker": "NVDA",
            "sections": [
                {
                    "heading": "Financial Snapshot",
                    "claims": [{"text": "P/E 82.", "source_field": "ratios.pe_ratio"}],
                },
                {
                    "heading": "Sentiment",
                    "claims": [],           # LLM left this empty — completeness gap
                },
                {
                    "heading": "Risk Factors",
                    "claims": [
                        # Accurately cites risk_flags=[] — should be GROUNDED after fix
                        {"text": "No risk flags found.", "source_field": "risk_flags"}
                    ],
                },
                {
                    "heading": "Recommendation",
                    "claims": [{"text": "HOLD.", "source_field": "ratios.pe_ratio"}],
                },
            ],
        },
        "retry_count": 0,
    }
    result = await self_critic_run(state)  # type: ignore[arg-type]

    # risk_flags=[] is NOW grounded → no ungrounded claims → completeness is the
    # only issue → retry_agent must be "memo_writer", not "risk_agent"
    assert result.get("retry_agent") == "memo_writer", (
        f"FAIL: expected retry_agent='memo_writer' (empty Sentiment section = "
        f"completeness gap, not a grounding issue), got {result.get('retry_agent')!r}\n"
        f"  Hint: if this returns 'risk_agent', the empty-container fix didn't take."
    )
    assert result.get("retry_count") == 1, (
        f"FAIL: expected retry_count=1, got {result.get('retry_count')}"
    )
    print(
        f"  [PASS] empty risk_flags=[] is grounded → retry_agent='memo_writer' "
        f"(completeness 0.75, ungrounded_by_agent empty), retry_count=1"
    )


async def test_missing_field_still_ungrounded() -> None:
    """A claim citing a key that doesn't exist at all must still be ungrounded."""
    state = {
        "ticker": "NVDA",
        "company_facts": {"foo": "bar"},
        "ratios": {"pe_ratio": 82.0},
        # "sentiment" not in state at all
        "risk_flags": [],
        "memo": {
            "ticker": "NVDA",
            "sections": [
                {
                    "heading": "Financial Snapshot",
                    "claims": [{"text": "P/E 82.", "source_field": "ratios.pe_ratio"}],
                },
                {
                    "heading": "Sentiment",
                    "claims": [
                        # "sentiment" key absent from state → KeyError → ungrounded
                        {"text": "Positive.", "source_field": "sentiment.positive_pct"}
                    ],
                },
                {
                    "heading": "Risk Factors",
                    "claims": [{"text": "No flags.", "source_field": "risk_flags"}],
                },
                {
                    "heading": "Recommendation",
                    "claims": [{"text": "HOLD.", "source_field": "ratios.pe_ratio"}],
                },
            ],
        },
        "retry_count": 0,
    }
    result = await self_critic_run(state)  # type: ignore[arg-type]

    # sentiment.positive_pct is absent → ungrounded → retry sentiment_agent
    assert result.get("retry_agent") == "sentiment_agent", (
        f"FAIL: expected retry_agent='sentiment_agent' (missing key), "
        f"got {result.get('retry_agent')!r}"
    )
    assert result.get("retry_count") == 1, (
        f"FAIL: expected retry_count=1, got {result.get('retry_count')}"
    )
    print(
        f"  [PASS] missing key still ungrounded → retry_agent='sentiment_agent', "
        f"retry_count=1"
    )


async def test_self_critic_no_retry_when_grounded_and_complete() -> None:
    """self_critic must NOT set retry_agent when memo is grounded and complete.

    After the empty-container fix, risk_flags=[] is grounded, so we can use it
    directly in state instead of the workaround of a non-empty list.
    """
    state = {
        "ticker": "NVDA",
        "company_facts": {"foo": "bar"},
        "ratios": {"pe_ratio": 82.0},
        "sentiment": {"positive_pct": 0.6},
        "risk_flags": [],   # empty list is now grounded — no workaround needed
        "memo": _FOUR_HEADING_MEMO,
        "retry_count": 0,
    }
    result = await self_critic_run(state)  # type: ignore[arg-type]

    assert result.get("retry_agent") is None, (
        f"FAIL: expected retry_agent=None for fully grounded memo, "
        f"got {result.get('retry_agent')!r}"
    )
    assert "retry_count" not in result, (
        f"FAIL: retry_count should not be present when no retry is needed, "
        f"got {result.get('retry_count')}"
    )
    print(
        f"  [PASS] fully grounded memo (risk_flags=[]) → retry_agent=None, "
        f"no retry_count in result"
    )


async def test_cap_end_to_end() -> None:
    """Full sequence: two retries exhaust the cap and _route_self_critic returns END."""
    # Simulate state after self_critic has already issued MAX_RETRIES retries
    # (self_critic.run() would have set retry_count = MAX_RETRIES).
    # _route_self_critic must return END even though retry_agent is set.
    state_after_second_retry = {
        "retry_agent": "sentiment_agent",
        "retry_count": MAX_RETRIES,
    }
    result = _route_self_critic(state_after_second_retry)  # type: ignore[arg-type]
    assert result == END, (
        f"FAIL: after {MAX_RETRIES} retries _route_self_critic should return END, "
        f"got {result!r}"
    )

    # And at one retry remaining (retry_count = MAX_RETRIES - 1), it should route:
    state_after_first_retry = {
        "retry_agent": "sentiment_agent",
        "retry_count": MAX_RETRIES - 1,
    }
    result2 = _route_self_critic(state_after_first_retry)  # type: ignore[arg-type]
    assert result2 == "sentiment_agent", (
        f"FAIL: with one retry remaining should return 'sentiment_agent', got {result2!r}"
    )
    print(
        f"  [PASS] end-to-end cap: retry #{MAX_RETRIES - 1} → 'sentiment_agent', "
        f"retry #{MAX_RETRIES} → END"
    )


# ---------------------------------------------------------------------------
# Graph-structure tests: defer=True fan-in fix (no DB or MCP needed)
# ---------------------------------------------------------------------------

def test_memo_writer_deferred() -> None:
    """
    memo_writer must carry defer=True because it has two predecessors that arrive
    in different supersteps (sentiment_agent at depth N+2, risk_agent at depth N+3).
    Without defer=True, LangGraph fires memo_writer once per superstep a predecessor
    settles in, causing two LLM calls per run.
    """
    from langgraph.graph._node import StateNodeSpec
    from backend.graph import _builder

    node_spec = _builder.nodes["memo_writer"]
    assert isinstance(node_spec, StateNodeSpec), (
        f"FAIL: expected StateNodeSpec, got {type(node_spec)}"
    )
    assert node_spec.defer is True, (
        f"FAIL: memo_writer.defer is {node_spec.defer!r}, expected True\n"
        f"  Without defer=True memo_writer fires twice per run (once when "
        f"sentiment_agent finishes, once when risk_agent finishes)."
    )
    print("  [PASS] memo_writer has defer=True — will wait for all predecessors")


def test_self_critic_not_deferred() -> None:
    """self_critic has exactly one predecessor (memo_writer); defer=True not needed."""
    from langgraph.graph._node import StateNodeSpec
    from backend.graph import _builder

    node_spec = _builder.nodes["self_critic"]
    assert isinstance(node_spec, StateNodeSpec)
    # defer is False by default and should remain so — self_critic is not a fan-in node
    assert node_spec.defer is False, (
        f"FAIL: self_critic.defer is {node_spec.defer!r}, expected False"
    )
    print("  [PASS] self_critic has defer=False — single predecessor, no fan-in issue")


def test_only_memo_writer_is_fan_in() -> None:
    """Structural check: memo_writer is the ONLY node with multiple incoming edges."""
    from collections import defaultdict
    from backend.graph import _builder

    in_edges: dict[str, list[str]] = defaultdict(list)
    for src, dst in _builder.edges:
        if src != "__start__":
            in_edges[dst].append(src)

    fan_in_nodes = {node: preds for node, preds in in_edges.items() if len(preds) > 1}
    assert set(fan_in_nodes.keys()) == {"memo_writer"}, (
        f"FAIL: expected only memo_writer to be a fan-in node, "
        f"got {set(fan_in_nodes.keys())}\n"
        f"  Any new fan-in node with asymmetric predecessors may also need defer=True."
    )
    print(
        f"  [PASS] memo_writer is the only fan-in node "
        f"(predecessors: {fan_in_nodes['memo_writer']})"
    )


def test_graph_still_compiles() -> None:
    """Compile the graph with an in-memory checkpointer to catch edge/node errors."""
    from langgraph.checkpoint.memory import MemorySaver
    from backend.graph import _builder

    g = _builder.compile(checkpointer=MemorySaver())
    node_names = set(g.nodes.keys()) - {"__start__"}
    expected = {"supervisor", "data_agent", "quant_agent", "sentiment_agent",
                "risk_agent", "memo_writer", "self_critic"}
    assert node_names == expected, (
        f"FAIL: compiled graph nodes {node_names} != expected {expected}"
    )
    print(f"  [PASS] graph compiles OK with MemorySaver ({len(expected)} nodes)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_async_tests() -> None:
    await test_empty_risk_flags_grounded_routes_to_memo_writer()
    await test_missing_field_still_ungrounded()
    await test_self_critic_no_retry_when_grounded_and_complete()
    await test_cap_end_to_end()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Retry-cap unit tests")
    print("=" * 60)

    print("\n-- _route_self_critic (synchronous) --")
    test_cap_enforced()
    test_cap_not_yet_reached()
    test_no_retry_needed()
    test_unknown_node_name()

    print("\n-- self_critic.run() (async) --")
    asyncio.run(_run_async_tests())

    print("\n-- Graph structure: defer=True fan-in fix --")
    test_memo_writer_deferred()
    test_self_critic_not_deferred()
    test_only_memo_writer_is_fan_in()
    test_graph_still_compiles()

    print("\n" + "=" * 60)
    print("ALL tests passed")
    print("=" * 60)
