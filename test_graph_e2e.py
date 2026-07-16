"""
Checkpoint-resumability E2E test for the due-diligence graph.

Prerequisites
-------------
All four MCP servers must be running:
  python -m mcp_servers.data_agent.server      # :9001
  python -m mcp_servers.quant_agent.server     # :9002
  python -m mcp_servers.sentiment_agent.server # :9003
  python -m mcp_servers.risk_agent.server      # :9004

A .env file must contain DATABASE_URL, MCP_*_TOKEN, ALPHAVANTAGE_API_KEY.

Run
---
  python test_graph_e2e.py

What it tests
-------------
This is NOT a "run it twice" test.  It tests genuine LangGraph interrupt /
resume behaviour:

  Run 1  — compiled with interrupt_after=["data_agent"].
            ainvoke({"ticker": "NVDA"}, config) pauses after data_agent
            completes and checkpoints the partial state.  quant_agent,
            sentiment_agent, and risk_agent have NOT run yet.

  Run 2  — compiled WITHOUT interrupt_after.
            ainvoke(None, config) with the SAME thread_id resumes from the
            checkpoint.  LangGraph replays from the last saved node
            (data_agent) rather than from START, so supervisor and
            data_agent do NOT re-execute.

Assertions
----------
  • After run 1: cik, company_facts, filing_sections are present;
                 ratios, ratio_history, sentiment, risk_flags are absent.
  • After run 2: all fields populated; supervisor and data_agent trace
                 entry counts unchanged from run 1 (they didn't re-run);
                 quant_agent, sentiment_agent, risk_agent have new entries.

DELETE THIS FILE once checkpoint resumability is confirmed.
"""
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

from backend._platform import apply_windows_event_loop_fix
apply_windows_event_loop_fix()

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_count(trace: list[dict], node: str) -> int:
    return sum(1 for e in trace if e.get("node") == node)


def _print_trace(trace: list[dict]) -> None:
    for e in trace:
        ms = f"{e.get('latency_ms', 0):.0f} ms" if "latency_ms" in e else "—"
        print(f"  {e.get('node', '?'):20} {str(e.get('tool') or '—'):30} "
              f"{e.get('status', '?'):8} {ms}")


def _pp(label: str, value: object) -> None:
    print(f"\n{label}:")
    print(json.dumps(value, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    from backend.graph import make_graph

    # Use a unique thread_id so the test is repeatable without leftover state.
    # Change this string if you need a clean run after a previous failed attempt.
    thread_cfg = {"configurable": {"thread_id": "e2e-resumability-nvda-001"}}

    # ── Run 1: interrupt after data_agent ────────────────────────────────────
    print("\n" + "=" * 65)
    print("RUN 1 — interrupt_after=['data_agent']  (partial run)")
    print("=" * 65)

    async with make_graph(interrupt_after=["data_agent"]) as graph:
        state1 = await graph.ainvoke({"ticker": "NVDA"}, config=thread_cfg)

    trace1: list[dict] = state1.get("agent_trace") or []
    print(f"\nagent_trace ({len(trace1)} entries):")
    _print_trace(trace1)

    print(f"\ncik                 : {state1.get('cik')}")
    print(f"company_facts set   : {bool(state1.get('company_facts'))}")
    print(f"filing_sections keys: {sorted((state1.get('filing_sections') or {}).keys())}")
    print(f"ratios set          : {bool(state1.get('ratios'))}  ← should be False")
    print(f"ratio_history set   : {bool(state1.get('ratio_history'))}  ← should be False")
    print(f"sentiment set       : {bool(state1.get('sentiment'))}  ← should be False")
    print(f"risk_flags set      : {state1.get('risk_flags') is not None}  ← should be False")
    if state1.get("errors"):
        print(f"errors: {state1['errors']}")

    # Hard assertions for run 1
    assert state1.get("cik"), "FAIL: cik not set — supervisor failed"
    assert state1.get("company_facts"), "FAIL: company_facts empty — data_agent failed"
    assert not state1.get("ratios"), \
        "FAIL: ratios already set — interrupt_after did not pause before quant_agent"
    assert state1.get("risk_flags") is None, \
        "FAIL: risk_flags already set — interrupt_after did not pause before risk_agent"

    # Baseline trace counts after run 1 (only supervisor + data_agent have run)
    sup_before   = _node_count(trace1, "supervisor")
    data_before  = _node_count(trace1, "data_agent")
    quant_before = _node_count(trace1, "quant_agent")
    sent_before  = _node_count(trace1, "sentiment_agent")
    risk_before  = _node_count(trace1, "risk_agent")

    assert sup_before  > 0, "FAIL: supervisor left no trace entries"
    assert data_before > 0, "FAIL: data_agent left no trace entries"
    assert quant_before == 0, "FAIL: quant_agent ran before resume"
    assert sent_before  == 0, "FAIL: sentiment_agent ran before resume"
    assert risk_before  == 0, "FAIL: risk_agent ran before resume"

    print("\n✓ Run 1 assertions passed — partial state checkpointed correctly")

    # ── Run 2: resume from checkpoint ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RUN 2 — ainvoke(None, same thread_id)  (resume from checkpoint)")
    print("=" * 65)

    async with make_graph() as graph:          # no interrupt_after
        state2 = await graph.ainvoke(None, config=thread_cfg)

    trace2: list[dict] = state2.get("agent_trace") or []
    print(f"\nagent_trace ({len(trace2)} entries):")
    _print_trace(trace2)

    sup_after   = _node_count(trace2, "supervisor")
    data_after  = _node_count(trace2, "data_agent")
    quant_after = _node_count(trace2, "quant_agent")
    sent_after  = _node_count(trace2, "sentiment_agent")
    risk_after  = _node_count(trace2, "risk_agent")

    print(f"\n{'Node':22} {'before':>8} {'after':>7}  note")
    print("-" * 55)
    print(f"{'supervisor':22} {sup_before:>8} {sup_after:>7}  should be unchanged")
    print(f"{'data_agent':22} {data_before:>8} {data_after:>7}  should be unchanged")
    print(f"{'quant_agent':22} {quant_before:>8} {quant_after:>7}  should be > 0")
    print(f"{'sentiment_agent':22} {sent_before:>8} {sent_after:>7}  should be > 0")
    print(f"{'risk_agent':22} {risk_before:>8} {risk_after:>7}  should be > 0")

    _pp("ratios",      state2.get("ratios") or {})
    print(f"\nratio_history ({len(state2.get('ratio_history') or [])} periods):")
    for row in (state2.get("ratio_history") or []):
        print(f"  {row}")
    _pp("sentiment",   state2.get("sentiment") or {})
    flags = state2.get("risk_flags") or []
    print(f"\nrisk_flags ({len(flags)}):")
    for f in flags:
        print(f"  [{f.get('severity','?').upper():6}] {f.get('flag')}: "
              f"{str(f.get('evidence',''))[:80]}")
    if state2.get("errors"):
        print(f"\nerrors: {state2['errors']}")

    # Hard assertions for run 2
    assert sup_after == sup_before, \
        f"FAIL: supervisor re-executed in run 2 ({sup_before} → {sup_after} entries)"
    assert data_after == data_before, \
        f"FAIL: data_agent re-executed in run 2 ({data_before} → {data_after} entries)"
    assert quant_after > 0, \
        "FAIL: quant_agent has no trace entries — did not run during resume"
    assert sent_after > 0, \
        "FAIL: sentiment_agent has no trace entries — did not run during resume"
    assert risk_after > 0, \
        "FAIL: risk_agent has no trace entries — did not run during resume"
    assert state2.get("ratios") is not None, \
        "FAIL: ratios not populated after full run"
    assert state2.get("risk_flags") is not None, \
        "FAIL: risk_flags not populated after full run"

    print("\n" + "=" * 65)
    print("✓ ALL checkpoint-resumability assertions passed")
    print("  supervisor and data_agent did NOT re-execute in run 2")
    print("  quant_agent, sentiment_agent, risk_agent ran for the first time")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
