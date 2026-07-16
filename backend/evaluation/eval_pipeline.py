#!/usr/bin/env python3
"""
Offline evaluation pipeline for the due-diligence graph.

Usage
-----
    # Full golden set (~24 Alpha Vantage calls):
    python backend/evaluation/eval_pipeline.py

    # Subset — burn only 1-2 AV calls while debugging the eval itself:
    python backend/evaluation/eval_pipeline.py --tickers NVDA MSFT

Writes results to data/eval_results.json, which GET /eval reads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the project root (python backend/evaluation/eval_pipeline.py).
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv                                   # noqa: E402
load_dotenv(ROOT / ".env")                                       # must precede backend imports

from backend._platform import apply_windows_event_loop_fix       # noqa: E402
apply_windows_event_loop_fix()

from backend.graph import make_graph                             # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR      = ROOT / "data"
TICKERS_PATH  = DATA_DIR / "eval_tickers.json"
RESULTS_PATH  = DATA_DIR / "eval_results.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOLERANCE = 0.02  # 2% relative tolerance for ratio field comparisons

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(numerator: float, denominator: float) -> float | None:
    """Safe fraction, rounded to 4 dp. Returns None when denominator is 0."""
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _within_tolerance(computed: float | None, expected: float) -> bool:
    """True iff computed is within TOLERANCE relative to expected."""
    if computed is None:
        return False
    if expected == 0:
        return abs(computed) <= TOLERANCE
    return abs(computed - expected) / abs(expected) <= TOLERANCE


def _flag_matches(raised: str, expected_token: str) -> bool:
    """Case-insensitive substring: expected_token appears somewhere in the raised flag name."""
    return expected_token.lower() in raised.lower()


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]. Returns 0.0 for empty list."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------

async def run_eval(tickers_subset: list[str] | None = None) -> None:
    # Load golden set
    raw = json.loads(TICKERS_PATH.read_text(encoding="utf-8"))
    golden: list[dict] = raw if isinstance(raw, list) else raw["eval_set"]

    if tickers_subset:
        golden = [e for e in golden if e["ticker"] in tickers_subset]
        if not golden:
            sys.exit(
                f"None of {tickers_subset} found in {TICKERS_PATH}. "
                f"Available: {[e['ticker'] for e in (raw if isinstance(raw, list) else raw['eval_set'])]}"
            )

    run_ts = datetime.now(timezone.utc).isoformat()
    # Use a compact timestamp in thread IDs (colons are fine for LangGraph; this form is readable)
    run_ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    per_ticker: dict[str, dict] = {}
    latencies:  list[float]     = []

    # Running totals for aggregate metrics
    ratio_pass  = ratio_total  = 0
    total_raised = matching_raised = 0
    total_expected = found_expected = 0
    coverage_sum = 0.0
    retry_count_positive = 0

    print(f"\nEval run {run_ts_tag}  ({len(golden)} tickers)\n{'─'*50}")

    async with make_graph() as graph:
        for entry in golden:
            ticker          = entry["ticker"]
            expected_ratios = entry.get("expected_ratios") or {}
            expected_flags  = entry.get("expected_flags")  or []

            # Fresh thread_id each run — never resumes a stale checkpoint.
            thread_id = f"eval-{ticker}-{run_ts_tag}"
            state_in  = {
                "ticker":      ticker,
                "errors":      [],
                "agent_trace": [],
                "retry_count": 0,
            }
            config = {"configurable": {"thread_id": thread_id}}

            print(f"  {ticker:<6}", end=" ", flush=True)
            t0 = time.perf_counter()
            error_msg: str | None = None
            result: dict = {}

            try:
                result = await graph.ainvoke(state_in, config=config)
            except Exception as exc:
                error_msg = str(exc)

            latency = time.perf_counter() - t0
            latencies.append(latency)

            cov          = result.get("citation_coverage") or 0.0
            retry_count  = result.get("retry_count")       or 0
            coverage_sum += cov

            if retry_count > 0:
                retry_count_positive += 1

            print(
                f"{latency:5.1f}s  "
                f"cc={cov:.2f}  "
                f"retries={retry_count}"
                + (f"  ERROR: {error_msg}" if error_msg else "")
            )

            # ── Ratio comparison ────────────────────────────────────────
            computed_ratios = result.get("ratios") or {}
            field_results: dict[str, dict] = {}

            for field, exp_val in expected_ratios.items():
                comp_val = computed_ratios.get(field)
                passed   = _within_tolerance(comp_val, exp_val)
                field_results[field] = {
                    "expected": exp_val,
                    "computed": comp_val,
                    "pass":     passed,
                }
                ratio_total += 1
                if passed:
                    ratio_pass += 1

            # ── Flag precision / recall ─────────────────────────────────
            raised_flags = [f.get("flag", "") for f in (result.get("risk_flags") or [])]

            # precision denominator: every flag raised, across all tickers
            for rf in raised_flags:
                total_raised += 1
                if any(_flag_matches(rf, ef) for ef in expected_flags):
                    matching_raised += 1

            # recall denominator: every expected token, across all tickers
            for ef in expected_flags:
                total_expected += 1
                if any(_flag_matches(rf, ef) for rf in raised_flags):
                    found_expected += 1

            per_ticker[ticker] = {
                "ratio_fields":      field_results,
                "flags_raised":      raised_flags,
                "flags_expected":    expected_flags,
                "citation_coverage": cov,
                "retry_count":       retry_count,
                "latency_s":         round(latency, 2),
                "error":             error_msg,
            }

    # ── Aggregates ──────────────────────────────────────────────────────────
    n = len(golden)
    output = {
        "run_timestamp":         run_ts,
        "tickers_evaluated":     n,
        "ratio_accuracy":        _pct(ratio_pass, ratio_total),
        "avg_citation_coverage": round(coverage_sum / n, 4) if n else None,
        "retry_rate":            round(retry_count_positive / n, 4) if n else None,
        "red_flag_precision":    _pct(matching_raised, total_raised),
        "red_flag_recall":       _pct(found_expected,  total_expected),
        "latency_p50":           round(_percentile(latencies, 50), 2),
        "latency_p95":           round(_percentile(latencies, 95), 2),
        "per_ticker":            per_ticker,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'─'*50}")
    print(f"Results → {RESULTS_PATH}")
    print(f"  ratio_accuracy        : {output['ratio_accuracy']}")
    print(f"  avg_citation_coverage : {output['avg_citation_coverage']}")
    print(f"  retry_rate            : {output['retry_rate']}")
    print(f"  red_flag_precision    : {output['red_flag_precision']}")
    print(f"  red_flag_recall       : {output['red_flag_recall']}")
    print(f"  latency p50 / p95     : {output['latency_p50']}s / {output['latency_p95']}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Due-diligence evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python backend/evaluation/eval_pipeline.py\n"
            "  python backend/evaluation/eval_pipeline.py --tickers NVDA MSFT\n"
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Run only these tickers (default: full 8-ticker golden set)",
    )
    args = parser.parse_args()
    asyncio.run(run_eval(args.tickers))


if __name__ == "__main__":
    main()
