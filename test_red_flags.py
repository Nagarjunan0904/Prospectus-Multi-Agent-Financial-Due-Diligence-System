"""
Verification tests for detect_debt_spike sign-crossing fix.

Core case: Boeing (BA) live data — four annual 10-K periods of D/E:
  FY2021: -3.5758
  FY2022: -3.0207   ← same-sign improvement; was already handled correctly
  FY2023: -13.7219  ← same-sign deepening (more negative); already correct
  FY2024:  9.8731   ← SIGN CROSSING: equity turned positive

Pre-fix: detect_debt_spike reported "D/E +172% year-over-year, severity: high"
         for the FY2023→FY2024 comparison (-13.72 → +9.87), and a similar
         false positive for the multi-year comparison.
Post-fix: both sign-crossing comparisons must emit equity_sign_change (medium)
          instead, and no debt_spike_* flag at all for BA.
"""
import pytest
from mcp_servers.risk_agent.tools.red_flags import detect_debt_spike


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(de: float, year: str) -> dict:
    return {"debt_to_equity": de, "period_end": year}


BA_HISTORY = [
    _p(-3.5758,  "2021-12-31"),
    _p(-3.0207,  "2022-12-31"),
    _p(-13.7219, "2023-12-31"),
    _p( 9.8731,  "2024-12-31"),
]


# ---------------------------------------------------------------------------
# BA sign-crossing tests (the bug cases)
# ---------------------------------------------------------------------------

def test_ba_no_false_debt_spike_recent():
    """FY2023→FY2024 (-13.72 → +9.87) must NOT produce debt_spike_recent."""
    names = [f["flag"] for f in detect_debt_spike(BA_HISTORY)]
    assert "debt_spike_recent" not in names


def test_ba_no_false_debt_spike_multi_year():
    """FY2021→FY2024 (-3.58 → +9.87) must NOT produce debt_spike_multi_year."""
    names = [f["flag"] for f in detect_debt_spike(BA_HISTORY)]
    assert "debt_spike_multi_year" not in names


def test_ba_emits_equity_sign_change_for_each_crossing():
    """Each sign-crossing comparison emits exactly one equity_sign_change.
    BA has two: recent (FY2023→FY2024) and multi-year (FY2021→FY2024)."""
    flags = detect_debt_spike(BA_HISTORY)
    sign_flags = [f for f in flags if f["flag"] == "equity_sign_change"]
    assert len(sign_flags) == 2, f"expected 2 equity_sign_change flags, got {sign_flags}"


def test_ba_equity_sign_change_severity_is_medium():
    flags = detect_debt_spike(BA_HISTORY)
    for f in flags:
        if f["flag"] == "equity_sign_change":
            assert f["severity"] == "medium", f"expected medium, got {f['severity']}"


# ---------------------------------------------------------------------------
# Regression: same-sign cases that were already correct must stay correct
# ---------------------------------------------------------------------------

def test_same_sign_negative_small_improvement_no_flag():
    """FY2021→FY2022: -3.5758 → -3.0207, pct = +15.5% < 25% threshold → []."""
    history = [_p(-3.5758, "2021-12-31"), _p(-3.0207, "2022-12-31")]
    assert detect_debt_spike(history) == []


def test_same_sign_negative_deepens_no_flag():
    """-3.0207 → -13.7219: pct is negative (leverage worsened) → []."""
    history = [_p(-3.0207, "2022-12-31"), _p(-13.7219, "2023-12-31")]
    assert detect_debt_spike(history) == []


# ---------------------------------------------------------------------------
# Sanity: normal positive-equity spikes must still be detected
# ---------------------------------------------------------------------------

def test_positive_spike_above_50pct_is_high():
    """0.50 → 0.80 = +60% > 50% threshold → debt_spike_recent severity=high."""
    flags = detect_debt_spike([_p(0.50, "2022-12-31"), _p(0.80, "2023-12-31")])
    assert len(flags) == 1
    assert flags[0]["flag"] == "debt_spike_recent"
    assert flags[0]["severity"] == "high"


def test_positive_spike_25_to_50pct_is_medium():
    """0.40 → 0.52 = +30% → debt_spike_recent severity=medium."""
    flags = detect_debt_spike([_p(0.40, "2022-12-31"), _p(0.52, "2023-12-31")])
    assert len(flags) == 1
    assert flags[0]["flag"] == "debt_spike_recent"
    assert flags[0]["severity"] == "medium"


def test_positive_spike_below_threshold_no_flag():
    """0.40 → 0.49 = +22.5% < 25% → []."""
    assert detect_debt_spike([_p(0.40, "2022-12-31"), _p(0.49, "2023-12-31")]) == []


def test_insufficient_data_returns_empty():
    assert detect_debt_spike([]) == []
    assert detect_debt_spike([_p(1.0, "2024-01-01")]) == []
