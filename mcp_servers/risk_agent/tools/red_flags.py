"""
Risk-flag detectors for the due-diligence pipeline.

Public API (MCP-exposed via server.py)
---------------------------------------
detect_debt_spike(ratio_history)
    Compares the most recent annual D/E against the prior period (QoQ)
    and the period ~4 years back (YoY).  Returns [] if fewer than 2
    usable (non-None) data points.

detect_insider_selling_cluster(cik, days=30)
    Fetches the full insider-transaction list from the Data Agent.
    Flags when 3+ distinct insiders sold with zero offsetting purchases
    in the same window.

flag_audit_language(mdna_text, risk_factors_text)
    Word-boundary regex scan for going-concern language, material
    weakness disclosures, and restatement language.  Returns [] if
    both input texts are empty.

run_all_checks(cik, ratio_history, mdna_text, risk_factors_text)
    Orchestrates all three detectors.  Each detector is wrapped in its
    own try/except so a single failure never silences the others.
    Output matches DueDiligenceState["risk_flags"] exactly.

Flag shape
----------
Every flag returned by any detector is:
  {
      "flag":        str,   # snake_case identifier
      "severity":    str,   # "high" | "medium"
      "evidence":    str,   # human-readable explanation / excerpt
      "source_tool": str,   # "agent.tool_name" that produced the data
  }
"""
from __future__ import annotations

import logging
import re
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# detect_debt_spike
# ---------------------------------------------------------------------------

def detect_debt_spike(ratio_history: list[dict]) -> list[dict]:
    """Detect anomalous D/E increases across annual 10-K periods.

    ``ratio_history`` contains only annual (10-K) data, so all comparisons
    are between fiscal year-ends — there is no quarterly granularity here.

    Thresholds (applied to percentage increase):
      > 25 %  →  severity='medium'
      > 50 %  →  severity='high'

    Two comparisons:
      debt_spike_recent     — most recent annual filing vs the immediately
                              preceding one (adjacent-year change).
      debt_spike_multi_year — most recent annual filing vs the one ~4 filings
                              back (or as far back as available).  Skipped when
                              it would duplicate the recent comparison (fewer
                              than 3 usable periods available).

    # Phase 3 follow-up: extend get_ratio_history to also return 10-Q periods
    # so detect_debt_spike can offer a genuine quarter-over-quarter check
    # alongside the existing annual comparisons.
    """
    usable = sorted(
        [h for h in ratio_history if h.get("debt_to_equity") is not None],
        key=lambda h: h["period_end"],
    )

    if len(usable) < 2:
        return []

    flags: list[dict] = []

    def _severity(pct: float) -> str | None:
        if pct > 0.50:
            return "high"
        if pct > 0.25:
            return "medium"
        return None

    def _flag(name: str, label: str, prev: dict, curr: dict, pct: float) -> dict:
        return {
            "flag": name,
            "severity": _severity(pct),  # type: ignore[arg-type]
            "evidence": (
                f"D/E rose from {prev['debt_to_equity']:.4f} to "
                f"{curr['debt_to_equity']:.4f} "
                f"(+{pct:.0%}) {label} "
                f"({prev['period_end']} → {curr['period_end']})"
            ),
            "source_tool": "quant_agent.get_ratio_history",
        }

    def _signs_differ(a: float, b: float) -> bool:
        """True when a and b have opposite signs.

        When D/E crosses zero between periods (positive equity → negative, or
        vice versa), the percentage-change formula produces a large,
        confidently-labelled but substantively meaningless number.  Skip the
        percentage calculation entirely for such pairs.
        """
        return (a < 0) != (b < 0)

    def _equity_sign_change_flag(label: str, prev: dict, curr: dict) -> dict:
        return {
            "flag": "equity_sign_change",
            "severity": "medium",
            "evidence": (
                f"D/E crossed zero {label} "
                f"({prev['period_end']}: {prev['debt_to_equity']:.4f} → "
                f"{curr['period_end']}: {curr['debt_to_equity']:.4f}); "
                "percentage-change comparison suppressed — D/E trend is "
                "mathematically meaningless across an equity sign boundary."
            ),
            "source_tool": "quant_agent.get_ratio_history",
        }

    # Recent: last two available annual filings
    recent_prev, recent_curr = usable[-2], usable[-1]
    if recent_prev["debt_to_equity"] != 0:
        if _signs_differ(recent_prev["debt_to_equity"], recent_curr["debt_to_equity"]):
            flags.append(_equity_sign_change_flag("year-over-year", recent_prev, recent_curr))
        else:
            pct = (
                (recent_curr["debt_to_equity"] - recent_prev["debt_to_equity"])
                / abs(recent_prev["debt_to_equity"])
            )
            if _severity(pct):
                flags.append(
                    _flag("debt_spike_recent", "year-over-year", recent_prev, recent_curr, pct)
                )

    # Multi-year: most recent vs ~4 annual filings back; skip when it duplicates recent
    multi_prev_idx = max(0, len(usable) - 5)
    multi_prev = usable[multi_prev_idx]
    if multi_prev["period_end"] != recent_prev["period_end"]:
        multi_curr = usable[-1]
        if multi_prev["debt_to_equity"] != 0:
            if _signs_differ(multi_prev["debt_to_equity"], multi_curr["debt_to_equity"]):
                flags.append(_equity_sign_change_flag("over multi-year period", multi_prev, multi_curr))
            else:
                pct = (
                    (multi_curr["debt_to_equity"] - multi_prev["debt_to_equity"])
                    / abs(multi_prev["debt_to_equity"])
                )
                if _severity(pct):
                    flags.append(
                        _flag("debt_spike_multi_year", "over multi-year period", multi_prev, multi_curr, pct)
                    )

    return flags


# ---------------------------------------------------------------------------
# detect_insider_selling_cluster
# ---------------------------------------------------------------------------

async def detect_insider_selling_cluster(
    cik: str,
    days: int = 30,
) -> list[dict]:
    """Flag coordinated insider selling with no offsetting purchases.

    Severity:
      3–4 distinct sellers → 'medium'
      5+  distinct sellers → 'high'

    Returns [] when:
      • fewer than 3 distinct sellers in the window, OR
      • any insider made a purchase ('P') in the same window —
        offsetting purchases cancel the signal.
    """
    from mcp_clients.data_client import get_insider_transactions

    result = await get_insider_transactions(cik, days)
    transactions: list[dict] = result.get("transactions", [])

    sellers: set[str] = set()
    buyers: set[str] = set()

    for txn in transactions:
        code = (txn.get("transaction_code") or "").strip()
        filer = (txn.get("filer_name") or "").strip()
        if not filer:
            continue
        if code == "S":
            sellers.add(filer)
        elif code == "P":
            buyers.add(filer)

    if buyers:
        return []
    if len(sellers) < 3:
        return []

    severity = "high" if len(sellers) >= 5 else "medium"
    seller_list = ", ".join(sorted(sellers))

    return [
        {
            "flag": "insider_selling_cluster",
            "severity": severity,
            "evidence": (
                f"{len(sellers)} distinct insiders sold within {days}-day window "
                f"(sellers: {seller_list}), zero offsetting purchases"
            ),
            "source_tool": "data_agent.get_insider_transactions",
        }
    ]


# ---------------------------------------------------------------------------
# flag_audit_language
# ---------------------------------------------------------------------------

# Each entry: (flag_name, severity, [compiled_pattern, ...])
# First matching pattern in the category wins; one flag emitted per category.
_AUDIT_CATEGORIES: list[tuple[str, str, list[re.Pattern[str]]]] = [
    (
        "going_concern",
        "high",
        [
            re.compile(r"\bgoing concern\b", re.IGNORECASE),
            re.compile(r"\bsubstantial doubt\b", re.IGNORECASE),
        ],
    ),
    (
        "material_weakness",
        "high",
        [
            re.compile(r"\bmaterial weakness\b", re.IGNORECASE),
        ],
    ),
    (
        "restatement",
        "medium",
        [
            re.compile(r"\brestate\b", re.IGNORECASE),
            re.compile(r"\brestatement of\b", re.IGNORECASE),
        ],
    ),
]

_EVIDENCE_RADIUS = 75  # chars on each side of the match → ~150-char window


def flag_audit_language(
    mdna_text: str,
    risk_factors_text: str,
) -> list[dict]:
    """Scan MD&A and Risk Factors text for high-risk disclosure language.

    Uses word-boundary regex (\\b…\\b) for each phrase so partial-word
    matches (e.g. "concerns" for "concern") are excluded.

    One flag per matched category regardless of how many times the phrase
    appears.  Evidence is the ~150-char sentence window around the first
    match across both texts.
    """
    if not mdna_text and not risk_factors_text:
        return []

    # Search each source separately for cleaner evidence extraction;
    # flag as soon as either source matches.
    sources = [
        ("MD&A",           mdna_text         or ""),
        ("Risk Factors",   risk_factors_text or ""),
    ]

    flags: list[dict] = []
    for flag_name, severity, patterns in _AUDIT_CATEGORIES:
        matched = False
        for _src_label, text in sources:
            if matched:
                break
            for pat in patterns:
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - _EVIDENCE_RADIUS)
                    end   = min(len(text), m.end() + _EVIDENCE_RADIUS)
                    evidence = text[start:end].strip()
                    flags.append(
                        {
                            "flag":        flag_name,
                            "severity":    severity,
                            "evidence":    evidence,
                            "source_tool": "data_agent.get_filing_sections",
                        }
                    )
                    matched = True
                    break

    return flags


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

async def run_all_checks(
    cik: str,
    ratio_history: list[dict],
    mdna_text: str,
    risk_factors_text: str,
    days: int = 90,
) -> list[dict]:
    """Run all three risk detectors and concatenate their results.

    Parameters
    ----------
    days:
        Look-back window for the insider-selling check.  Default 90 (one
        full quarter) rather than 30 — verified against NVDA Form 4 data
        where a genuine 3-seller cluster only surfaces at the 90-day window.

    Each detector is individually fault-tolerant: an exception in one
    is logged and skipped so the others still run.  Partial results are
    always better than total silence.

    Output is a flat list[dict] matching DueDiligenceState["risk_flags"].
    """
    all_flags: list[dict] = []

    try:
        all_flags.extend(detect_debt_spike(ratio_history))
    except Exception as exc:
        _log.error("detect_debt_spike failed for cik=%s: %s", cik, exc)

    try:
        all_flags.extend(await detect_insider_selling_cluster(cik, days=days))
    except Exception as exc:
        _log.error("detect_insider_selling_cluster failed for cik=%s: %s", cik, exc)

    try:
        all_flags.extend(flag_audit_language(mdna_text, risk_factors_text))
    except Exception as exc:
        _log.error("flag_audit_language failed for cik=%s: %s", cik, exc)

    return all_flags
