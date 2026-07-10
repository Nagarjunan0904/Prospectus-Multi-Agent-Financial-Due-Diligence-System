"""
Memo Writer node — calls gpt-4.1-mini (via instructor) to produce a
structured InvestmentMemo from the accumulated pipeline state.

Skip condition
--------------
Unlike other nodes, memo_writer does NOT gate on ``required_agents``.
A partial memo that explicitly calls out missing data is more useful than
no memo at all.  The only hard skip is when ``company_facts`` is entirely
absent — meaning ticker resolution itself failed and there is genuinely
nothing to write about.

Context building
----------------
Each data source is rendered as a labeled block with a ``[source: ...]``
hint so the LLM can write valid ``source_field`` paths for each claim.
Sources that errored are rendered as ``unavailable — <error snippet>``
so the LLM notes the gap rather than fabricating figures.

LLM call
--------
instructor enforces the InvestmentMemo Pydantic schema, so the response
is always structurally valid even if semantically thin.  The result is
stored as ``model.model_dump()`` — a plain dict that psycopg3 can
checkpoint — never as the raw Pydantic object.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import instructor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from backend.models import InvestmentMemo
from backend.state import AgentTraceEntry, DueDiligenceState

_log = logging.getLogger(__name__)

_MODEL = "gpt-4.1-mini"
_client = instructor.from_openai(OpenAI())

_SYSTEM_PROMPT = """\
You are a senior equity analyst writing a concise, structured investment \
due-diligence memo.

Using ONLY the data provided by the user, produce a memo with EXACTLY these \
four sections in this order, using these EXACT heading strings verbatim \
(any variation will break downstream validation):

    "Financial Snapshot"
    "Sentiment"
    "Risk Factors"
    "Recommendation"

Each section must contain at least one claim. For every claim set \
``source_field`` to the dotted path in the supplied data where the evidence \
lives.  Use the ``[source: ...]`` hints in the user message to construct \
valid paths.  Examples:
  • claim about P/E           → source_field = "ratios.pe_ratio"
  • claim about sentiment pct → source_field = "sentiment.positive_pct"
  • claim about a risk flag   → source_field = "risk_flags"
  • claim citing MD&A text    → source_field = "filing_sections.7"
  • claim citing D/E trend    → source_field = "ratio_history"
  • claim citing insider data → source_field = "insider_summary"

If a source is marked "unavailable", state that it could not be assessed — \
do NOT invent figures, percentages, or flag names.

The Recommendation section must open with exactly one of: BUY / HOLD / SELL, \
followed by a one-sentence rationale grounded in the data above.
"""

_TEXT_LIMIT = 2_500   # chars per filing section sent to the LLM


# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def _find_error(errors: list[str], keyword: str) -> str:
    return next((e for e in errors if keyword in e), "no data returned")


def _fmt_ratios(ratios: dict[str, Any]) -> str:
    # Label is display-only; the parenthesised path is what the LLM must use
    # as source_field when citing this figure.
    mapping = {
        "pe_ratio":           "P/E ratio",
        "debt_to_equity":     "Debt/Equity",
        "current_ratio":      "Current ratio",
        "gross_margin":       "Gross margin",
        "operating_margin":   "Operating margin",
        "net_margin":         "Net margin",
        "revenue_growth_yoy": "Revenue growth (YoY)",
        "revenue_growth_qoq": "Revenue growth (QoQ)",
    }
    lines = []
    for key, label in mapping.items():
        val = ratios.get(key)
        if val is None:
            continue
        path = f"ratios.{key}"
        if "margin" in key or "growth" in key:
            lines.append(f"  {label} ({path}): {val * 100:.1f}%")
        else:
            lines.append(f"  {label} ({path}): {val:.2f}")
    if ratios.get("warnings"):
        lines.append("  Warnings (ratios.warnings): " + "; ".join(ratios["warnings"][:2]))
    return "\n".join(lines) or "  (no recognised ratio fields)"


def _fmt_ratio_history(history: list[dict[str, Any]]) -> str:
    # The list is cited as "ratio_history"; individual rows have key "debt_to_equity".
    return "\n".join(
        f"  {row.get('period_end', '?')}: debt_to_equity = {row.get('debt_to_equity')}"
        for row in history
    ) or "  (empty)"


def _fmt_sentiment(s: dict[str, Any]) -> str:
    return (
        f"  Positive (sentiment.positive_pct): {s.get('positive_pct', 0) * 100:.1f}%\n"
        f"  Neutral (sentiment.neutral_pct): {s.get('neutral_pct', 0) * 100:.1f}%\n"
        f"  Negative (sentiment.negative_pct): {s.get('negative_pct', 0) * 100:.1f}%\n"
        f"  Headlines analysed (sentiment.headline_count): {s.get('headline_count', 0)}"
    )


def _fmt_risk_flags(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "  [none found]"
    return "\n".join(
        f"  [{f.get('severity', '?').upper()}] {f.get('flag')}: "
        f"{str(f.get('evidence', ''))[:120]}"
        for f in flags
    )


def _fmt_insider(summary: dict[str, Any]) -> str:
    keys = ("buy_count", "sell_count", "net_shares", "distinct_buyers", "distinct_sellers")
    parts = [f"insider_summary.{k}: {summary[k]}" for k in keys if k in summary]
    return "  " + ", ".join(parts) if parts else "  (empty summary)"


_MISSING = object()  # sentinel returned when a source_field path doesn't resolve


def _resolve_source_field(source_field: str, state: DueDiligenceState) -> Any:
    """Walk a dotted source_field path into state.

    Returns the resolved value, or _MISSING if any segment fails.
    Treats None and empty containers as resolved (grounding is checked
    separately by self_critic); here we only care that the path is valid.
    """
    parts = source_field.split(".") if source_field else []
    if not parts:
        return _MISSING
    try:
        obj: Any = state[parts[0]]  # type: ignore[literal-required]
        for part in parts[1:]:
            try:
                obj = obj[part]
            except (KeyError, TypeError, IndexError):
                obj = getattr(obj, part)
        return obj
    except (KeyError, AttributeError, TypeError, IndexError, ValueError):
        return _MISSING


def _build_context(state: DueDiligenceState) -> str:
    ticker = state.get("ticker", "UNKNOWN")
    cik    = state.get("cik", "unknown")
    errors = state.get("errors") or []
    fs     = state.get("filing_sections") or {}

    blocks: list[str] = [f"COMPANY: {ticker}  (CIK: {cik})\n"]

    # ── Quant ratios ─────────────────────────────────────────────────────────
    blocks.append("QUANT RATIOS:  [source: ratios.<field_name>]")
    if state.get("ratios"):
        blocks.append(_fmt_ratios(state["ratios"]))  # type: ignore[arg-type]
    else:
        blocks.append(f"  unavailable — {_find_error(errors, 'compute_ratios')}")

    # ── D/E ratio history ─────────────────────────────────────────────────────
    blocks.append("\nD/E RATIO HISTORY (annual 10-K):  [source: ratio_history]")
    if state.get("ratio_history"):
        blocks.append(_fmt_ratio_history(state["ratio_history"]))  # type: ignore[arg-type]
    else:
        blocks.append(f"  unavailable — {_find_error(errors, 'get_ratio_history')}")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    blocks.append("\nSENTIMENT (FinBERT, 14-day headlines):  [source: sentiment.<field_name>]")
    if state.get("sentiment"):
        blocks.append(_fmt_sentiment(state["sentiment"]))  # type: ignore[arg-type]
    else:
        blocks.append(f"  unavailable — {_find_error(errors, 'get_sentiment_summary')}")

    # ── Risk flags ────────────────────────────────────────────────────────────
    # risk_flags is a list; empty list = "no flags found" (not missing)
    blocks.append("\nRISK FLAGS:  [source: risk_flags]")
    if state.get("risk_flags") is not None:
        blocks.append(_fmt_risk_flags(state["risk_flags"]))  # type: ignore[arg-type]
    else:
        blocks.append(f"  unavailable — {_find_error(errors, 'run_all_checks')}")

    # ── Insider activity ──────────────────────────────────────────────────────
    blocks.append("\nINSIDER ACTIVITY (90-day):  [source: insider_summary]")
    if state.get("insider_summary"):
        blocks.append(_fmt_insider(state["insider_summary"]))  # type: ignore[arg-type]
    else:
        blocks.append(f"  unavailable — {_find_error(errors, 'get_insider_transactions')}")

    # ── MD&A excerpt ──────────────────────────────────────────────────────────
    blocks.append(f"\nMD&A EXCERPT (Item 7, first {_TEXT_LIMIT} chars):  [source: filing_sections.7]")
    if fs.get("7"):
        blocks.append(f"  {fs['7'][:_TEXT_LIMIT]}")
    else:
        blocks.append("  unavailable — filing sections not fetched")

    # ── Risk Factors excerpt ──────────────────────────────────────────────────
    blocks.append(f"\nRISK FACTORS EXCERPT (Item 1A, first {_TEXT_LIMIT} chars):  [source: filing_sections.1A]")
    if fs.get("1A"):
        blocks.append(f"  {fs['1A'][:_TEXT_LIMIT]}")
    else:
        blocks.append("  unavailable — filing sections not fetched")

    # ── Error summary ─────────────────────────────────────────────────────────
    if errors:
        blocks.append(f"\nAGENT ERRORS ({len(errors)} total — first 5):")
        for e in errors[:5]:
            blocks.append(f"  • {e[:120]}")

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _append_trace(
    agent_trace: list[AgentTraceEntry],
    tool: str | None,
    status: str,
    latency_ms: float | None = None,
) -> None:
    entry: AgentTraceEntry = {"node": "memo_writer", "tool": tool, "status": status}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    agent_trace.append(entry)


async def run(state: DueDiligenceState) -> dict[str, Any]:
    """LangGraph node: generate a structured InvestmentMemo via gpt-4.1-mini."""
    errors: list[str] = []
    agent_trace: list[AgentTraceEntry] = []

    # Hard skip: if there is no company data at all (ticker resolution failed),
    # a memo would be entirely fabricated — skip it entirely.
    # All other upstream failures produce partial data and a partial memo is
    # more useful than none, so we do NOT gate on required_agents here.
    if not state.get("company_facts"):
        _append_trace(agent_trace, None, "skipped")
        _log.info("memo_writer: skipping — company_facts empty (ticker resolution failed)")
        return {"agent_trace": agent_trace}

    ticker = state.get("ticker", "UNKNOWN")
    context = _build_context(state)

    t0 = time.monotonic()
    try:
        memo_obj: InvestmentMemo = _client.chat.completions.create(
            model=_MODEL,
            response_model=InvestmentMemo,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": context},
            ],
        )
        memo_dict = memo_obj.model_dump()
        if not memo_dict.get("ticker"):
            memo_dict["ticker"] = ticker

        # Strip claims whose source_field doesn't resolve in the actual state.
        # This catches hallucinated paths (e.g. "ratios.debt_equity" instead of
        # "ratios.debt_to_equity") before they reach the checkpointer.
        # self_critic will detect any resulting empty sections and route back here.
        stripped_count = 0
        bad_paths: list[str] = []
        for section in memo_dict.get("sections", []):
            valid: list[dict[str, Any]] = []
            for claim in section.get("claims", []):
                sf = claim.get("source_field", "")
                if _resolve_source_field(sf, state) is not _MISSING:
                    valid.append(claim)
                else:
                    bad_paths.append(sf or "(empty)")
                    stripped_count += 1
            section["claims"] = valid
        if stripped_count:
            _log.warning(
                "memo_writer: stripped %d claim(s) with bad source_field: %s",
                stripped_count, bad_paths,
            )
            errors.append(
                f"memo_writer: stripped {stripped_count} claim(s) with unresolvable "
                f"source_field — {', '.join(bad_paths[:5])}"
            )

        latency_ms = (time.monotonic() - t0) * 1000
        _append_trace(agent_trace, _MODEL, "success", latency_ms)
        _log.info(
            "memo_writer: memo generated for %s — %d section(s), %d claim(s) stripped, %.0f ms",
            ticker, len(memo_dict.get("sections", [])), stripped_count, latency_ms,
        )
        return {
            "memo":        memo_dict,
            "errors":      errors,
            "agent_trace": agent_trace,
        }

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        _append_trace(agent_trace, _MODEL, "error", latency_ms)
        errors.append(f"memo_writer: {exc}")
        _log.warning("memo_writer: LLM call failed for %s — %s", ticker, exc)
        return {
            "errors":      errors,
            "agent_trace": agent_trace,
        }
