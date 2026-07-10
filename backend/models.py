"""
Pydantic structured-output schemas for the due-diligence pipeline.

InvestmentMemo
--------------
Produced by memo_writer.py via instructor and consumed by self_critic.py.
Store as ``memo_obj.model_dump()`` in ``state['memo']`` — never as the raw
Pydantic object, which psycopg3 cannot checkpoint.

source_field format
-------------------
Each Claim carries a ``source_field`` — a dotted path into DueDiligenceState
that self_critic uses to verify the claim is grounded in actual pipeline data:

    "ratios.pe_ratio"           → state["ratios"]["pe_ratio"]
    "sentiment.positive_pct"    → state["sentiment"]["positive_pct"]
    "risk_flags"                → state["risk_flags"]  (the list itself)
    "filing_sections.7"         → state["filing_sections"]["7"]
    "ratio_history"             → state["ratio_history"]
    "insider_summary"           → state["insider_summary"]

The first path component doubles as the key for self_critic's per-agent
ungrounded-claim tracking via _STATE_FIELD_TO_AGENT in self_critic.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single factual claim tied to its verifiable data source."""

    text: str = Field(description="The claim in one or two plain-English sentences.")
    source_field: str = Field(
        description=(
            "Dotted path into DueDiligenceState where supporting data lives. "
            "Examples: 'ratios.pe_ratio', 'sentiment.positive_pct', 'risk_flags', "
            "'filing_sections.7'. The first segment must be a top-level state key."
        )
    )


class MemoSection(BaseModel):
    """One of the four required sections of an InvestmentMemo."""

    heading: str = Field(
        description=(
            "Section heading — must be one of these four strings, verbatim: "
            "'Financial Snapshot', 'Sentiment', 'Risk Factors', 'Recommendation'."
        )
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="One or more claims that support this section's narrative.",
    )


class InvestmentMemo(BaseModel):
    """Structured due-diligence memo with exactly four fixed sections.

    The LLM is instructed to produce all four required sections.
    self_critic evaluates completeness (all four headings present, each
    with ≥ 1 claim) and groundedness (each claim's source_field resolves
    to non-empty pipeline data).
    """

    ticker: str
    sections: list[MemoSection]
