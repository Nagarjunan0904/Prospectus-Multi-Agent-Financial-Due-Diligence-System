"""
FinBERT sentiment scoring and aggregation.

Model loading
-------------
ProsusAI/finbert is loaded ONCE at module import time via the
``transformers.pipeline`` API.  Cold-start cost:

  • GPU (CUDA): ~1–2 s for model weights to transfer to VRAM.
  • CPU only:   ~4–8 s on a modern multi-core machine.

In HTTP server mode (python -m mcp_servers.sentiment_agent.server) the
process stays alive, so this cost is paid once per server lifetime.

In stdio mode (--transport stdio) a new process is spawned for *every*
Claude Desktop tool call, so the model is re-loaded each time.  For
production or latency-sensitive use, always prefer HTTP transport for
the sentiment agent.

Out-of-scope note
-----------------
``get_sentiment_summary`` intentionally omits ``dominant_themes``.
Extracting per-topic themes requires a topic model (BERTopic, LDA, etc.)
that FinBERT does not provide, and adding one is a significant scope
increase with diminishing analytical return relative to the per-headline
label distribution.  The absence is a deliberate design choice, not an
oversight.

Public API
----------
score_sentiment(texts)          → sync, internal only; not exposed as MCP tool
get_sentiment_summary(ticker)   → async; shape matches DueDiligenceState["sentiment"]
get_sentiment_trend(ticker)     → async; bucketed time series for the SentimentGauge chart
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import torch
from transformers import pipeline

from mcp_servers.sentiment_agent.tools.news_fetcher import NewsAPIError, fetch_headlines

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model — loaded once at import time
# ---------------------------------------------------------------------------

_device = 0 if torch.cuda.is_available() else -1
_log.info(
    "Loading ProsusAI/finbert on %s (device=%s) …",
    "GPU" if _device == 0 else "CPU",
    _device,
)

# pipeline() triggers model + tokenizer download on first use.
# Subsequent imports reuse the on-disk HuggingFace cache (~438 MB).
_pipe = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=_device,
)

_log.info("ProsusAI/finbert ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compound(result: dict) -> float:
    """Map FinBERT label → signed compound score in [-1, 1].

    positive → +confidence_score
    negative → -confidence_score
    neutral  → 0.0  (no directional signal regardless of confidence)
    """
    label = result["label"].lower()
    score: float = result["score"]
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


# ---------------------------------------------------------------------------
# score_sentiment — internal; not exposed as an MCP tool
# ---------------------------------------------------------------------------

def score_sentiment(texts: list[str]) -> list[dict]:
    """Run ProsusAI/finbert over *texts* in batches of 16.

    Returns [{"label": "positive"|"neutral"|"negative", "score": float}, ...]
    in the same order as *texts*.  Long texts are truncated to 512 tokens.

    This is a synchronous function.  Callers inside an async context must
    wrap it with ``asyncio.to_thread(score_sentiment, texts)``.
    """
    if not texts:
        return []
    results = _pipe(texts, batch_size=16, truncation=True)
    return [{"label": r["label"].lower(), "score": float(r["score"])} for r in results]


# ---------------------------------------------------------------------------
# get_sentiment_summary
# ---------------------------------------------------------------------------

async def get_sentiment_summary(
    ticker: str,
    days: int = 14,
) -> dict:
    """Fetch headlines and aggregate FinBERT scores into a summary dict.

    Return shape (matches DueDiligenceState["sentiment"]):
    {
        "positive_pct":  float,   # fraction of headlines labelled positive
        "neutral_pct":   float,
        "negative_pct":  float,
        "headline_count": int,
        "trend":          list[float],  # daily avg compound score, oldest→newest
    }

    Returns zeroed-out dict if no headlines are available (not an error).
    Does not include ``dominant_themes`` — see module docstring for rationale.
    """
    _EMPTY = {
        "positive_pct": 0.0,
        "neutral_pct": 0.0,
        "negative_pct": 0.0,
        "headline_count": 0,
        "trend": [],
    }

    headlines = await fetch_headlines(ticker, days)
    if not headlines:
        _log.info("sentiment/%s: no headlines in last %d days", ticker, days)
        return _EMPTY

    texts = [h["text"] for h in headlines]
    # FinBERT inference is CPU/GPU-bound; run in thread pool to keep event loop free
    scores = await asyncio.to_thread(score_sentiment, texts)

    total = len(scores)
    pos = neg = neu = 0
    daily: dict[str, list[float]] = defaultdict(list)

    for headline, result in zip(headlines, scores):
        label = result["label"]
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1
        daily[headline["published_date"]].append(_compound(result))

    trend = [
        round(sum(vals) / len(vals), 4)
        for _date, vals in sorted(daily.items())  # oldest → newest
    ]

    return {
        "positive_pct": round(pos / total, 4),
        "neutral_pct":  round(neu / total, 4),
        "negative_pct": round(neg / total, 4),
        "headline_count": total,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# get_sentiment_trend
# ---------------------------------------------------------------------------

async def get_sentiment_trend(
    ticker: str,
    days: int = 30,
) -> list[dict]:
    """Fetch headlines and return a day-by-day sentiment time series.

    Used by the SentimentGauge chart in Phase 5.

    Return shape:
    [
        {"date": "YYYY-MM-DD", "compound_score": float, "headline_count": int},
        ...  # sorted oldest → newest, one entry per day that has headlines
    ]

    Days with no headlines are omitted (gaps are meaningful — the chart
    can render them as absent data points rather than false zeros).
    """
    headlines = await fetch_headlines(ticker, days)
    if not headlines:
        return []

    texts = [h["text"] for h in headlines]
    scores = await asyncio.to_thread(score_sentiment, texts)

    daily: dict[str, list[float]] = defaultdict(list)
    for headline, result in zip(headlines, scores):
        daily[headline["published_date"]].append(_compound(result))

    return [
        {
            "date": date,
            "compound_score": round(sum(vals) / len(vals), 4),
            "headline_count": len(vals),
        }
        for date, vals in sorted(daily.items())
    ]
