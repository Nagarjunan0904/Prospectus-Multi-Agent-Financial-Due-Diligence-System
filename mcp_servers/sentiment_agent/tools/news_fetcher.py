"""
Alpha Vantage NEWS_SENTIMENT fetcher with aggressive DB caching.

RATE-LIMIT BUDGET: Alpha Vantage free tier allows 25 requests/day TOTAL,
shared across every process on the machine.  A single re-run of a test
suite without a cache hit could burn several calls instantly.  The 6-hour
TTL is intentionally much longer than the 15-min price cache or 24-h
filing cache — treat every cache miss as expensive.

Cache design
------------
``news_cache`` table: one row per ticker (PK), storing the *raw API
response JSON*.  Freshness is checked against ``fetched_at``; if the row
is < 6 hours old the stored JSON is re-parsed rather than hitting the API.

When a fresh response arrives it *replaces* the old one (ON CONFLICT DO
UPDATE) — new headlines should push out old ones.

Filtering by ``days`` is applied at parse time against the current clock,
not the fetch clock.  This means:

  • If the cache was populated with days=14 and the caller requests days=7,
    the 7-day subset is returned from cache correctly.
  • If the caller requests days=30 but the cache was only populated for 14,
    only 14 days will be returned.  Callers that need a longer horizon should
    request it on the first call so the cache is pre-filled with enough data.

Error handling
--------------
Alpha Vantage returns HTTP 200 even when rate-limited.  The actual signal
is a ``"Note"`` or ``"Information"`` key in the JSON.  We raise
``NewsAPIError`` instead of silently treating it as zero headlines.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.db import Base, SessionLocal, engine

_log = logging.getLogger(__name__)

_NEWS_TTL = timedelta(hours=6)
_AV_URL = (
    "https://www.alphavantage.co/query"
    "?function=NEWS_SENTIMENT"
    "&tickers={ticker}"
    "&time_from={time_from}"
    "&limit=200"
    "&apikey={key}"
)


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------

class NewsAPIError(Exception):
    """Raised when Alpha Vantage signals a rate-limit or API error via JSON."""


# ---------------------------------------------------------------------------
# news_cache table
# ---------------------------------------------------------------------------

class NewsCache(Base):
    __tablename__ = "news_cache"

    ticker = Column(String(16), primary_key=True)
    response_json = Column(Text, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Sync DB helpers (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _cache_is_fresh(fetched_at: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (now - fetched_at) < _NEWS_TTL


def _load_news_cache(ticker: str) -> dict | None:
    """Return parsed response JSON if cache row exists and is fresh, else None."""
    with SessionLocal() as session:
        row = session.execute(
            select(NewsCache).where(NewsCache.ticker == ticker.upper())
        ).scalar_one_or_none()
        if row is None or not _cache_is_fresh(row.fetched_at):
            return None
        return json.loads(row.response_json)


def _store_news_cache(ticker: str, response: dict) -> None:
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        session.execute(
            pg_insert(NewsCache)
            .values(
                ticker=ticker.upper(),
                response_json=json.dumps(response),
                fetched_at=now,
            )
            .on_conflict_do_update(
                index_elements=["ticker"],
                set_={"response_json": json.dumps(response), "fetched_at": now},
            )
        )
        session.commit()


# ---------------------------------------------------------------------------
# Response parsing (pure, side-effect-free)
# ---------------------------------------------------------------------------

# 0.7 chosen empirically: AV scores 0.5–0.65 for articles that merely list NVDA
# among many fund holdings ("Fund X bought NVDA, AAPL, MSFT…"), diluting the signal.
_RELEVANCE_THRESHOLD = 0.7


def _relevance_for(article: dict, ticker: str) -> float | None:
    """Return the relevance_score for *ticker* in this article, or None if absent.

    Alpha Vantage may prefix tickers with their exchange (e.g. ``NASDAQ:NVDA``).
    We strip everything up to the last ``:`` before comparing, so both formats
    match the bare symbol the caller passes.
    """
    needle = ticker.upper()
    for entry in article.get("ticker_sentiment", []):
        av_ticker = (entry.get("ticker") or "").split(":")[-1].upper()
        if av_ticker == needle:
            try:
                return float(entry["relevance_score"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _parse_headlines(response: dict, days: int, ticker: str) -> list[dict]:
    """Extract headlines from a raw Alpha Vantage response dict.

    Filters to articles that:
      • were published within the last *days* days from now
      • have a ``ticker_sentiment`` entry for *ticker* with
        ``relevance_score >= 0.3`` — articles where the company is only
        a passing mention are excluded to avoid diluting the signal

    Returns [{"text": "title summary", "published_date": "YYYY-MM-DD"}, ...].
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []

    for art in response.get("feed", []):
        try:
            pub = datetime.strptime(
                art["time_published"], "%Y%m%dT%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue

        if pub < cutoff:
            continue

        # Relevance gate — skip articles where this ticker isn't the focus
        relevance = _relevance_for(art, ticker)
        if relevance is None or relevance < _RELEVANCE_THRESHOLD:
            continue

        title = (art.get("title") or "").strip()
        summary = (art.get("summary") or "").strip()
        text = f"{title} {summary}".strip() if summary else title
        if not text:
            continue

        out.append({
            "text": text,
            "published_date": pub.strftime("%Y-%m-%d"),
        })

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_headlines(
    ticker: str,
    days: int = 14,
) -> list[dict]:
    """Return recent news headlines for *ticker* from Alpha Vantage.

    Checks ``news_cache`` first (6-hour TTL).  On a cache miss, fetches
    from the Alpha Vantage NEWS_SENTIMENT endpoint and caches the raw JSON.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (e.g. ``"NVDA"``).
    days:
        Number of days of history to return.  If the cache was populated
        with a larger window, this filters the cached results.  If smaller,
        only the cached window is returned.

    Raises
    ------
    NewsAPIError
        If Alpha Vantage returns a rate-limit or error response (HTTP 200
        with a ``"Note"`` or ``"Information"`` key in the JSON body).
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")

    # Check cache first
    cached = await asyncio.to_thread(_load_news_cache, ticker)
    if cached is not None:
        _log.debug("news/%s: cache hit", ticker)
        return _parse_headlines(cached, days, ticker)

    # Cache miss → fetch from Alpha Vantage
    if not api_key:
        raise NewsAPIError("ALPHAVANTAGE_API_KEY not set")

    time_from = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).strftime("%Y%m%dT%H%M")

    url = _AV_URL.format(
        ticker=ticker.upper(),
        time_from=time_from,
        key=api_key,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
    except httpx.TimeoutException as exc:
        raise NewsAPIError(f"Alpha Vantage request timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise NewsAPIError(f"Alpha Vantage HTTP error: {exc}") from exc

    if resp.status_code != 200:
        raise NewsAPIError(
            f"Alpha Vantage returned HTTP {resp.status_code}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise NewsAPIError(f"Could not parse Alpha Vantage JSON: {exc}") from exc

    # Alpha Vantage rate-limit / error signal — HTTP 200 but JSON has
    # "Note" or "Information" instead of "feed"
    if "Note" in data:
        raise NewsAPIError(data["Note"])
    if "Information" in data:
        raise NewsAPIError(data["Information"])

    # Cache the raw response
    await asyncio.to_thread(_store_news_cache, ticker, data)
    _log.info("news/%s: fetched %d articles from Alpha Vantage", ticker, len(data.get("feed", [])))

    return _parse_headlines(data, days, ticker)
