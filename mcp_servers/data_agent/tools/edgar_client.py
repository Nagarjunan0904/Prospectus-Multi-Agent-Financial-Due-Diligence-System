"""
SEC EDGAR async client with Postgres-backed 24 h caching.

Tables (auto-created on first import, idempotent):
  ticker_cache  – full SEC ticker→CIK map, refreshed once per 24 h
  filing_cache  – per-CIK submissions / company-facts blobs, 24 h TTL

Rate limiting: ONE module-level asyncio.Semaphore(8) shared across all
three public coroutines, keeping concurrent EDGAR requests ≤ 8 (SEC cap
is ~10 req/s).

Requires Python ≥ 3.10 (asyncio.Semaphore safe to create outside a loop).
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import Column, DateTime, String, Text, select

from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.db import Base, SessionLocal, engine


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------


class EdgarAPIError(Exception):
    """Non-200 response or timeout from the SEC EDGAR API."""

    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"EDGAR API returned HTTP {status_code} for {url}")


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class TickerCache(Base):
    __tablename__ = "ticker_cache"

    ticker = Column(String, primary_key=True)
    cik = Column(String(10), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class FilingCache(Base):
    __tablename__ = "filing_cache"

    cik = Column(String(10), primary_key=True)
    endpoint = Column(String(64), primary_key=True)
    response_json = Column(Text, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)


# Create tables on first import — idempotent, skips existing tables.
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Module-level shared rate-limit semaphore
# ---------------------------------------------------------------------------

_semaphore = asyncio.Semaphore(8)
_TTL = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {"User-Agent": os.environ["SEC_EDGAR_USER_AGENT"]}


def _is_fresh(ts: datetime) -> bool:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - ts < _TTL


async def _fetch_json(url: str) -> object:
    """GET *url* under the shared semaphore; raise EdgarAPIError on failure."""
    async with _semaphore:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=_headers())
        except httpx.TimeoutException:
            raise EdgarAPIError(408, url)
    if resp.status_code != 200:
        raise EdgarAPIError(resp.status_code, url)
    return resp.json()


async def _load_filing_cache(cik: str, endpoint: str) -> dict | None:
    def _query() -> dict | None:
        with SessionLocal() as session:
            row = session.execute(
                select(FilingCache).where(
                    FilingCache.cik == cik,
                    FilingCache.endpoint == endpoint,
                )
            ).scalar_one_or_none()
            if row and _is_fresh(row.fetched_at):
                return json.loads(row.response_json)
        return None

    return await asyncio.to_thread(_query)


async def _store_filing_cache(cik: str, endpoint: str, data: dict) -> None:
    payload = json.dumps(data)
    now = datetime.now(timezone.utc)

    def _upsert() -> None:
        with SessionLocal() as session:
            stmt = (
                pg_insert(FilingCache)
                .values(
                    cik=cik,
                    endpoint=endpoint,
                    response_json=payload,
                    fetched_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["cik", "endpoint"],
                    set_={"response_json": payload, "fetched_at": now},
                )
            )
            session.execute(stmt)
            session.commit()

    await asyncio.to_thread(_upsert)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve_cik(ticker: str) -> str:
    """Return the zero-padded 10-digit CIK for *ticker*.

    Checks ``ticker_cache`` first.  If the whole cache was refreshed within
    the last 24 h and the ticker is absent, raises ValueError immediately
    (no extra SEC hit).  Raises :exc:`EdgarAPIError` on network failure.
    """
    ticker = ticker.upper().strip()

    def _check_cache() -> tuple[str | None, bool]:
        """(cik_or_None, cache_is_globally_fresh)"""
        with SessionLocal() as session:
            # Freshness: look at the most-recently updated row in the table
            newest = session.execute(
                select(TickerCache).order_by(TickerCache.updated_at.desc()).limit(1)
            ).scalar_one_or_none()
            cache_fresh = newest is not None and _is_fresh(newest.updated_at)

            match = session.execute(
                select(TickerCache).where(TickerCache.ticker == ticker)
            ).scalar_one_or_none()

            return (match.cik if match else None), cache_fresh

    cik, cache_fresh = await asyncio.to_thread(_check_cache)

    if cik is not None:
        return cik

    if cache_fresh:
        # Full ticker list was fetched recently; ticker simply doesn't exist.
        raise ValueError(
            f"Ticker {ticker!r} not found in SEC EDGAR company_tickers.json"
        )

    # Cache is stale or empty — refresh from SEC.
    url = "https://www.sec.gov/files/company_tickers.json"
    data = await _fetch_json(url)

    now = datetime.now(timezone.utc)
    ticker_map: dict[str, str] = {
        str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
        for v in data.values()
    }

    def _upsert_all() -> None:
        with SessionLocal() as session:
            for t, c in ticker_map.items():
                stmt = (
                    pg_insert(TickerCache)
                    .values(ticker=t, cik=c, updated_at=now)
                    .on_conflict_do_update(
                        index_elements=["ticker"],
                        set_={"cik": c, "updated_at": now},
                    )
                )
                session.execute(stmt)
            session.commit()

    await asyncio.to_thread(_upsert_all)

    if ticker not in ticker_map:
        raise ValueError(
            f"Ticker {ticker!r} not found in SEC EDGAR company_tickers.json"
        )
    return ticker_map[ticker]


async def get_submissions(cik: str) -> dict:
    """Return the raw submissions JSON for zero-padded *cik*.

    Shape: ``{cik, entityType, sic, name, filings: {recent: {form, filingDate,
    accessionNumber, ...}}}``

    Cached 24 h in ``filing_cache``.  Raises :exc:`EdgarAPIError` on failure.
    """
    endpoint = "submissions"
    cached = await _load_filing_cache(cik, endpoint)
    if cached is not None:
        return cached

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = await _fetch_json(url)
    await _store_filing_cache(cik, endpoint, data)
    return data


async def get_recent_filings(
    cik: str,
    form_types: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return the most recent *limit* filings for *cik*.

    Uses the cached submissions JSON — no extra SEC call.
    Submissions are already newest-first; returns at most *limit* entries.
    Filters to *form_types* when provided (e.g. ``['10-K', '10-Q']``);
    returns all form types when *form_types* is ``None``.
    """
    subs = await get_submissions(cik)
    recent = subs.get("filings", {}).get("recent", {})

    forms: list[str] = recent.get("form", [])
    filing_dates: list[str] = recent.get("filingDate", [])
    acc_numbers: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])
    descriptions: list[str] = recent.get("primaryDocDescription", [])

    results: list[dict] = []
    for i, form in enumerate(forms):
        if form_types is not None and form not in form_types:
            continue
        results.append(
            {
                "form": form,
                "filingDate": filing_dates[i] if i < len(filing_dates) else "",
                "accessionNumber": acc_numbers[i] if i < len(acc_numbers) else "",
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
                "primaryDocDescription": descriptions[i] if i < len(descriptions) else "",
            }
        )
        if len(results) >= limit:
            break

    return results


async def get_company_facts(cik: str) -> dict:
    """Return all us-gaap XBRL concepts for *cik* with historical values.

    Shape: ``{cik, entityName, facts: {us-gaap: {<concept>: {label, description,
    units: {<unit>: [{end, val, accn, fy, fp, form, filed, frame}]}}}}}``

    Cached 24 h in ``filing_cache``.  Raises :exc:`EdgarAPIError` on failure.
    """
    endpoint = "company_facts"
    cached = await _load_filing_cache(cik, endpoint)
    if cached is not None:
        return cached

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data = await _fetch_json(url)
    await _store_filing_cache(cik, endpoint, data)
    return data
