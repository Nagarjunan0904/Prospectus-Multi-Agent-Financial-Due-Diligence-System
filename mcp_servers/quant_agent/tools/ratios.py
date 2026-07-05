"""
Financial ratio computation from SEC EDGAR XBRL company_facts.

Public API
----------
compute_ratios(company_facts, ticker=None) -> dict
    Each ratio is computed independently in its own try/except.
    A missing XBRL concept → None for that ratio only.
    Returns {ratio_name: value_or_None, ..., "warnings": [str, ...]}.

compare_peers(ticker, peer_tickers) -> dict
    Runs compute_ratios for the main ticker + all peers.
    Fetches company_facts from the Data Agent MCP server for each.
    Returns {"table": [...sorted by market_cap desc...], "warnings": [...]}.

Price data
----------
Latest regularMarketPrice fetched from Yahoo Finance chart API (unofficial),
cached 15 min in ``price_cache`` (ticker PK; ON CONFLICT DO UPDATE).

Note: spec originally called for Stooq CSV API, but Stooq's /q/l/ endpoint
returns 404 for all URL variants as of 2026-07.  Yahoo Finance's v8/chart
endpoint returns the same last-session close and is freely accessible.

XBRL concept mapping
--------------------
P/E         EarningsPerShareDiluted → EarningsPerShareBasic (fallback)
D/E         LongTermDebtNoncurrent + LongTermDebtCurrent → LongTermDebt (fallback sum)
            / StockholdersEquity → StockholdersEquityAttributableToParent (fallback)
Current     AssetsCurrent / LiabilitiesCurrent  (any form type)
Margins     Revenues → RevenueFromContractWithCustomerExcludingAssessedTax → SalesRevenueNet
            CostOfRevenue → CostOfGoodsAndServicesSold
            OperatingIncomeLoss, NetIncomeLoss
Rev growth  frame-filtered: '^CY\\d{4}$' (annual YoY), '^CY\\d{4}Q\\d$' (quarterly QoQ)
Mkt cap     dei:EntityCommonStockSharesOutstanding × Yahoo close
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import Column, DateTime, Float, String, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.db import Base, SessionLocal, engine
from mcp_clients.data_client import EdgarAPIError, get_company_facts, resolve_cik

_log = logging.getLogger(__name__)

_PRICE_TTL = timedelta(minutes=15)
# Yahoo Finance chart API — unofficial but reliable; same last-session close as Stooq
_YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d"
_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

class PriceCache(Base):
    __tablename__ = "price_cache"

    ticker = Column(String(16), primary_key=True)
    price = Column(Float, nullable=False)
    price_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    fetched_at = Column(DateTime(timezone=True), nullable=False)


Base.metadata.create_all(bind=engine)


def _price_is_fresh(fetched_at: datetime) -> bool:
    now = datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (now - fetched_at) < _PRICE_TTL


def _load_price_cache(ticker: str) -> float | None:
    with SessionLocal() as session:
        row = session.execute(
            select(PriceCache).where(PriceCache.ticker == ticker.upper())
        ).scalar_one_or_none()
        if row is None or not _price_is_fresh(row.fetched_at):
            return None
        return row.price


def _store_price_cache(ticker: str, price: float, price_date: str) -> None:
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        session.execute(
            pg_insert(PriceCache)
            .values(ticker=ticker.upper(), price=price, price_date=price_date, fetched_at=now)
            .on_conflict_do_update(
                index_elements=["ticker"],
                set_={"price": price, "price_date": price_date, "fetched_at": now},
            )
        )
        session.commit()


async def _get_cached_price(ticker: str) -> float | None:
    """Return the latest regularMarketPrice for *ticker*, 15-min cached.

    Fetches from Yahoo Finance chart API (v8/finance/chart).  Returns None
    — never raises — if the ticker is unknown or the fetch fails.
    P/E and market-cap callers must handle None gracefully.
    """
    cached = await asyncio.to_thread(_load_price_cache, ticker)
    if cached is not None:
        return cached

    url = _YF_URL.format(ticker=ticker.upper())
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_YF_HEADERS) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            _log.warning("price/%s: Yahoo Finance HTTP %s", ticker, resp.status_code)
            return None

        data = resp.json()
        result = (data.get("chart") or {}).get("result") or []
        if not result:
            _log.warning("price/%s: Yahoo Finance returned empty result", ticker)
            return None

        meta = result[0].get("meta", {})
        price: float | None = meta.get("regularMarketPrice")
        ts: int | None = meta.get("regularMarketTime")

        if not price or price <= 0:
            _log.warning("price/%s: Yahoo Finance returned invalid price %s", ticker, price)
            return None

        price_date = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if ts else "unknown"
        )
        await asyncio.to_thread(_store_price_cache, ticker, float(price), price_date)
        return float(price)

    except (httpx.TimeoutException, httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        _log.warning("price/%s: Yahoo Finance fetch failed — %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# XBRL extraction helpers
# ---------------------------------------------------------------------------

def _get_entries(
    company_facts: dict,
    concept: str,
    namespace: str = "us-gaap",
    unit: str = "USD",
    form: str | None = None,
    frame_re: str | None = None,
) -> list[dict]:
    """Return all period entries for *concept*, optionally filtered and sorted newest-first."""
    try:
        raw: list[dict] = list(
            company_facts["facts"][namespace][concept]["units"][unit]
        )
    except (KeyError, TypeError):
        return []

    if form:
        raw = [e for e in raw if e.get("form") == form]
    if frame_re:
        pat = re.compile(frame_re)
        raw = [e for e in raw if pat.match(e.get("frame", ""))]

    return sorted(raw, key=lambda e: e["end"], reverse=True)


def _latest(
    company_facts: dict,
    concept: str,
    namespace: str = "us-gaap",
    unit: str = "USD",
    form: str | None = None,
    frame_re: str | None = None,
) -> float | None:
    entries = _get_entries(company_facts, concept, namespace, unit, form, frame_re)
    return entries[0]["val"] if entries else None


def _revenue_entries(company_facts: dict, frame_re: str) -> list[dict]:
    """Try canonical revenue concepts in order; return frame-filtered entries."""
    for concept in (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ):
        entries = _get_entries(company_facts, concept, frame_re=frame_re)
        if entries:
            return entries
    return []


def _latest_revenue(company_facts: dict, form: str | None = None) -> float | None:
    for concept in (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ):
        v = _latest(company_facts, concept, form=form)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# compute_ratios
# ---------------------------------------------------------------------------

async def compute_ratios(
    company_facts: dict,
    ticker: str | None = None,
) -> dict:
    """Compute financial ratios from EDGAR XBRL *company_facts*.

    Each ratio is computed in an independent try/except — a missing concept
    sets that ratio to None without affecting the others.

    Returns a flat dict of ratio values (float or None) plus a "warnings"
    list explaining each None.
    """
    ratios: dict[str, Any] = {}
    warnings: list[str] = []

    # ── P/E ──────────────────────────────────────────────────────────────────
    try:
        eps = (
            _latest(company_facts, "EarningsPerShareDiluted", unit="USD/shares")
            or _latest(company_facts, "EarningsPerShareBasic", unit="USD/shares")
        )
        if ticker is None:
            ratios["pe_ratio"] = None
            warnings.append("pe_ratio: ticker not provided, cannot fetch Stooq price")
        elif eps is None:
            ratios["pe_ratio"] = None
            warnings.append("pe_ratio: EarningsPerShareDiluted/Basic absent")
        elif eps == 0:
            ratios["pe_ratio"] = None
            warnings.append("pe_ratio: EPS is zero")
        else:
            price = await _get_cached_price(ticker)
            if price is None:
                ratios["pe_ratio"] = None
                warnings.append("pe_ratio: Stooq price fetch failed")
            else:
                ratios["pe_ratio"] = round(price / eps, 2)
    except Exception as exc:
        ratios["pe_ratio"] = None
        warnings.append(f"pe_ratio: {exc}")

    # ── Debt / Equity (annual 10-K only) ─────────────────────────────────────
    try:
        lt_noncurrent = _latest(company_facts, "LongTermDebtNoncurrent", form="10-K")
        lt_current = _latest(company_facts, "LongTermDebtCurrent", form="10-K")

        if lt_noncurrent is None and lt_current is None:
            # Some filers report a single LongTermDebt line without splitting
            consolidated = _latest(company_facts, "LongTermDebt", form="10-K")
            if consolidated is None:
                total_debt = 0.0
                warnings.append(
                    "debt_to_equity: LongTermDebt concepts not found in filing, "
                    "defaulted to 0 — may understate actual debt"
                )
            else:
                total_debt = consolidated
        else:
            total_debt = (lt_noncurrent or 0.0) + (lt_current or 0.0)

        equity = (
            _latest(company_facts, "StockholdersEquity", form="10-K")
            or _latest(company_facts, "StockholdersEquityAttributableToParent", form="10-K")
        )

        if equity is None:
            ratios["debt_to_equity"] = None
            warnings.append("debt_to_equity: StockholdersEquity absent")
        elif equity == 0:
            ratios["debt_to_equity"] = None
            warnings.append("debt_to_equity: equity is zero")
        else:
            ratios["debt_to_equity"] = round(total_debt / equity, 4)
    except Exception as exc:
        ratios["debt_to_equity"] = None
        warnings.append(f"debt_to_equity: {exc}")

    # ── Current ratio (most recent period, any form) ──────────────────────────
    try:
        assets_cur = _latest(company_facts, "AssetsCurrent")
        liab_cur = _latest(company_facts, "LiabilitiesCurrent")
        if assets_cur is None or liab_cur is None:
            ratios["current_ratio"] = None
            warnings.append("current_ratio: AssetsCurrent or LiabilitiesCurrent absent")
        elif liab_cur == 0:
            ratios["current_ratio"] = None
            warnings.append("current_ratio: LiabilitiesCurrent is zero")
        else:
            ratios["current_ratio"] = round(assets_cur / liab_cur, 4)
    except Exception as exc:
        ratios["current_ratio"] = None
        warnings.append(f"current_ratio: {exc}")

    # ── Margins (annual 10-K) ─────────────────────────────────────────────────
    try:
        revenues = _latest_revenue(company_facts, form="10-K")
        if not revenues:
            for k in ("gross_margin", "operating_margin", "net_margin"):
                ratios[k] = None
            warnings.append("margins: Revenues absent or zero in 10-K filings")
        else:
            # Gross margin
            try:
                cogs = (
                    _latest(company_facts, "CostOfRevenue", form="10-K")
                    or _latest(company_facts, "CostOfGoodsAndServicesSold", form="10-K")
                )
                if cogs is None:
                    ratios["gross_margin"] = None
                    warnings.append(
                        "gross_margin: CostOfRevenue absent (common for software/services)"
                    )
                else:
                    ratios["gross_margin"] = round((revenues - cogs) / revenues, 4)
            except Exception as exc:
                ratios["gross_margin"] = None
                warnings.append(f"gross_margin: {exc}")

            # Operating margin
            try:
                op_inc = _latest(company_facts, "OperatingIncomeLoss", form="10-K")
                if op_inc is None:
                    ratios["operating_margin"] = None
                    warnings.append("operating_margin: OperatingIncomeLoss absent")
                else:
                    ratios["operating_margin"] = round(op_inc / revenues, 4)
            except Exception as exc:
                ratios["operating_margin"] = None
                warnings.append(f"operating_margin: {exc}")

            # Net margin
            try:
                net_inc = _latest(company_facts, "NetIncomeLoss", form="10-K")
                if net_inc is None:
                    ratios["net_margin"] = None
                    warnings.append("net_margin: NetIncomeLoss absent")
                else:
                    ratios["net_margin"] = round(net_inc / revenues, 4)
            except Exception as exc:
                ratios["net_margin"] = None
                warnings.append(f"net_margin: {exc}")
    except Exception as exc:
        for k in ("gross_margin", "operating_margin", "net_margin"):
            ratios[k] = None
        warnings.append(f"margins: {exc}")

    # ── Revenue growth YoY (annual frame '^CY\d{4}$') ────────────────────────
    try:
        annual = _revenue_entries(company_facts, r"^CY\d{4}$")
        if len(annual) >= 2:
            cur, prev = annual[0]["val"], annual[1]["val"]
            ratios["revenue_growth_yoy"] = (
                round((cur - prev) / prev, 4) if prev != 0 else None
            )
            if prev == 0:
                warnings.append("revenue_growth_yoy: prior year revenue is zero")
        else:
            ratios["revenue_growth_yoy"] = None
            warnings.append(
                f"revenue_growth_yoy: only {len(annual)} annual CY frame(s) found"
            )
    except Exception as exc:
        ratios["revenue_growth_yoy"] = None
        warnings.append(f"revenue_growth_yoy: {exc}")

    # ── Revenue growth QoQ (quarterly frame '^CY\d{4}Q\d$') ─────────────────
    try:
        quarterly = _revenue_entries(company_facts, r"^CY\d{4}Q\d$")
        if len(quarterly) >= 2:
            cur, prev = quarterly[0]["val"], quarterly[1]["val"]
            ratios["revenue_growth_qoq"] = (
                round((cur - prev) / prev, 4) if prev != 0 else None
            )
            if prev == 0:
                warnings.append("revenue_growth_qoq: prior quarter revenue is zero")
        else:
            ratios["revenue_growth_qoq"] = None
            warnings.append(
                f"revenue_growth_qoq: only {len(quarterly)} quarterly CY frame(s) found"
            )
    except Exception as exc:
        ratios["revenue_growth_qoq"] = None
        warnings.append(f"revenue_growth_qoq: {exc}")

    ratios["warnings"] = warnings
    return ratios


# ---------------------------------------------------------------------------
# compare_peers
# ---------------------------------------------------------------------------

async def compare_peers(ticker: str, peer_tickers: list[str]) -> dict:
    """Compare *ticker* against *peer_tickers* on the same ratio set.

    Fetches company_facts from the Data Agent for every ticker concurrently.
    Failed fetches are excluded from the table with a warning, never crash
    the whole call.  Result table is sorted by market_cap descending.
    """
    all_tickers = [ticker] + list(peer_tickers)

    async def _fetch_one(t: str) -> tuple[str, dict | None, str | None]:
        try:
            cik = await resolve_cik(t)
            facts = await get_company_facts(cik)
            return (t, facts, None)
        except Exception as exc:
            return (t, None, str(exc))

    fetched = await asyncio.gather(*[_fetch_one(t) for t in all_tickers])

    table: list[dict[str, Any]] = []
    warnings: list[str] = []

    for t, facts, err in fetched:
        if facts is None:
            warnings.append(f"Excluded {t}: {err}")
            continue

        row_ratios = await compute_ratios(facts, ticker=t)
        ratio_warns = row_ratios.pop("warnings", [])
        warnings.extend(f"{t}: {w}" for w in ratio_warns)

        # Market cap: shares × price  (price already cached from compute_ratios P/E step)
        shares = _latest(facts, "EntityCommonStockSharesOutstanding",
                         namespace="dei", unit="shares")
        price = await _get_cached_price(t)
        market_cap = round(shares * price) if (shares and price) else None

        table.append({"ticker": t, "market_cap": market_cap, **row_ratios})

    # Sort by market_cap descending; None market_caps go last
    table.sort(
        key=lambda r: (r.get("market_cap") is not None, r.get("market_cap") or 0),
        reverse=True,
    )

    return {"table": table, "warnings": warnings}
