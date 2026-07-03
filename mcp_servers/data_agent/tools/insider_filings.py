"""
SEC EDGAR Form 4 insider-transaction fetcher.

Public API:
  get_insider_transactions(cik, days=90) -> dict

  Returns:
    {
      "transactions": [list of dicts, newest-first],
      "summary": {
        "net_shares_bought": float,   # sum of shares for code 'P'
        "net_shares_sold":   float,   # sum of shares for code 'S'
        "unique_sellers":    int,     # distinct filer names with ≥1 'S'
      }
    }

  Summary fields are ALWAYS present and zeroed when there is no data —
  never null or missing — because the Risk Agent reads them unconditionally.

Table (auto-created on first import, idempotent):
  form4_cache — raw Form 4 XML keyed on (cik, accession_number).
                NO TTL: accepted filings are immutable once filed.

All outbound HTTP requests share the same module-level semaphore,
User-Agent header, and httpx.AsyncClient pattern defined in edgar_client.py.
"""
import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.db import Base, SessionLocal, engine
from mcp_servers.data_agent.tools.edgar_client import (
    EdgarAPIError,
    _headers,
    _semaphore,
    get_submissions,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------


class Form4Cache(Base):
    __tablename__ = "form4_cache"

    cik = Column(String(10), primary_key=True)
    accession_number = Column(String(18), primary_key=True)  # no-dash form
    xml = Column(Text, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm_acc(acc: str) -> tuple[str, str]:
    """Return ``(with_dashes, no_dashes)`` for an accession number."""
    clean = acc.replace("-", "")
    if len(clean) != 18:
        raise ValueError(f"Invalid accession number {acc!r}: expected 18 digits")
    return f"{clean[:10]}-{clean[10:12]}-{clean[12:]}", clean


async def _fetch_bytes(url: str) -> bytes:
    """GET *url* through the shared edgar_client semaphore and User-Agent header."""
    async with _semaphore:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=_headers())
        except httpx.TimeoutException:
            raise EdgarAPIError(408, url)
    if resp.status_code != 200:
        raise EdgarAPIError(resp.status_code, url)
    return resp.content


async def _load_form4_cache(cik: str, acc_nodash: str) -> str | None:
    def _q() -> str | None:
        with SessionLocal() as session:
            row = session.execute(
                select(Form4Cache).where(
                    Form4Cache.cik == cik,
                    Form4Cache.accession_number == acc_nodash,
                )
            ).scalar_one_or_none()
            return row.xml if row else None

    return await asyncio.to_thread(_q)


async def _store_form4_cache(cik: str, acc_nodash: str, xml: str) -> None:
    now = datetime.now(timezone.utc)

    def _u() -> None:
        with SessionLocal() as session:
            stmt = (
                pg_insert(Form4Cache)
                .values(
                    cik=cik,
                    accession_number=acc_nodash,
                    xml=xml,
                    fetched_at=now,
                )
                .on_conflict_do_nothing(index_elements=["cik", "accession_number"])
            )
            session.execute(stmt)
            session.commit()

    await asyncio.to_thread(_u)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _elem_text(parent: ET.Element, *path: str) -> str | None:
    """Walk *path* from *parent*; return stripped text of the final element."""
    node: ET.Element | None = parent
    for step in path:
        if node is None:
            return None
        node = node.find(step)
    if node is None:
        return None
    return (node.text or "").strip() or None


def _parse_role(rel: ET.Element | None) -> str:
    """Map reportingOwnerRelationship flags to a human-readable role string."""
    if rel is None:
        return "other"
    parts: list[str] = []
    if (rel.findtext("isDirector") or "").strip() == "1":
        parts.append("director")
    if (rel.findtext("isOfficer") or "").strip() == "1":
        parts.append("officer")
    if (rel.findtext("isTenPercentOwner") or "").strip() == "1":
        parts.append("10% owner")
    return "+".join(parts) if parts else "other"


def _parse_form4_xml(
    xml_str: str, acc_dashed: str, filing_date: str
) -> list[dict[str, Any]]:
    """Parse a Form 4 XML string into a list of transaction dicts.

    Handles both nonDerivativeTable and derivativeTable.
    Does NOT raise on missing optional fields — leaves them None.
    Raises ``xml.etree.ElementTree.ParseError`` on malformed XML (caller logs).
    """
    root = ET.fromstring(xml_str)
    period = (root.findtext("periodOfReport") or "").strip()

    # Collect all reporting owners.  Most filings have exactly one; joint
    # filings (e.g. a spousal trust) can have several — combine them.
    owners: list[tuple[str, str]] = []
    for owner_elem in root.findall("reportingOwner"):
        name = _elem_text(owner_elem, "reportingOwnerId", "rptOwnerName") or "Unknown"
        role = _parse_role(owner_elem.find("reportingOwnerRelationship"))
        owners.append((name, role))

    if not owners:
        return []

    if len(owners) == 1:
        filer_name, role = owners[0]
    else:
        filer_name = "; ".join(o[0] for o in owners)
        # Deduplicate role parts while preserving order
        seen: dict[str, None] = {}
        for _, r in owners:
            for part in r.split("+"):
                seen[part] = None
        role = "+".join(seen) or "other"

    transactions: list[dict[str, Any]] = []

    def _harvest(table_tag: str, txn_tag: str, role_type: str) -> None:
        table = root.find(table_tag)
        if table is None:
            return

        for txn in table.findall(txn_tag):
            # transactionCode lives in <transactionCoding>, a sibling of
            # <transactionAmounts> — NOT inside transactionAmounts.
            coding = txn.find("transactionCoding")
            code = (
                (coding.findtext("transactionCode") or "").strip()
                if coding is not None else ""
            )
            if not code:
                continue

            amounts = txn.find("transactionAmounts")
            if amounts is None:
                continue

            # shares: wrapped in <value> child
            shares: float | None = None
            shares_str = _elem_text(amounts, "transactionShares", "value")
            if shares_str is not None:
                try:
                    shares = float(shares_str)
                except ValueError:
                    pass

            # price: may be absent for non-cash grants (option exercises,
            # RSU vests, etc.) — leave None, never coerce missing to 0
            price: float | None = None
            price_str = _elem_text(amounts, "transactionPricePerShare", "value")
            if price_str is not None:
                try:
                    price = float(price_str)
                except ValueError:
                    pass

            txn_date = _elem_text(txn, "transactionDate", "value") or filing_date

            transactions.append(
                {
                    "filer_name": filer_name,
                    "role": role,
                    "role_type": role_type,
                    "transaction_code": code,
                    "shares": shares,
                    "price_per_share": price,
                    "transaction_date": txn_date,
                    "filing_date": filing_date,
                    "accession_number": acc_dashed,
                    "period_of_report": period,
                }
            )

    _harvest("nonDerivativeTable", "nonDerivativeTransaction", "non-derivative")
    _harvest("derivativeTable", "derivativeTransaction", "derivative")
    return transactions


# ---------------------------------------------------------------------------
# Per-filing fetch + parse coroutine
# ---------------------------------------------------------------------------


async def _fetch_and_parse(
    cik: str,
    acc_dashed: str,
    acc_nodash: str,
    primary_doc: str,
    filing_date: str,
) -> list[dict[str, Any]]:
    """Fetch (or load from cache) the Form 4 XML and return parsed transactions."""
    xml_str = await _load_form4_cache(cik, acc_nodash)

    if xml_str is None:
        cik_int = str(int(cik))  # strip leading zeros for Archive URL
        # submissions.json "primaryDocument" for ownership forms often has an
        # "xslF345X0x/" prefix (EDGAR's XSLT render path).  That path returns
        # browser-rendered HTML even when the filename ends in .xml.  The raw
        # XML lives at the same filename WITHOUT the prefix, directly in the
        # accession folder.
        filename = primary_doc.rsplit("/", 1)[-1]
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
            f"/{acc_nodash}/{filename}"
        )
        raw = await _fetch_bytes(url)
        xml_str = raw.decode("utf-8", errors="replace")
        await _store_form4_cache(cik, acc_nodash, xml_str)

    return _parse_form4_xml(xml_str, acc_dashed, filing_date)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ZERO_SUMMARY: dict[str, Any] = {
    "net_shares_bought": 0.0,
    "net_shares_sold": 0.0,
    "unique_sellers": 0,
}


async def get_insider_transactions(cik: str, days: int = 90) -> dict[str, Any]:
    """Return Form 4 insider transactions filed within the last *days* days.

    Uses get_submissions (already 24 h-cached in filing_cache) to enumerate
    filings — no extra SEC call for the index.  Each Form 4 XML is cached
    indefinitely in form4_cache.

    Returns a dict with keys:
      "transactions"  – list of transaction dicts, newest filing first.
      "summary"       – always-present dict with net_shares_bought,
                        net_shares_sold, and unique_sellers (all zeroed
                        when no Form 4s fall in the window).

    Individual filing failures are warned and skipped; they do not abort
    the entire batch.
    """
    subs = await get_submissions(cik)
    recent = subs.get("filings", {}).get("recent", {})

    forms: list[str] = recent.get("form", [])
    filing_dates: list[str] = recent.get("filingDate", [])
    acc_numbers: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])

    cutoff = date.today() - timedelta(days=days)

    # (filing_date_str, acc_dashed, acc_nodash, primary_doc)
    form4_batch: list[tuple[str, str, str, str]] = []

    for i, form in enumerate(forms):
        if form != "4":
            continue

        filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
        if not filing_date_str:
            continue
        try:
            if date.fromisoformat(filing_date_str) < cutoff:
                continue
        except ValueError:
            continue

        acc = acc_numbers[i] if i < len(acc_numbers) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        if not acc or not primary_doc:
            continue

        try:
            _, acc_nodash = _norm_acc(acc)
        except ValueError:
            continue

        form4_batch.append((filing_date_str, acc, acc_nodash, primary_doc))

    if not form4_batch:
        return {"transactions": [], "summary": dict(_ZERO_SUMMARY)}

    tasks = [
        _fetch_and_parse(cik, acc_dashed, acc_nodash, primary_doc, filing_date)
        for filing_date, acc_dashed, acc_nodash, primary_doc in form4_batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_transactions: list[dict[str, Any]] = []
    for (filing_date, acc_dashed, acc_nodash, _), result in zip(form4_batch, results):
        if isinstance(result, BaseException):
            _log.warning(
                "Skipping Form 4 %s (cik=%s, filed=%s): %s: %s",
                acc_dashed,
                cik,
                filing_date,
                type(result).__name__,
                result,
            )
        else:
            all_transactions.extend(result)

    # Newest filing first; within same filing date, newest transaction date first
    all_transactions.sort(
        key=lambda t: (t["filing_date"], t["transaction_date"]),
        reverse=True,
    )

    net_bought = sum(
        t["shares"]
        for t in all_transactions
        if t["transaction_code"] == "P" and t["shares"] is not None
    )
    net_sold = sum(
        t["shares"]
        for t in all_transactions
        if t["transaction_code"] == "S" and t["shares"] is not None
    )
    unique_sellers = len(
        {t["filer_name"] for t in all_transactions if t["transaction_code"] == "S"}
    )

    return {
        "transactions": all_transactions,
        "summary": {
            "net_shares_bought": net_bought,
            "net_shares_sold": net_sold,
            "unique_sellers": unique_sellers,
        },
    }
