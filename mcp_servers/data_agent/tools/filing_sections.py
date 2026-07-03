"""
SEC EDGAR filing section extractor.

# XBRL gives us the numbers; this gives us the narrative context XBRL can't —
# going-concern language, litigation disclosures, management's own risk framing.

Public API:
  get_filing_document(cik, accession_number) -> raw HTML, cached indefinitely
  extract_section(html, item_pattern, accession_number="") -> text, ≤ ~8 000 tokens
  get_full_section(cik, accession_number, item_pattern) -> untruncated text
  get_risk_factors(cik, accession) -> Item 1A convenience wrapper
  get_mdna(cik, accession)        -> Item 7  convenience wrapper

Table (auto-created on first import, idempotent):
  document_cache  – raw filing HTML keyed on (cik, accession_number).
                    NO TTL: EDGAR accepted filings are immutable once filed.
"""
import asyncio
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import Column, DateTime, String, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from shared.db import Base, SessionLocal, engine

# Reuse the single shared semaphore, header builder, and typed error from
# edgar_client so ALL outbound SEC requests share one rate-limit gate.
from mcp_servers.data_agent.tools.edgar_client import (
    EdgarAPIError,
    _headers,
    _semaphore,
    get_submissions,
)


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------


class SectionNotFoundError(Exception):
    """Raised when item_pattern matches no section in the filing HTML."""

    def __init__(self, item_pattern: str, accession_number: str) -> None:
        self.item_pattern = item_pattern
        self.accession_number = accession_number
        super().__init__(
            f"Pattern {item_pattern!r} not found in filing {accession_number!r}"
        )


# ---------------------------------------------------------------------------
# ORM model  (no fetched_at TTL check — docs are immutable)
# ---------------------------------------------------------------------------


class DocumentCache(Base):
    __tablename__ = "document_cache"

    cik = Column(String(10), primary_key=True)
    accession_number = Column(String(18), primary_key=True)  # no-dash form
    html = Column(Text, nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Section-extraction constants
# ---------------------------------------------------------------------------

# Matches "Item 1A.", "ITEM 7.", "item 1B " at the start of a line — used to
# detect the boundary where the next section begins.
_ITEM_BOUNDARY = re.compile(
    r'^\s*Item\s+\d{1,2}[A-Za-z]?[.\s]',
    re.IGNORECASE | re.MULTILINE,
)

# 8 000 tokens × ~4 chars/token (rough approximation; no tiktoken dep)
_MAX_CHARS = 32_000

_RISK_FACTORS_PATTERN = r'Item\s+1A\.?\s*Risk\s+Factors'
_MDNA_PATTERN = r'Item\s+7\.?\s*Management[’\'s]*\s+Discussion'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_accession(acc: str) -> tuple[str, str]:
    """Return ``(with_dashes, no_dashes)`` for an accession number."""
    clean = acc.replace("-", "")
    if len(clean) != 18:
        raise ValueError(f"Invalid accession number {acc!r}: expected 18 digits")
    dashed = f"{clean[:10]}-{clean[10:12]}-{clean[12:]}"
    return dashed, clean


async def _fetch_bytes(url: str) -> bytes:
    """GET *url* through the shared edgar_client semaphore and User-Agent header."""
    async with _semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url, headers=_headers())
        except httpx.TimeoutException:
            raise EdgarAPIError(408, url)
    if resp.status_code != 200:
        raise EdgarAPIError(resp.status_code, url)
    return resp.content


async def _primary_doc_from_submissions(cik: str, acc_dashed: str) -> str | None:
    """Look up the primary document filename in the cached submissions JSON."""
    try:
        subs = await get_submissions(cik)
    except EdgarAPIError:
        return None

    recent = subs.get("filings", {}).get("recent", {})
    acc_list = recent.get("accessionNumber", [])
    doc_list = recent.get("primaryDocument", [])

    try:
        return doc_list[acc_list.index(acc_dashed)]
    except (ValueError, IndexError):
        return None


async def _primary_doc_from_index(cik_int: str, acc_nodash: str) -> str:
    """Fetch the filing index JSON from EDGAR Archives when submissions omits it."""
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{acc_nodash}-index.json"
    )
    async with _semaphore:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=_headers())
        except httpx.TimeoutException:
            raise EdgarAPIError(408, url)
    if resp.status_code != 200:
        raise EdgarAPIError(resp.status_code, url)

    items = resp.json().get("directory", {}).get("item", [])

    # Prefer items whose type matches a form (not "GRAPHIC", "EX-*", etc.)
    for item in items:
        t = item.get("type", "")
        name = item.get("name", "")
        if name.lower().endswith((".htm", ".html")) and t not in ("", "GRAPHIC"):
            return name

    # Fallback: first .htm/.html in the directory
    for item in items:
        name = item.get("name", "")
        if name.lower().endswith((".htm", ".html")):
            return name

    raise ValueError(
        f"No primary HTML document found in EDGAR index for accession {acc_nodash!r}"
    )


def _do_extract(
    html: str, item_pattern: str, accession_number: str, *, truncate: bool
) -> str:
    """Core extraction: strip noise, locate section, optionally truncate."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()

    raw = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    text = "\n".join(lines)

    pat = re.compile(item_pattern, re.IGNORECASE)

    # Walk matches; skip TOC entries (too short to be the real section body).
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            raise SectionNotFoundError(item_pattern, accession_number)
        nxt = _ITEM_BOUNDARY.search(text, m.end() + 1)
        end = nxt.start() if nxt else len(text)
        if end - m.start() > 200:   # real section, not a one-liner TOC link
            break
        pos = m.end()

    section = text[m.start():end].strip()

    if truncate and len(section) > _MAX_CHARS:
        section = (
            section[:_MAX_CHARS]
            + "\n\n[... truncated at ~8 000 tokens — "
            "call get_full_section() for the complete text ...]"
        )

    return section


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_filing_document(cik: str, accession_number: str) -> str:
    """Return the raw HTML for the primary document of the given filing.

    Cached indefinitely in ``document_cache``; filed documents are immutable
    once accepted by EDGAR so no TTL is needed.
    Raises :exc:`EdgarAPIError` on network failure.
    """
    acc_dashed, acc_nodash = _normalize_accession(accession_number)

    def _cached() -> str | None:
        with SessionLocal() as session:
            row = session.execute(
                select(DocumentCache).where(
                    DocumentCache.cik == cik,
                    DocumentCache.accession_number == acc_nodash,
                )
            ).scalar_one_or_none()
            return row.html if row else None

    cached = await asyncio.to_thread(_cached)
    if cached is not None:
        return cached

    cik_int = str(int(cik))  # strip leading zeros for Archive URL

    primary_doc = await _primary_doc_from_submissions(cik, acc_dashed)
    if primary_doc is None:
        primary_doc = await _primary_doc_from_index(cik_int, acc_nodash)

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{primary_doc}"
    )
    raw_bytes = await _fetch_bytes(url)

    try:
        html = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = raw_bytes.decode("latin-1")

    now = datetime.now(timezone.utc)

    def _store() -> None:
        with SessionLocal() as session:
            stmt = (
                pg_insert(DocumentCache)
                .values(
                    cik=cik,
                    accession_number=acc_nodash,
                    html=html,
                    fetched_at=now,
                )
                .on_conflict_do_nothing(index_elements=["cik", "accession_number"])
            )
            session.execute(stmt)
            session.commit()

    await asyncio.to_thread(_store)
    return html


def extract_section(
    html: str, item_pattern: str, accession_number: str = ""
) -> str:
    """Locate *item_pattern* in *html* and return its text, capped at ~8 000 tokens.

    Strips ``<table>``, ``<script>``, and ``<style>`` tags before extracting.
    Collapses blank lines and leading/trailing whitespace.
    Raises :exc:`SectionNotFoundError` if the pattern has no match.
    """
    return _do_extract(html, item_pattern, accession_number, truncate=True)


async def get_full_section(
    cik: str, accession_number: str, item_pattern: str
) -> str:
    """Return the complete, untruncated section text for the Risk Agent's audit scan.

    Fetches and caches the document via :func:`get_filing_document`, then
    extracts without the 8 000-token cap applied by :func:`extract_section`.
    Raises :exc:`SectionNotFoundError` if *item_pattern* has no match.
    """
    html = await get_filing_document(cik, accession_number)
    return _do_extract(html, item_pattern, accession_number, truncate=False)


async def get_risk_factors(cik: str, accession: str) -> str:
    """Return Item 1A (Risk Factors), capped at ~8 000 tokens."""
    html = await get_filing_document(cik, accession)
    return extract_section(html, _RISK_FACTORS_PATTERN, accession)


async def get_mdna(cik: str, accession: str) -> str:
    """Return Item 7 (MD&A), capped at ~8 000 tokens."""
    html = await get_filing_document(cik, accession)
    return extract_section(html, _MDNA_PATTERN, accession)
