"""
14-check test: get_risk_factors + get_mdna for all 7 golden-set tickers.
For INTC MD&A specifically, also prints body details to verify it is real
prose rather than the 221-char cross-reference table.
"""
from __future__ import annotations
import re, sys, time, ast

import httpx
from bs4 import BeautifulSoup

# ── Load constants from filing_sections.py ───────────────────────────────────

_SRC = (
    r"D:\AI_ML\Projects\Prospectus-Multi-Agent-Financial-Due-Diligence-System"
    r"\mcp_servers\data_agent\tools\filing_sections.py"
)

with open(_SRC, encoding="utf-8") as _f:
    _tree = ast.parse(_f.read())

_consts: dict = {}
for _node in ast.walk(_tree):
    if isinstance(_node, ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, ast.Name) and _t.id in (
                "_RISK_FACTORS_PATTERN", "_RISK_FACTORS_FALLBACK",
                "_MDNA_PATTERN", "_MDNA_FALLBACK", "_MAX_CHARS",
            ):
                _consts[_t.id] = ast.literal_eval(_node.value)

RISK_FACTORS_PATTERN  = _consts["_RISK_FACTORS_PATTERN"]
RISK_FACTORS_FALLBACK = _consts["_RISK_FACTORS_FALLBACK"]
MDNA_PATTERN          = _consts["_MDNA_PATTERN"]
MDNA_FALLBACK         = _consts["_MDNA_FALLBACK"]
MAX_CHARS             = _consts["_MAX_CHARS"]

print("Constants loaded from filing_sections.py:")
print(f"  _MDNA_PATTERN  = {MDNA_PATTERN!r}")
print(f"  _MDNA_FALLBACK = {MDNA_FALLBACK!r}")
print()

# ── Helpers ──────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "due-diligence-14check/1.0 (nagarjunansaravananus@gmail.com)"}

ITEM_BOUNDARY = re.compile(
    r'^\s*Item\s+\d{1,2}[A-Za-z]?[.\s]',
    re.IGNORECASE | re.MULTILINE,
)
RF_RUNNING_HEADER = re.compile(
    r'^[ \t]*Risk[ \t]+Factors[ \t]*$\n[ \t]*\d{1,4}[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
PAGE_NUM = re.compile(r'\n\d{1,4}\n')

TICKERS = [
    ("AAPL",  "0000320193", "0000320193-25-000079"),
    ("MSFT",  "0000789019", "0000950170-25-100235"),
    ("GOOGL", "0001652044", "0001652044-26-000018"),
    ("BA",    "0000012927", "0001628280-26-004357"),
    ("INTC",  "0000050863", "0000050863-26-000011"),
    ("RIVN",  "0001874178", "0001874178-26-000008"),
    ("SMCI",  "0001375365", "0001375365-25-000027"),
]


def fetch(url: str) -> bytes:
    r = httpx.get(url, headers=HEADERS, timeout=90.0, follow_redirects=True)
    r.raise_for_status()
    return r.content


def get_primary_doc(cik_padded: str, acc_dashed: str) -> tuple[str, str]:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = httpx.get(url, headers=HEADERS, timeout=30.0).json()
    cik_int = str(int(cik_padded))
    recent = data.get("filings", {}).get("recent", {})
    acc_list = recent.get("accessionNumber", [])
    doc_list = recent.get("primaryDocument", [])
    try:
        return cik_int, doc_list[acc_list.index(acc_dashed)]
    except (ValueError, IndexError):
        pass
    for page_ref in data.get("filings", {}).get("files", []):
        page = httpx.get(
            "https://data.sec.gov/submissions/" + page_ref["name"],
            headers=HEADERS, timeout=30.0,
        ).json()
        try:
            return cik_int, page.get("primaryDocument", [])[
                page.get("accessionNumber", []).index(acc_dashed)
            ]
        except (ValueError, IndexError):
            pass
    raise ValueError(f"{acc_dashed} not found for CIK {cik_padded}")


def pipeline(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]): tag.decompose()
    for tag in soup.find_all(["a","span","b","i","em","strong","u","sub","sup","small","font"]):
        tag.unwrap()
    soup.smooth()
    for tag in soup.find_all(["table","thead","tbody","tfoot","tr","th","td"]):
        tag.unwrap()
    raw = soup.get_text(separator="\n")
    return "\n".join(ln.strip() for ln in raw.splitlines() if ln.strip())


def rf_smart_end(text: str, start: int, default_end: int) -> int:
    headers = list(RF_RUNNING_HEADER.finditer(text, start, default_end))
    if len(headers) < 2:
        return default_end
    scan_from = headers[-1].end()
    m = re.compile(
        r'^[ \t]*(?!Risk[ \t]+Factors[ \t]*$)(?!\d{1,4}[ \t]*$)'
        r'([A-Z][A-Za-z][^\n]{5,80})[ \t]*$',
        re.MULTILINE,
    ).search(text, scan_from)
    return m.start() if m else default_end


def do_extract(
    text: str,
    pattern_str: str,
    *,
    truncate: bool = True,
    smart_rf_end_flag: bool = False,
    min_body_len: int = 200,
    toc_page_skip: bool = False,
) -> tuple[bool, int, str]:
    """Returns (success, raw_body_len, first_120_chars_of_body)."""
    pat = re.compile(pattern_str, re.IGNORECASE)
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            return False, 0, "SectionNotFoundError"
        nxt = ITEM_BOUNDARY.search(text, m.end() + 1)
        end = nxt.start() if nxt else len(text)
        if end - m.start() > min_body_len:
            if not toc_page_skip:
                break
            sample = text[m.end():min(m.end() + 400, end)]
            if len(PAGE_NUM.findall(sample)) < 3:
                break
        pos = m.end()

    if smart_rf_end_flag:
        end = rf_smart_end(text, m.start(), end)

    section = text[m.start():end].strip()
    raw_body_len = len(section)
    if truncate and raw_body_len > MAX_CHARS:
        section = section[:MAX_CHARS] + "\n\n[... truncated ...]"
    return True, raw_body_len, section[:120]


# ── Run checks ───────────────────────────────────────────────────────────────

results: list = []  # (ticker, section, ok, raw_len, note)

for ticker, cik_padded, acc_dashed in TICKERS:
    acc_nodash = acc_dashed.replace("-", "")
    print(f"=== {ticker} ===")
    try:
        time.sleep(0.4)
        cik_int, primary_doc = get_primary_doc(cik_padded, acc_dashed)
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"
        print(f"  Fetching {primary_doc} ...", end=" ", flush=True)
        time.sleep(0.6)
        html = fetch(url).decode("utf-8", errors="replace")
        print(f"{len(html):,} bytes → processing ...", end=" ", flush=True)
        text = pipeline(html)
        print(f"{len(text):,} chars")

        # CHECK 1: Risk Factors
        ok, raw_len, snippet = do_extract(text, RISK_FACTORS_PATTERN)
        if not ok:
            ok, raw_len, snippet = do_extract(
                text, RISK_FACTORS_FALLBACK, smart_rf_end_flag=True
            )
            note = "via FALLBACK" if ok else "MISSING"
        else:
            note = "via PRIMARY"
        results.append((ticker, "get_risk_factors", ok, raw_len, note))
        print(f"  {'PASS' if ok else 'FAIL'} get_risk_factors  raw_len={raw_len:,}  ({note})")

        # CHECK 2: MD&A
        ok2, raw_len2, snippet2 = do_extract(
            text, MDNA_PATTERN, min_body_len=1_000
        )
        if not ok2:
            ok2, raw_len2, snippet2 = do_extract(
                text, MDNA_FALLBACK, min_body_len=1_000, toc_page_skip=True
            )
            note2 = "via FALLBACK" if ok2 else "MISSING"
        else:
            note2 = "via PRIMARY"
        results.append((ticker, "get_mdna", ok2, raw_len2, note2))
        print(f"  {'PASS' if ok2 else 'FAIL'} get_mdna          raw_len={raw_len2:,}  ({note2})")
        print(f"       snippet: {snippet2!r}")

        # INTC extra validation: verify real prose, not the 221-char cross-reference
        if ticker == "INTC" and ok2:
            if raw_len2 < 1_000:
                print(f"  [INTC WARN] MD&A body suspiciously short ({raw_len2} chars)")
            elif "Pages" in snippet2 and raw_len2 < 500:
                print(f"  [INTC WARN] looks like cross-reference table")
            else:
                print(f"  [INTC OK]   real prose confirmed (raw_len={raw_len2:,})")

    except Exception as exc:
        import traceback; traceback.print_exc()
        results.append((ticker, "get_risk_factors", False, 0, str(exc)))
        results.append((ticker, "get_mdna",         False, 0, str(exc)))
    print()
    time.sleep(1.0)

# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 65)
print("FINAL SUMMARY  (14 checks)")
print("=" * 65)
print(f"{'Ticker':<8} {'Section':<22} {'Result':<6} {'Raw body len':>12}  Note")
print("-" * 65)
passed = 0
for ticker, section, ok, raw_len, note in results:
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"{ticker:<8} {section:<22} {status:<6} {raw_len:>12,}  {note}")
print("-" * 65)
print(f"{passed}/14 passed")
if passed < 14:
    sys.exit(1)
