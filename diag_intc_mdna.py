"""Find INTC's actual MD&A heading — the one without Item 7 prefix."""
import re, time
import httpx
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "due-diligence-diag/1.0 (nagarjunansaravananus@gmail.com)"}

def fetch(url):
    r = httpx.get(url, headers=HEADERS, timeout=90.0, follow_redirects=True)
    r.raise_for_status()
    return r.content


def pipeline(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]): tag.decompose()
    for tag in soup.find_all(["a","span","b","i","em","strong","u","sub","sup","small","font"]):
        tag.unwrap()
    soup.smooth()
    for tag in soup.find_all(["table","thead","tbody","tfoot","tr","th","td"]):
        tag.unwrap()
    raw = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return "\n".join(lines)


url = ("https://www.sec.gov/Archives/edgar/data/50863"
       "/000005086326000011/intc-20251227.htm")
print(f"Fetching INTC ... ", end="", flush=True)
html = fetch(url).decode("utf-8", errors="replace")
text = pipeline(html)
print(f"{len(text):,} chars")

BOUNDARY = re.compile(r'^\s*Item\s+\d{1,2}[A-Za-z]?[.\s]', re.IGNORECASE | re.MULTILINE)

# 1. All hits for "management" + "discussion" anywhere (case-insensitive, broad)
print("\n=== All 'management...discussion' hits (broad, case-insensitive) ===")
broad = re.compile(r'management.{0,80}discussion', re.IGNORECASE | re.DOTALL)
for i, m in enumerate(broad.finditer(text)):
    ctx_start = max(0, m.start() - 30)
    ctx_end   = min(len(text), m.end() + 120)
    ctx = text[ctx_start:ctx_end]
    nxt = BOUNDARY.search(text, m.end() + 1)
    end = nxt.start() if nxt else len(text)
    body_len = end - m.start()
    print(f"  [{i}] offset={m.start():,}  body_len_to_next_item={body_len:,}")
    print(f"      {ctx!r}")
    if i >= 9:
        print("  ... (truncated)")
        break

# 2. Look for standalone line versions of "Management's Discussion and Analysis"
print("\n=== Standalone heading: '^Management.*Discussion.*Analysis' ===")
standalone = re.compile(
    r'^Management(?:[^\S\n]|\n)*(?:[‘’\']s\s+)?Discussion\s+and\s+Analysis',
    re.IGNORECASE | re.MULTILINE,
)
for i, m in enumerate(standalone.finditer(text)):
    ctx_start = max(0, m.start() - 10)
    ctx_end   = min(len(text), m.end() + 200)
    nxt = BOUNDARY.search(text, m.end() + 1)
    end = nxt.start() if nxt else len(text)
    body_len = end - m.start()
    print(f"  [{i}] offset={m.start():,}  body_len={body_len:,}")
    print(f"      {text[ctx_start:ctx_end]!r}")
    if i >= 5:
        print("  ... (truncated)")
        break

# 3. What comes right before offset 568167 (the Item 7 cross-ref)?
print("\n=== 500 chars immediately before the Item 7 cross-reference (offset 568167) ===")
print(repr(text[567700:568200]))
