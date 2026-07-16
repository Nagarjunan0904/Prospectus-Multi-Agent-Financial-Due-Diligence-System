"""
Fix curly-quote corruption introduced by the Edit tool in filing_sections.py.

Problems to fix:
  L91  _MDNA_PATTERN:  char class [U+2018 U+2019 U+2019]  -> [U+2018 U+2019 U+0027]
  L97  _MDNA_FALLBACK: char class [U+2018 U+2019 U+2019]  -> [U+2018 U+2019 U+0027]
  L102 _PAGE_NUM:      raw-string delimiters U+2019        -> ASCII apostrophe
"""
PATH = (
    r"D:\AI_ML\Projects\Prospectus-Multi-Agent-Financial-Due-Diligence-System"
    r"\mcp_servers\data_agent\tools\filing_sections.py"
)

with open(PATH, encoding="utf-8") as f:
    lines = f.readlines()

RQUOTE = "’"   # ' RIGHT SINGLE QUOTATION MARK
LQUOTE = "‘"   # ' LEFT SINGLE QUOTATION MARK
APOS   = "'"        # U+0027 ASCII apostrophe

changes = []

for i, line in enumerate(lines):
    lineno = i + 1
    new_line = line

    if lineno == 91:
        # _MDNA_PATTERN: replace the 3-char class [LQUOTE RQUOTE RQUOTE]
        # with                                     [LQUOTE RQUOTE APOS]
        old_cls = "[" + LQUOTE + RQUOTE + RQUOTE + "]"
        new_cls = "[" + LQUOTE + RQUOTE + APOS   + "]"
        if old_cls in line:
            new_line = line.replace(old_cls, new_cls)
            changes.append((lineno, "fixed MDNA_PATTERN char class"))

    if lineno == 97:
        # _MDNA_FALLBACK: same char class fix
        old_cls = "[" + LQUOTE + RQUOTE + RQUOTE + "]"
        new_cls = "[" + LQUOTE + RQUOTE + APOS   + "]"
        if old_cls in new_line:
            new_line = new_line.replace(old_cls, new_cls)
            changes.append((lineno, "fixed MDNA_FALLBACK char class"))

    if lineno == 102:
        # _PAGE_NUM: raw-string delimiters U+2019 -> ASCII apostrophe
        # But simpler: just replace the whole raw-string quote form
        old_raw = "r" + RQUOTE + r"\n\d{1,4}\n" + RQUOTE
        new_raw = r'r"\n\d{1,4}\n"'
        if old_raw in new_line:
            new_line = new_line.replace(old_raw, new_raw)
            changes.append((lineno, "fixed PAGE_NUM string delimiters"))

    lines[i] = new_line

if not changes:
    print("ERROR: no target strings found — check line numbers or character encoding")
    raise SystemExit(1)

for lineno, desc in changes:
    print(f"  L{lineno}: {desc}")

with open(PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nWrote {PATH}")

# Verify
import py_compile, ast
py_compile.compile(PATH, doraise=True)
print("py_compile: OK")

with open(PATH, encoding="utf-8") as f:
    src = f.read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("_MDNA_PATTERN", "_MDNA_FALLBACK"):
                val = ast.literal_eval(node.value)
                print(f"  {t.id} = {val!r}")
                # Show codepoints in the character class
                for j, c in enumerate(val):
                    if ord(c) > 127:
                        print(f"    pos {j}: U+{ord(c):04X}")
