from __future__ import annotations

import re

from selectolax.parser import HTMLParser

# Canonical names for the most-cited 10-K / 10-Q items. Items not in the map
# keep a generic "Item N" label.
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

# "Item 1A." / "Item 7:" / "Item 2 -" etc. (dash/en-dash/em-dash tolerated)
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-Z]?)\s*[.\:\-–—]", re.IGNORECASE)


def looks_like_html(raw: str) -> bool:
    head = raw[:2000].lower()
    return any(t in head for t in ("<html", "<!doctype html", "<div", "<p ", "<table"))


def html_to_text(raw: str) -> str:
    tree = HTMLParser(raw)
    for tag in tree.css("script, style"):
        tag.decompose()
    node = tree.body or tree.root
    text = node.text(separator="\n") if node else ""
    # Normalise whitespace: collapse intra-line runs (the char class includes a
    # non-breaking space, common in EDGAR HTML) and excess blank lines.
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def section_filing(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_label, content) by Item headers.

    Heuristic for the table-of-contents problem: an Item header usually appears
    twice — once in the ToC, once at the real section. We take each item's LAST
    occurrence as the section start, then slice between consecutive starts.
    """
    last: dict[str, int] = {}
    for m in _ITEM_RE.finditer(text):
        last[m.group(1).upper()] = m.start()
    if not last:
        return [("Full Document", text)]

    ordered = sorted(last.items(), key=lambda kv: kv[1])
    sections: list[tuple[str, str]] = []

    # Keep the cover page / preamble before the first item if it's substantial.
    first_start = ordered[0][1]
    if first_start > 400:
        sections.append(("Cover / Preamble", text[:first_start].strip()))

    for idx, (item, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(text)
        body = text[start:end].strip()
        if len(body) < 200:  # stray header with no real content
            continue
        sections.append((ITEM_NAMES.get(item, f"Item {item}"), body))

    return sections or [("Full Document", text)]
