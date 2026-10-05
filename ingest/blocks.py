"""Surya block HTML -> structured content (lines of text, or a table grid)."""
from __future__ import annotations

import html as htmllib
import re

from bs4 import BeautifulSoup

_BREAK_TAGS = re.compile(r"<br\s*/?>|</p>|</h\d>|</li>|</div>|</tr>|</caption>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_lines(h: str) -> list[str]:
    h = re.sub(r"\s*\n\s*", " ", h or "")
    h = _BREAK_TAGS.sub("\n", h)
    text = htmllib.unescape(_TAG.sub("", h))
    return [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n") if ln.strip()]


def _cell_text(cell) -> str:
    return " ".join(html_lines(str(cell))) if cell else ""


def html_table(h: str) -> dict:
    """{'rows': [[cell, ...], ...], 'header_rows': n}. colspan -> empty filler cells; rowspan -> repeated text."""
    soup = BeautifulSoup(re.sub(r"\s*\n\s*", " ", h or ""), "html.parser")
    table = soup.find("table")
    if table is None:
        return {"rows": [[ln] for ln in html_lines(h)], "header_rows": 0, "copies": []}
    grid: dict[tuple[int, int], str] = {}
    copies: set[tuple[int, int]] = set()          # cells filled from a rowspan (not read separately)
    header_flags = []
    for r, tr in enumerate(table.find_all("tr")):
        c = 0
        cells = tr.find_all(["td", "th"], recursive=False)
        header_flags.append(bool(cells) and (all(x.name == "th" for x in cells) or tr.find_parent("thead") is not None))
        for cell in cells:
            while (r, c) in grid:
                c += 1
            text = _cell_text(cell)
            rs = int(cell.get("rowspan", 1) or 1)
            cs = int(cell.get("colspan", 1) or 1)
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = text if dc == 0 else ""
                    if dr > 0 and dc == 0 and text:
                        copies.add((r + dr, c + dc))
            c += cs
    if not grid:
        return {"rows": [], "header_rows": 0, "copies": []}
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    rows = [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]
    header_rows = 0
    while header_rows < len(header_flags) and header_flags[header_rows]:
        header_rows += 1
    return {"rows": rows, "header_rows": header_rows, "copies": sorted(copies)}


def is_table(block: dict) -> bool:
    return block.get("label") == "Table" or "<table" in (block.get("html") or "").lower()
