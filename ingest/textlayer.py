"""PDF text layer: words with boxes and font information, and page classification.

The text layer is exact for digits, codes, English and table cells, but NOT for Bangla in the Nikosh
font (wrong glyph-to-Unicode table). Legacy ANSI Bangla (SutonnyMJ / Bijoy) is stored as ASCII
("kZ©vewj" = "শর্তাবলি", "1" = "১"); those words are flagged `legacy` so the merge never copies them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from core.config import settings
from core.textnorm import clean

_LEGACY_RE = re.compile(settings.legacy_font_pattern, re.I)


@dataclass
class Word:
    text: str
    bbox: tuple[float, float, float, float]   # PDF points
    font: str
    legacy: bool
    line: int                                   # running line number in reading order


def words(page: pymupdf.Page) -> list[Word]:
    """Whitespace-separated words built from characters, so each word knows its font."""
    out: list[Word] = []
    line_no = 0
    raw = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            line_no += 1
            cur: list[tuple[str, tuple, str]] = []

            def flush():
                if not cur:
                    return
                text = clean("".join(c for c, _, _ in cur))
                if text.strip():
                    xs0, ys0, xs1, ys1 = zip(*(b for _, b, _ in cur))
                    fonts = [f for _, _, f in cur]
                    font = max(set(fonts), key=fonts.count)
                    out.append(Word(text, (min(xs0), min(ys0), max(xs1), max(ys1)), font,
                                    any(_LEGACY_RE.search(f) for f in fonts), line_no))
                cur.clear()

            for span in line["spans"]:
                for ch in span["chars"]:
                    c = ch["c"]
                    if c.isspace():
                        flush()
                    else:
                        cur.append((c, tuple(ch["bbox"]), span["font"]))
            flush()
    return out


def classify(page: pymupdf.Page, ws: list[Word] | None = None) -> str:
    """'digital', 'scanned' or 'blank'."""
    ws = words(page) if ws is None else ws
    area = page.rect.width * page.rect.height
    n_chars = sum(len(w.text) for w in ws)
    text_area = sum((w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]) for w in ws) / area
    img_area = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        img_area += r.width * r.height
    img_area /= area
    if n_chars < settings.scanned_min_chars:
        if img_area > 0.05 or has_ink(page):
            return "scanned"
        return "blank"
    if text_area < settings.scanned_text_area and img_area > settings.scanned_image_area:
        return "scanned"
    return "digital"


def has_ink(page: pymupdf.Page, dpi: int = 36, thresh: float = 0.002) -> bool:
    """True if the rendered page has more than a trace of non-white pixels."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    data = pix.samples
    dark = sum(1 for b in data if b < 200)
    return dark / max(1, len(data)) > thresh


def render(page: pymupdf.Page, dpi: int):
    """PIL image of the page."""
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
