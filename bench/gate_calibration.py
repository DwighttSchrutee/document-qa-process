"""Gate calibration: the bench pages must PASS (or WARN only for row-level flags); the known-bad
page (tariff p101 OCR'd at 150 dpi: 37 wrong codes/rates) must be HOLD.

    python -m bench.gate_calibration
"""
import hashlib
import json

import pymupdf

from core.config import settings
from ingest import extract, gate, textlayer
from ingest.surya_ocr import cache_path

PAGES = [("Tariff-2025-2026(02-06-2025).pdf", p, 192) for p in (1, 61, 101)] + \
        [("Tariff-2025-2026(02-06-2025).pdf", 101, 150)] + \
        [("nbr-01.pdf", p, 192) for p in (2, 151, 356)] + [("nbr-02.pdf", p, 192) for p in (1, 2, 6, 8, 101)] + \
        [("nbr_06.pdf", 2, 192), ("nbr_07.pdf", 1, 192)]
EXPECT_HOLD = {("Tariff-2025-2026(02-06-2025).pdf", 101, 150)}


def run():
    rows = []
    for f, p, dpi in PAGES:
        path = settings.docs_dir / f
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cp = cache_path(sha, p, dpi)
        if not cp.exists():
            rows.append((f, p, dpi, "NO CACHE", "", None))
            continue
        page = pymupdf.open(path)[p - 1]
        ws = textlayer.words(page)
        rec = extract._pdf_page(p, page, ws, textlayer.classify(page, ws), json.loads(cp.read_text("utf-8")))
        g = gate.evaluate(rec)
        reasons = "; ".join(f"{r['status']} {r['check']}: {r['detail']}" for r in g["reasons"])
        holds = sum(1 for x in g["items"] if x["status"] == "HOLD")
        warns = sum(1 for x in g["items"] if x["status"] == "WARN")
        ok = (g["status"] == "HOLD") == ((f, p, dpi) in EXPECT_HOLD)
        rows.append((f, p, dpi, g["status"], f"{reasons} | items HOLD {holds} WARN {warns}", ok))
    return rows


if __name__ == "__main__":
    bad = 0
    for f, p, dpi, st, why, ok in run():
        bad += ok is False
        print(f"{'ok ' if ok else 'BAD'} {f[:22]:22} p{p:<4} {dpi}dpi  {st:6} {why}")
    print("calibration", "PASSED" if not bad else f"FAILED ({bad})")
