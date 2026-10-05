"""OCR bake-off: Tesseract variants vs Surya on the same rendered pages.

Everything runs locally. The PDF text layer is used only as an ANSWER KEY for the
things it is known to get right (HS codes, digits, rate rows); Bangla quality is
scored against hand-verified reference pages in refs/.

Usage:
    python ocr_bench.py --pdf-dir pdfs                 # Tesseract variants (+ Surya if installed)
    python ocr_bench.py --pdf-dir pdfs --engines surya # one engine only
    python ocr_bench.py --list                         # show test pages

Outputs: results.csv (one row per engine x page) and out/<engine>/<page>.txt
for side-by-side reading.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent

# (file, 1-based page, kind, tesseract language)
#   tariff_sheet : CD SD VAT AIT AT RD TTI rows  -> row-exact score
#   table        : HS-code tables                -> HS + number recall
#   prose_bn     : Bangla prose                  -> number recall (+ CER if refs/<file>_p<n>.txt exists)
TEST_PAGES = [
    ("Tariff-2025-202602-06-2025.pdf", 1, "tariff_sheet", "eng"),
    ("Tariff-2025-202602-06-2025.pdf", 61, "tariff_sheet", "eng"),
    ("Tariff-2025-202602-06-2025.pdf", 101, "tariff_sheet", "eng"),
    ("nbr-01.pdf", 2, "table", "eng"),
    ("nbr-01.pdf", 151, "table", "eng"),
    ("nbr-01.pdf", 356, "table", "eng"),
    ("nbr-02.pdf", 6, "table", "ben+eng"),
    ("nbr-02.pdf", 8, "table", "ben+eng"),
    ("nbr_06.pdf", 2, "table", "ben+eng"),
    ("nbr-02.pdf", 1, "prose_bn", "ben+eng"),
    ("nbr-02.pdf", 2, "prose_bn", "ben+eng"),
    ("nbr-02.pdf", 101, "prose_bn", "ben+eng"),
    ("nbr_07.pdf", 1, "prose_bn", "ben+eng"),
]

TESSERACT_VARIANTS = [  # (name, dpi, psm, preprocess)
    ("tess_300_psm4", 300, "4", None),
    ("tess_300_psm6", 300, "6", None),
    ("tess_300_psm11", 300, "11", None),
    ("tess_400_psm4", 400, "4", None),
    ("tess_300_psm4_bin", 300, "4", "otsu"),
]
SURYA_DPI = 150  # Surya works at low resolution internally; 96-150 is typical

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
HS_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{2}\.\d{2}|\d{8})(?![\d.])")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


# ---------------------------------------------------------------- normalisation

def norm(text: str) -> str:
    """NFC + remove zero-width joiners + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    return re.sub(r"\s+", " ", text).strip()


def numbers(text: str) -> collections.Counter:
    return collections.Counter(NUM_RE.findall(text.translate(BN_DIGITS)))


def hs_codes(text: str) -> collections.Counter:
    return collections.Counter(c.replace(".", "") for c in HS_RE.findall(text.translate(BN_DIGITS)))


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref)) on normalised text."""
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / max(1, len(r))


# ---------------------------------------------------------------- answer key

def tariff_rows_from_textlayer(page) -> dict[str, tuple]:
    """Rebuild 'HS CD SD VAT AIT AT RD TTI' rows from word coordinates."""
    lines: dict[int, list] = collections.defaultdict(list)
    for w in page.get_text("words"):
        lines[round(w[1] / 3)].append(w)
    rows = {}
    for ws in lines.values():
        ws.sort(key=lambda w: w[0])
        toks = [w[4] for w in ws]
        if toks and re.fullmatch(r"\d{8}", toks[0]):
            vals = [t for t in toks[1:] if re.fullmatch(r"\d+|[\d,]+\.\d+|-", t)]
            if len(vals) >= 6:
                rows[toks[0]] = tuple(vals[-7:]) if len(vals) >= 7 else tuple(vals)
    return rows


def tariff_rows_from_ocr(text: str, expected: dict[str, tuple]) -> dict[str, tuple]:
    rows = {}
    for line in text.splitlines():
        m = re.match(r"\s*\|?\s*(\d{8})\b(.*)", line)
        if m and m.group(1) in expected:
            vals = re.findall(r"[\d,]+\.\d+|\d+|(?<![\w.])-(?![\w])", m.group(2))
            rows[m.group(1)] = tuple(vals[-len(expected[m.group(1)]):])
    return rows


# ---------------------------------------------------------------- engines

def render(pdf: Path, page_no: int, dpi: int, out: Path) -> Path:
    with pymupdf.open(pdf) as d:
        d[page_no - 1].get_pixmap(dpi=dpi).save(out)
    return out


def preprocess(img: Path, how: str | None) -> Path:
    if not how:
        return img
    from PIL import Image, ImageOps
    im = ImageOps.grayscale(Image.open(img))
    if how == "otsu":
        import numpy as np
        a = np.asarray(im)
        hist = np.bincount(a.ravel(), minlength=256).astype(float)
        total, sum_all = a.size, (hist * np.arange(256)).sum()
        best_t, best_v, w0, s0 = 128, -1.0, 0.0, 0.0
        for t in range(256):
            w0 += hist[t]
            if w0 == 0 or w0 == total:
                continue
            s0 += t * hist[t]
            m0, m1 = s0 / w0, (sum_all - s0) / (total - w0)
            v = w0 * (total - w0) * (m0 - m1) ** 2
            if v > best_v:
                best_v, best_t = v, t
        im = im.point(lambda p: 255 if p > best_t else 0)
    out = img.with_name(img.stem + f"_{how}.png")
    im.save(out)
    return out


def run_tesseract(img: Path, lang: str, psm: str) -> str:
    return subprocess.run(["tesseract", str(img), "-", "-l", lang, "--psm", psm],
                          capture_output=True, text=True, check=True).stdout


def surya_available() -> bool:
    return shutil.which("surya_ocr") is not None


def run_surya_batch(images: dict[str, Path], workdir: Path) -> dict[str, str]:
    """Run the Surya CLI once over a folder of page images; return text per key."""
    src = workdir / "surya_in"
    out = workdir / "surya_out"
    src.mkdir(exist_ok=True)
    for key, img in images.items():
        shutil.copy(img, src / f"{key}.png")
    subprocess.run(["surya_ocr", str(src), "--output_dir", str(out)], check=True)
    results = next(out.rglob("results.json"))
    data = json.loads(results.read_text("utf-8"))
    texts = {}
    for name, pages in data.items():
        key = Path(name).stem
        parts = []
        for page in pages if isinstance(pages, list) else [pages]:
            if "blocks" in page:  # newer releases: HTML per layout block
                for b in page["blocks"]:
                    h = b.get("html") or ""
                    h = re.sub(r"</t[dh]>", " | ", h)
                    h = re.sub(r"</tr>|<br\s*/?>|</p>|</h\d>|</li>", "\n", h)
                    parts.append(html.unescape(re.sub(r"<[^>]+>", "", h)))
            for line in page.get("text_lines", []):  # older releases
                parts.append(line.get("text", ""))
        texts[key] = "\n".join(parts)
    return texts


# ---------------------------------------------------------------- scoring

def score(kind: str, page, ocr: str, ref: str | None) -> dict:
    tl = page.get_text()
    res = {}
    key_hs, got_hs = hs_codes(tl), hs_codes(ocr)
    if key_hs:
        res["hs_total"] = sum(key_hs.values())
        res["hs_correct"] = sum((key_hs & got_hs).values())
        res["hs_extra"] = sum((got_hs - key_hs).values())  # codes OCR invented
    key_n, got_n = numbers(tl), numbers(ocr)
    res["num_total"] = sum(key_n.values())
    res["num_correct"] = sum((key_n & got_n).values())
    if kind == "tariff_sheet":
        key_rows = tariff_rows_from_textlayer(page)
        got_rows = tariff_rows_from_ocr(ocr, key_rows)
        res["rows_total"] = len(key_rows)
        res["rows_exact"] = sum(1 for k, v in key_rows.items() if got_rows.get(k) == v)
    if ref:
        res["bn_cer"] = round(cer(ref, ocr), 4)
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(HERE / "pdfs"))
    ap.add_argument("--engines", default="all", help="comma list: tesseract variant names, 'surya', or 'all'")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for t in TEST_PAGES:
            print(*t)
        return
    pdf_dir = Path(args.pdf_dir)
    wanted = None if args.engines == "all" else set(args.engines.split(","))
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pages = []
        for f, p, kind, lang in TEST_PAGES:
            pdf = pdf_dir / f
            if not pdf.exists():
                print(f"skip (missing): {pdf}", file=sys.stderr)
                continue
            key = f"{Path(f).stem}_p{p}"
            ref_file = HERE / "refs" / f"{key}.txt"
            pages.append((key, pdf, p, kind, lang, ref_file.read_text("utf-8") if ref_file.exists() else None))

        def record(engine, key, pdf, p, kind, text, secs, ref):
            (out_dir / engine).mkdir(exist_ok=True)
            (out_dir / engine / f"{key}.txt").write_text(text, "utf-8")
            with pymupdf.open(pdf) as d:
                s = score(kind, d[p - 1], text, ref)
            rows.append({"engine": engine, "page": key, "kind": kind, "sec": round(secs, 2), **s})
            print(f"{engine:20s} {key:38s} {s}")

        for name, dpi, psm, prep in TESSERACT_VARIANTS:
            if wanted and name not in wanted:
                continue
            for key, pdf, p, kind, lang, ref in pages:
                img = preprocess(render(pdf, p, dpi, tmp / f"{key}_{dpi}.png"), prep)
                t0 = time.time()
                text = run_tesseract(img, lang, psm)
                record(name, key, pdf, p, kind, text, time.time() - t0, ref)

        if (wanted is None or "surya" in wanted):
            if not surya_available():
                print("surya_ocr not found: pip install surya-ocr (see README)", file=sys.stderr)
            else:
                imgs = {key: render(pdf, p, SURYA_DPI, tmp / f"{key}_surya.png") for key, pdf, p, *_ in pages}
                t0 = time.time()
                texts = run_surya_batch(imgs, tmp)
                per_page = (time.time() - t0) / max(1, len(imgs))
                for key, pdf, p, kind, lang, ref in pages:
                    record("surya", key, pdf, p, kind, texts.get(key, ""), per_page, ref)

    fields = ["engine", "page", "kind", "sec", "hs_total", "hs_correct", "hs_extra", "num_total",
              "num_correct", "rows_total", "rows_exact", "bn_cer"]
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary per engine
    print("\nSUMMARY")
    agg = collections.defaultdict(collections.Counter)
    cers = collections.defaultdict(list)
    for r in rows:
        for k in ("hs_total", "hs_correct", "hs_extra", "num_total", "num_correct", "rows_total", "rows_exact", "sec"):
            agg[r["engine"]][k] += r.get(k) or 0
        if r.get("bn_cer") is not None:
            cers[r["engine"]].append(r["bn_cer"])
    for e, a in agg.items():
        pct = lambda x, y: f"{100 * a[x] / a[y]:.1f}%" if a[y] else "-"
        print(f"{e:20s} HS {pct('hs_correct', 'hs_total'):>6s} (invented {a['hs_extra']})  "
              f"numbers {pct('num_correct', 'num_total'):>6s}  rate rows exact {pct('rows_exact', 'rows_total'):>6s}  "
              f"Bangla CER {sum(cers[e]) / len(cers[e]) if cers[e] else float('nan'):.3f}  "
              f"avg {a['sec'] / max(1, sum(1 for r in rows if r['engine'] == e)):.1f}s/page")


if __name__ == "__main__":
    main()
