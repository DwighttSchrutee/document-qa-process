"""OCR regression bench: Surya alone vs Surya + text-layer merge on the 13 bench pages.

Uses the same pages and scoring as ocr_bench.py (answer key = PDF text layer; Bangla CER from refs/).
Surya results come from the ingestion cache, so run ingestion (or scratch/ocr_bench_pages.py) first.

    python -m bench.run_bench            # prints the table, writes bench/results_fusion.csv
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import pymupdf

from core.config import settings
from ingest import fusion, textlayer
from ingest.surya_ocr import cache_path

HERE = Path(__file__).parent
sys.argv = sys.argv[:1]
from bench import ocr_bench as ob   # noqa: E402  (scoring helpers, unchanged)

FILES = {"Tariff-2025-202602-06-2025.pdf": "Tariff-2025-2026(02-06-2025).pdf"}


def surya_text(sp: dict) -> str:
    """Surya's own reading, rendered exactly like the merge output (no corrections)."""
    out = []
    for b in sorted(sp["blocks"], key=lambda b: b.get("reading_order", 0)):
        if b.get("skipped") or b.get("label") in fusion.SKIP_LABELS:
            continue
        runs, struct = fusion.block_runs(b)
        out.append(fusion.rebuild(runs, struct)["text"])
    return "\n".join(out)


def run(engines=("surya", "fusion")) -> list[dict]:
    rows = []
    for f, p, kind, _ in ob.TEST_PAGES:
        path = settings.docs_dir / FILES.get(f, f)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cp = cache_path(sha, p)
        if not cp.exists():
            print(f"missing Surya cache for {f} p{p}; run ingestion first", file=sys.stderr)
            continue
        sp = json.loads(cp.read_text("utf-8"))
        page = pymupdf.open(path)[p - 1]
        ref_file = HERE / "refs" / f"{Path(f).stem}_p{p}.txt"
        ref = ref_file.read_text("utf-8") if ref_file.exists() else None
        texts = {"surya": surya_text(sp)}
        if "fusion" in engines:
            m = fusion.merge_page(sp, textlayer.words(page), (page.rect.width, page.rect.height))
            texts["fusion"] = "\n".join(b["text"] for b in m["blocks"] if b.get("text"))
            unconf = m["metrics"]["unconfirmed"]
        for e in engines:
            r = {"engine": e, "page": f"{Path(f).stem}_p{p}", "kind": kind, "sec": sp["seconds"],
                 **ob.score(kind, page, texts[e], ref)}
            if e == "fusion":
                r["unconfirmed"] = unconf
                r.update({k: m["metrics"][k] for k in ("corrections", "corrections_num", "restored")})
            rows.append(r)
            out = HERE / "out" / e
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{r['page']}.txt").write_text(texts[e], "utf-8")
    return rows


def summary(rows):
    agg = collections.defaultdict(collections.Counter)
    cers = collections.defaultdict(list)
    for r in rows:
        for k in ("hs_total", "hs_correct", "hs_extra", "num_total", "num_correct", "rows_total", "rows_exact",
                  "sec", "unconfirmed"):
            agg[r["engine"]][k] += r.get(k) or 0
        if r.get("bn_cer") is not None:
            cers[r["engine"]].append((r["page"], r["bn_cer"]))
    print(f"{'engine':8} {'HS':>16} {'invented':>8} {'numbers':>16} {'rate rows':>10} {'unconfirmed':>11}  Bangla CER")
    for e, a in agg.items():
        frac = lambda x, y: f"{a[x]}/{a[y]} ({100 * a[x] / a[y]:.1f}%)" if a[y] else "-"
        print(f"{e:8} {frac('hs_correct', 'hs_total'):>16} {a['hs_extra']:>8} {frac('num_correct', 'num_total'):>16} "
              f"{a['rows_exact']:>4}/{a['rows_total']:<5} {a['unconfirmed']:>11}  "
              + ", ".join(f"{p}: {c:.2%}" for p, c in cers[e]))


if __name__ == "__main__":
    rows = run()
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("engine", "page", "kind"), k))
    with open(HERE / "results_fusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary(rows)
    print("\nper page (fusion):")
    for r in rows:
        if r["engine"] == "fusion":
            print(f"  {r['page']:36} hs {r.get('hs_correct', '-')}/{r.get('hs_total', '-')} extra {r.get('hs_extra', '-')}"
                  f"  num {r['num_correct']}/{r['num_total']}  rows {r.get('rows_exact', '-')}/{r.get('rows_total', '-')}"
                  f"  corr {r['corrections']} (num {r['corrections_num']}) restored {r['restored']} unconf {r['unconfirmed']}")
