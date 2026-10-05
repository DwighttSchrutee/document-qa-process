"""Ingest files: deduplicate by SHA-256, extract every page, write page JSON and a per-page report.

    python -m ingest.pipeline                       # every file in settings.docs_dir
    python -m ingest.pipeline FILE [FILE ...] [--pages 1,61,101]

Output: data/docs/<sha>/meta.json, data/docs/<sha>/pages/p0001.json, data/docs/<sha>/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from core.config import settings
from ingest import extract


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_dir(sha: str) -> Path:
    return settings.data_dir / "docs" / sha


def dedupe(paths: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """sha -> first path (sorted by name), sha -> duplicate paths."""
    unique, dups = {}, {}
    for p in sorted(paths, key=lambda p: p.name):
        s = sha256(p)
        if s in unique:
            dups.setdefault(s, []).append(p)
        else:
            unique[s] = p
    return unique, dups


def ingest_file(path: Path, sha: str, pages=None, log=print, index: bool = True) -> dict:
    out = doc_dir(sha)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    meta = {"sha256": sha, "filename": path.name, "path": str(path), "kind": extract.file_kind(path)}
    if meta["kind"] == "pdf":
        import pymupdf
        meta["pages"] = pymupdf.open(path).page_count
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    t0 = time.time()
    rows = []
    for rec in extract.extract(path, sha, pages,
                               progress=lambda p, n: log(f"  {path.name}: page {p} ({n} to do)") if p % 25 == 0 else None):
        (out / "pages" / f"p{rec['page_no']:04d}.json").write_text(json.dumps(rec, ensure_ascii=False), "utf-8")
        m = rec["metrics"]
        rows.append({"page": rec["page_no"], "method": rec["method"], "class": rec["page_class"],
                     **{k: m.get(k) for k in ("blocks", "nonbn_tokens", "corrections", "corrections_num",
                                               "restored", "unconfirmed", "unconfirmed_num", "cancelled_pairs",
                                               "unassigned_ref_values", "correction_rate", "ocr_mode",
                                               "ocr_confidence", "ocr_seconds")}})
    report = {**meta, "seconds": round(time.time() - t0, 1), "pages_done": len(rows), "per_page": rows}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    from ingest import gate
    report["gate"] = gate.gate_document(sha)
    log(f"   {report['gate']['summary']}")
    if index:
        try:
            from store import indexer
            report["index"] = indexer.index_document(sha, log=lambda m: log("   " + m))
        except Exception as e:                    # database not running: ingestion results are kept on disk
            log(f"   not indexed ({type(e).__name__}: {e}); run `make index` later")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--pages", help="comma list of 1-based pages (single file only)")
    args = ap.parse_args()
    files = args.files or [p for p in settings.docs_dir.iterdir() if p.suffix.lower() in extract.SUPPORTED]
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    unique, dups = dedupe(files)
    for s, ps in dups.items():
        print(f"duplicate of {unique[s].name}: {', '.join(p.name for p in ps)} (skipped)")
    # small documents first so results arrive early
    order = sorted(unique.items(), key=lambda kv: kv[1].stat().st_size)
    for s, path in order:
        print(f"== {path.name} ({s[:10]})", flush=True)
        rep = ingest_file(path, s, pages, log=lambda m: print(m, flush=True))
        print(f"   {rep['pages_done']} pages in {rep['seconds']}s", flush=True)
    from ingest import surya_ocr
    surya_ocr.stop()


if __name__ == "__main__":
    sys.exit(main())
