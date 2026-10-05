"""Print the ingestion report for every ingested document (or one, by file name).

    python -m ingest.report [--pages] [name]
"""
from __future__ import annotations

import argparse
import json

from core.config import settings


def load_reports():
    for d in sorted((settings.data_dir / "docs").glob("*/report.json")):
        yield json.loads(d.read_text("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--pages", action="store_true", help="one line per page")
    args = ap.parse_args()
    tot = {}
    print(f"{'file':34} {'pages':>5} {'method':>22} {'corr(num)':>9} {'restored':>8} {'unconf':>6} "
          f"{'fallback':>8} {'s/page':>6}")
    for r in load_reports():
        if args.name and args.name not in r["filename"]:
            continue
        pp = r["per_page"]
        methods = {}
        for x in pp:
            methods[x["method"]] = methods.get(x["method"], 0) + 1
        s = lambda k: sum(x.get(k) or 0 for x in pp)
        ocr = [x for x in pp if x.get("ocr_seconds")]
        fb = sum(1 for x in pp if x.get("ocr_mode") == "block_fallback")
        print(f"{r['filename'][:34]:34} {len(pp):>5} {', '.join(f'{k} {v}' for k, v in methods.items()):>22} "
              f"{s('corrections'):>4}({s('corrections_num')}) {s('restored'):>8} {s('unconfirmed'):>6} {fb:>8} "
              f"{sum(x['ocr_seconds'] for x in ocr) / max(1, len(ocr)):>6.1f}")
        for k in ("corrections", "restored", "unconfirmed"):
            tot[k] = tot.get(k, 0) + s(k)
        tot["pages"] = tot.get("pages", 0) + len(pp)
        if args.pages:
            for x in pp:
                print(f"    p{x['page']:<4} {x['method']:<8} blocks {x.get('blocks')}  nonbn {x.get('nonbn_tokens')}  "
                      f"corr {x.get('corrections')}({x.get('corrections_num')})  rest {x.get('restored')}  "
                      f"unconf {x.get('unconfirmed')}  unassigned {x.get('unassigned_ref_values')}  "
                      f"conf {x.get('ocr_confidence')}  {x.get('ocr_mode')}")
    print("total", tot)


if __name__ == "__main__":
    main()
