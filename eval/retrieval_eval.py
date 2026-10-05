"""Retrieval recall on a question split.

recall@50: a gold (file, page) is among the candidates; recall@8: among the final context passages.
Unanswerable questions are excluded from recall (their top re-ranker score is reported instead).

    python -m eval.retrieval_eval --split dev [--config hybrid] [--rewrite regex|llm] [--all-configs]
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from answer.understand import understand
from retrieve.search import Options, retrieve
from store import db

HERE = Path(__file__).parent
CACHE = HERE / "cache_understand.json"

CONFIGS = {
    "hybrid": Options(),
    "vector_only": Options(keyword=False, identifiers=False, amendments=False),
    "hybrid_no_rerank": Options(rerank=False),
    "no_amendments": Options(amendments=False),
}


def load(split: str) -> list[dict]:
    return [json.loads(x) for x in (HERE / f"questions_{split}.jsonl").read_text("utf-8").splitlines() if x.strip()]


def understood(q: dict, mode: str, cache: dict) -> dict:
    if mode == "regex":
        return understand(q["question"], use_llm=False)
    key = q["question"]
    if key not in cache:
        cache[key] = understand(q["question"], use_llm=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), "utf-8")
    return cache[key]


def gold_hit(row: dict, gold: list[dict]) -> bool:
    return any(row["filename"] == g["file"] and g["page"] in (row["pages"] or []) for g in gold)


def run(split: str, opts: Options, rewrite: str, verbose: bool = False) -> dict:
    qs = load(split)
    cache = json.loads(CACHE.read_text("utf-8")) if CACHE.exists() else {}
    per = []
    with db.connect() as conn:
        for q in qs:
            u = understood(q, rewrite, cache)
            t0 = time.time()
            r = retrieve(q["question"], u, opts, conn=conn)
            secs = time.time() - t0
            if q["answerable"]:
                at50 = any(gold_hit(h.row, q["gold_sources"]) for h in r["candidates"])
                at8 = any(gold_hit(h.row, q["gold_sources"]) for h in r["hits"])
            else:
                at50 = at8 = None
            top = max((h.rerank for h in r["hits"] if h.rerank is not None), default=None)
            per.append({"id": q["id"], "category": q["category"], "language": q["language"],
                        "answerable": q["answerable"], "verified": q.get("verified", False), "r50": at50, "r8": at8,
                        "top_rerank": top, "seconds": round(secs, 2),
                        "hits": [(h.row["filename"], h.row["pages"], round(h.rerank or 0, 3)) for h in r["hits"]]})
            if verbose and q["answerable"] and not at8:
                print(f"  MISS@8 {q['id']} [{q['category']}/{q['language']}] {q['question'][:90]}")
                print(f"     gold {q['gold_sources']}  got {[(f, p) for f, p, _ in per[-1]['hits'][:4]]}")
    return summarise(per)


def summarise(per: list[dict]) -> dict:
    def rate(rows, k):
        rows = [x for x in rows if x["answerable"]]
        return (sum(1 for x in rows if x[k]), len(rows))
    groups = collections.defaultdict(list)
    for x in per:
        groups["all"].append(x)
        groups["cat:" + x["category"]].append(x)
        groups["lang:" + x["language"]].append(x)
        groups["verified" if x["verified"] else "unverified"].append(x)
    out = {g: {"r50": rate(rows, "r50"), "r8": rate(rows, "r8"), "n": len(rows),
               "p50_s": sorted(x["seconds"] for x in rows)[len(rows) // 2]} for g, rows in groups.items()}
    un = [x["top_rerank"] for x in per if not x["answerable"] and x["top_rerank"] is not None]
    an = [x["top_rerank"] for x in per if x["answerable"] and x["top_rerank"] is not None]
    out["top_rerank_unanswerable"] = sorted(round(v, 3) for v in un)
    out["top_rerank_answerable_median"] = sorted(an)[len(an) // 2] if an else None
    out["per_question"] = per
    return out


def show(name: str, s: dict):
    print(f"\n== {name}")
    for g in sorted(k for k in s if k not in ("per_question", "top_rerank_unanswerable", "top_rerank_answerable_median")):
        (a50, n50), (a8, n8) = s[g]["r50"], s[g]["r8"]
        if n8:
            print(f"  {g:22} recall@50 {a50:3}/{n50:<3} ({a50 / n50:6.1%})   recall@8 {a8:3}/{n8:<3} ({a8 / n8:6.1%})"
                  f"   median {s[g]['p50_s']}s")
    print(f"  unanswerable top re-ranker scores: {s['top_rerank_unanswerable']}; "
          f"answerable median {s['top_rerank_answerable_median']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default="hybrid", choices=list(CONFIGS))
    ap.add_argument("--rewrite", default="regex", choices=["regex", "llm"])
    ap.add_argument("--all-configs", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    runs = {}
    names = list(CONFIGS) if a.all_configs else [a.config]
    for n in names:
        runs[f"{n}/{a.rewrite}"] = run(a.split, CONFIGS[n], a.rewrite, verbose=a.v)
        show(f"{n}/{a.rewrite}", runs[f"{n}/{a.rewrite}"])
    out = HERE / f"retrieval_{a.split}.json"
    old = json.loads(out.read_text("utf-8")) if out.exists() else {}
    old.update(runs)
    out.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
