"""Ablations (brief 9.5): same questions, one component changed at a time. Runs in-process (the options are
internal), stores each configuration as an eval run named 'ablation:<name>' for eval/report.md.

    python -m eval.ablation --split dev [--only hybrid,vector_only]
qwen3-embedding:4b needs its vectors first: python -m store.vectors qwen3-embedding:4b
"""
from __future__ import annotations

import argparse
import json
import time

from psycopg.types.json import Jsonb

from answer import pipeline
from eval.run import HERE, metrics, score
from retrieve.search import Options
from store import db

CONFIGS = {
    "hybrid (default)": (Options(), True),
    "vector only": (Options(keyword=False, identifiers=False, amendments=False), True),
    "without re-ranker": (Options(rerank=False), True),
    "qwen3-embedding:4b": (Options(embed_model="qwen3-embedding:4b"), True),
    "without second round": (Options(), False),
}


def run_config(name, opts, second, qs, conn):
    results = []
    for q in qs:
        t0 = time.time()
        out = {"sources": []}
        for ev in pipeline.ask(q["question"], opts=opts, second_round=second):
            if ev["type"] == "sources":
                out["sources"] = ev["passages"]
            elif ev["type"] == "checks":
                out["checks"] = ev
            elif ev["type"] == "done":
                out.update(ev)
        out["seconds"] = round(time.time() - t0, 2)
        results.append({"id": q["id"], **score(q, out, conn)})
    m = metrics(results)
    m["sizes"] = {"questions": len(qs), "verified": sum(1 for q in qs if q.get("verified"))}
    rid = conn.execute("INSERT INTO eval_runs (name, config, metrics) VALUES (%s, %s, %s) RETURNING id",
                       (f"ablation:{name}", Jsonb({"options": opts.__dict__, "second_round": second}),
                        Jsonb(m))).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO eval_results (run_id, question_id, result) VALUES (%s, %s, %s)",
                        [(rid, s["id"], Jsonb(s)) for s in results])
    conn.commit()
    g = m["groups"]["all"]
    print(f"{name:24} accuracy {g['accuracy']}  recall@50 {g['recall50']}  recall@8 {g['recall8']}  "
          f"false-NF {g['false_not_found']}  p50 {g['p50']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    qs = [json.loads(x) for x in (HERE / f"questions_{a.split}.jsonl").read_text("utf-8").splitlines() if x.strip()]
    want = [x.strip() for x in a.only.split(",") if x.strip()]
    with db.connect() as conn:
        for name, (opts, second) in CONFIGS.items():
            if want and not any(w in name for w in want):
                continue
            run_config(name, opts, second, qs, conn)


if __name__ == "__main__":
    main()
