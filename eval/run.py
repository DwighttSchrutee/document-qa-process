"""Run a question set through the HTTP API exactly as a user would, score it, store and report it.

    python -m eval.run --split test [--verified-only] [--url http://127.0.0.1:8000] [--judge] [--name NAME]

Question file: eval/questions_<split>.jsonl, one JSON object per line:
  {id, question, language, category, expected_answer, must_contain: [...], must_not_contain: [...],
   gold_sources: [{file, page}], answerable: true|false, verified: true|false}

Scoring (brief 9.3):
  correct            answerable: every must_contain value present (numbers digit-normalised, exact) and no
                     must_not_contain value; unanswerable: the system answered "not found"
  judge (optional)   qwen3:8b rubric for prose categories: correct / partial / wrong / refused; disagreements with the
                     rule-based score are listed for a person to review
  recall@50 / @8     a gold page among the candidates / the final passages (first search round)
  citation accuracy  the answer cites at least one gold page
  false not-found    answerable but answered "not found"
  hallucinated       final answer has a number not in any retrieved passage, or an unanswerable question answered
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import httpx
from psycopg.types.json import Jsonb

from answer.verify import numbers_in
from core.textnorm import digits_ascii, search_form
from store import db

HERE = Path(__file__).parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", search_form(s)).replace(",", "")


def contains(answer: str, value: str) -> bool:
    a, v = norm(answer), norm(value)
    if re.search(r"\d", v):
        core = re.escape(v.strip(" .%"))
        return re.search(rf"(?<![0-9]){core}(?![0-9])", a) is not None or \
            re.sub(r"[.\-/]", "", v) in re.sub(r"[.\-/]", "", a) and len(re.sub(r"\D", "", v)) >= 6
    return v in a


def ask(url: str, question: str) -> dict:
    out = {"sources": [], "answer": "", "events": 0}
    t0 = time.time()
    with httpx.stream("POST", f"{url}/ask", json={"question": question}, timeout=600) as r:
        for line in r.iter_lines():
            if not line.strip():
                continue
            ev = json.loads(line)
            out["events"] += 1
            if ev["type"] == "sources":
                out["sources"] = ev["passages"]
                out["round"] = ev["round"]
            elif ev["type"] == "checks":
                out["checks"] = ev
            elif ev["type"] == "done":
                out.update(ev)
            elif ev["type"] == "error":
                out["error"] = ev["text"]
    out["seconds"] = round(time.time() - t0, 2)
    return out


def page_set(conn, ids: list[int]) -> set[tuple[str, int]]:
    if not ids:
        return set()
    rows = conn.execute("SELECT d.filename, c.pages FROM chunks c JOIN documents d ON d.id = c.document_id "
                        "WHERE c.id = ANY(%s)", (ids,)).fetchall()
    return {(f, p) for f, ps in rows for p in ps or []}


JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string",
                                                              "enum": ["correct", "partial", "wrong", "refused"]},
                                                  "reason": {"type": "string"}}, "required": ["verdict", "reason"]}


def judge(q: dict, answer: str) -> dict:
    from answer import llm
    prompt = (f"Grade an answer against the reference.\nQuestion: {q['question']}\nReference answer: "
              f"{q['expected_answer']}\nRequired facts: {q['must_contain']}\nAnswer to grade: {answer}\n\n"
              "verdict: correct (all required facts, nothing contradicting), partial, wrong, or refused "
              "(says it cannot answer). Return JSON with verdict and a one-sentence reason.")
    try:
        return json.loads(llm.chat([{"role": "user", "content": prompt}], fmt=JUDGE_SCHEMA, num_ctx=8192,
                                   num_predict=300))
    except Exception as e:
        return {"verdict": "error", "reason": str(e)[:200]}


def score(q: dict, r: dict, conn) -> dict:
    ans = r.get("answer", "") or ""
    nf = bool(r.get("not_found"))
    gold = {(g["file"], g["page"]) for g in q.get("gold_sources", [])}
    why = []
    if q["answerable"]:
        missing = [v for v in q.get("must_contain", []) if not contains(ans, v)]
        forbidden = [v for v in q.get("must_not_contain", []) if contains(ans, v)]
        correct = not nf and not missing and not forbidden
        if nf:
            why.append("answered not found")
        if missing:
            why.append("missing " + ", ".join(missing))
        if forbidden:
            why.append("contains forbidden " + ", ".join(forbidden))
    else:
        correct = nf
        if not nf:
            why.append("answered an unanswerable question")
    log = None
    if r.get("log_id"):
        log = conn.execute("SELECT retrieved FROM qa_log WHERE id = %s", (r["log_id"],)).fetchone()
    first = (log[0][0] if log and log[0] else {}) if log else {}
    cand_pages = page_set(conn, first.get("candidates", []))
    final_pages = page_set(conn, first.get("ids", []))
    cited_pages = {(s["filename"], p) for s in r.get("sources", []) if s["n"] in (r.get("citations") or [])
                   for p in s.get("pages") or []}
    all_text = " ".join(s.get("text", "") for s in r.get("sources", []))
    unsupported = [n for n in numbers_in(ans) if not _num_in(n, all_text) and not _num_in(n, q["question"])]
    hallucinated = bool(unsupported) or (not q["answerable"] and not nf)
    checks = r.get("checks") or {}
    return {"question": q["question"], "category": q["category"], "language": q["language"],
            "answerable": q["answerable"], "verified": q.get("verified", False), "expected": q.get("expected_answer"),
            "answer": ans, "not_found": nf, "correct": correct, "why": why,
            "recall50": bool(gold & cand_pages) if gold else None, "recall8": bool(gold & final_pages) if gold else None,
            "citation_ok": bool(gold & cited_pages) if gold and not nf else None,
            "hallucinated": hallucinated, "unsupported_numbers": unsupported,
            "verifier_fired": bool(checks.get("regenerated")), "verifier_removed": checks.get("removed", []),
            "warnings": r.get("warnings", []), "seconds": r.get("seconds"), "log_id": r.get("log_id"),
            "error": r.get("error"), "sources": [(s["filename"], s.get("pages")) for s in r.get("sources", [])]}


def _num_in(n: str, text: str) -> bool:
    v = digits_ascii(n).rstrip("%").replace(",", "")
    t = digits_ascii(text).replace(",", "")
    return re.search(rf"(?<![0-9]){re.escape(v)}(?![0-9])", t) is not None or \
        re.sub(r"\D", "", v) in re.sub(r"\D", "", t) and len(re.sub(r"\D", "", v)) >= 6


def metrics(results: list[dict]) -> dict:
    def grp(rows):
        ans = [x for x in rows if x["answerable"]]
        un = [x for x in rows if not x["answerable"]]
        secs = sorted(x["seconds"] for x in rows if x["seconds"] is not None)
        rate = lambda xs, k: round(sum(1 for x in xs if x[k]) / len(xs), 4) if xs else None
        r50 = [x for x in ans if x["recall50"] is not None]
        cit = [x for x in ans if x["citation_ok"] is not None]
        return {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(un),
                "accuracy": rate(ans, "correct"), "correct": sum(1 for x in ans if x["correct"]),
                "unanswerable_refused": rate(un, "correct"),
                "recall50": rate(r50, "recall50"), "recall8": rate(r50, "recall8"),
                "citation_accuracy": rate(cit, "citation_ok"),
                "false_not_found": rate(ans, "not_found"),
                "hallucinated_numbers": sum(1 for x in ans if x["unsupported_numbers"]),
                "unanswerable_answered": sum(1 for x in un if not x["correct"]),
                "verifier_fired": sum(1 for x in rows if x["verifier_fired"]),
                "verifier_removed": sum(1 for x in rows if x["verifier_removed"]),
                "p50": secs[len(secs) // 2] if secs else None,
                "p95": secs[min(len(secs) - 1, int(len(secs) * 0.95))] if secs else None}
    groups = {"all": results}
    for x in results:
        groups.setdefault("cat:" + x["category"], []).append(x)
        groups.setdefault("lang:" + x["language"], []).append(x)
    return {"groups": {k: grp(v) for k, v in groups.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--judge", action="store_true", help="also grade prose answers with qwen3:8b")
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    qs = [json.loads(x) for x in (HERE / f"questions_{a.split}.jsonl").read_text("utf-8").splitlines() if x.strip()]
    if a.verified_only:
        qs = [q for q in qs if q.get("verified")]
    if a.limit:
        qs = qs[:a.limit]
    results = []
    with db.connect() as conn:
        for i, q in enumerate(qs, 1):
            r = ask(a.url, q["question"])
            s = {"id": q["id"], **score(q, r, conn)}
            if a.judge and q["answerable"] and q["category"] != "table_lookup":
                s["judge"] = judge(q, s["answer"])
                rule = "correct" if s["correct"] else "wrong"
                s["judge_disagrees"] = (s["judge"]["verdict"] == "correct") != (rule == "correct")
            results.append(s)
            print(f"[{i}/{len(qs)}] {'OK ' if s['correct'] else 'BAD'} {s['seconds']}s {q['id']} {q['question'][:70]}"
                  + (f"  ({'; '.join(s['why'])})" if s["why"] else ""), flush=True)
        m = metrics(results)
        m["sizes"] = {"questions": len(qs), "verified": sum(1 for q in qs if q.get("verified"))}
        run_id = conn.execute("INSERT INTO eval_runs (name, config, metrics) VALUES (%s, %s, %s) RETURNING id",
                              (a.name or f"{a.split}{'-verified' if a.verified_only else ''}",
                               Jsonb({"split": a.split, "verified_only": a.verified_only, "judge": a.judge}),
                               Jsonb(m))).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO eval_results (run_id, question_id, result) VALUES (%s, %s, %s)",
                            [(run_id, s["id"], Jsonb(s)) for s in results])
    from eval.report import write_report
    path = write_report(run_id)
    g = m["groups"]["all"]
    print(f"\nrun {run_id}: accuracy {g['accuracy']} ({g['correct']}/{g['n_answerable']}), recall@8 {g['recall8']}, "
          f"false not-found {g['false_not_found']}, unanswerable refused {g['unanswerable_refused']}, "
          f"p50 {g['p50']}s p95 {g['p95']}s -> {path}")


if __name__ == "__main__":
    main()
