"""eval/report.md from a stored evaluation run (and the ablation table from runs named 'ablation:*').

    python -m eval.report [RUN_ID]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

from store import db

HERE = Path(__file__).parent


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def cause(r: dict) -> str:
    """First guess at the failing stage; a person confirms it in the review of failures."""
    if r.get("error"):
        return "system error"
    if not r["answerable"]:
        return "generation (answered an unanswerable question)"
    if r["recall50"] is False:
        return "retrieval (gold page not among candidates) — or extraction/quality gate if the page is held"
    if r["recall8"] is False:
        return "re-ranking (gold page found but not in the final context)"
    if r["not_found"]:
        return "generation (false not-found with the gold page in context)"
    if r["unsupported_numbers"]:
        return "generation (value not in passages)"
    return "generation or gold error (gold page in context, answer missing a required fact)"


def write_report(run_id: int | None = None) -> Path:
    with db.connect() as conn:
        if run_id is None:
            run_id = conn.execute("SELECT max(id) FROM eval_runs WHERE name NOT LIKE 'ablation:%%'").fetchone()[0]
        name, at, config, m = conn.execute("SELECT name, at, config, metrics FROM eval_runs WHERE id = %s",
                                           (run_id,)).fetchone()
        results = [r for (r,) in conn.execute("SELECT result FROM eval_results WHERE run_id = %s ORDER BY question_id",
                                              (run_id,)).fetchall()]
        ablations = conn.execute("""SELECT DISTINCT ON (name) name, metrics FROM eval_runs WHERE name LIKE 'ablation:%%'
                                    ORDER BY name, at DESC""").fetchall()
    g = m["groups"]
    L = [f"# Evaluation report — run {run_id} ({name}, {at:%Y-%m-%d %H:%M})", "",
         f"Questions: {m['sizes']['questions']} ({m['sizes']['verified']} human-verified). Config: {config}.",
         "Numbers below are exact counts; nothing is rounded away. Unverified questions are included only if the run "
         "was not `--verified-only`.", "",
         "## Overall and per group", "",
         "| group | n (ans/unans) | answer accuracy | unanswerable refused | recall@50 | recall@8 | citation acc. | "
         "false not-found | hallucinated numbers | unanswerable answered | verifier fired / removed | p50 / p95 s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ["all"] + sorted(x for x in g if x != "all"):
        v = g[k]
        L.append(f"| {k} | {v['n_answerable']}/{v['n_unanswerable']} | {pct(v['accuracy'])} ({v['correct']}) | "
                 f"{pct(v['unanswerable_refused'])} | {pct(v['recall50'])} | {pct(v['recall8'])} | "
                 f"{pct(v['citation_accuracy'])} | {pct(v['false_not_found'])} | {v['hallucinated_numbers']} | "
                 f"{v['unanswerable_answered']} | {v['verifier_fired']} / {v['verifier_removed']} | "
                 f"{v['p50']} / {v['p95']} |")
    a = g["all"]
    targets = [("Table lookups ≥ 98% accuracy", (g.get("cat:table_lookup") or {}).get("accuracy"), 0.98, True),
               ("Overall answerable accuracy ≥ 90%", a["accuracy"], 0.90, True),
               ("False not-found ≤ 3%", a["false_not_found"], 0.03, False),
               ("Unanswerable correctly refused ≥ 90%", a["unanswerable_refused"], 0.90, True)]
    L += ["", "## Targets", "", "| target | actual | met |", "|---|---|---|"]
    for t, v, goal, higher in targets:
        met = "–" if v is None else ("yes" if (v >= goal if higher else v <= goal) else "**no**")
        L.append(f"| {t} | {pct(v)} | {met} |")
    L.append(f"| Hallucinated numbers in final answers: 0 | {a['hallucinated_numbers']} | "
             f"{'yes' if a['hallucinated_numbers'] == 0 else '**no**'} |")
    if ablations:
        L += ["", "## Ablations", "", "| configuration | accuracy | recall@50 | recall@8 | false not-found | p50 s |",
              "|---|---|---|---|---|---|"]
        for n, am in ablations:
            v = am["groups"]["all"]
            L.append(f"| {n[9:]} | {pct(v['accuracy'])} | {pct(v['recall50'])} | {pct(v['recall8'])} | "
                     f"{pct(v['false_not_found'])} | {v['p50']} |")
    fails = [r for r in results if not r["correct"]]
    causes = collections.Counter(cause(r).split(" (")[0] for r in fails)
    L += ["", f"## Failures ({len(fails)})", "", "Diagnosed cause (first guess, to confirm by hand):", ""]
    L += [f"- {c}: {n}" for c, n in causes.most_common()]
    by_cat = collections.defaultdict(list)
    for r in fails:
        by_cat[r["category"]].append(r)
    for cat, rows in sorted(by_cat.items()):
        L += ["", f"### {cat} ({len(rows)})"]
        for r in rows:
            L += ["", f"**{r.get('id', '')}** {r['question']}", f"- expected: {r['expected']}",
                  f"- answer: {(r['answer'] or '').strip()[:600]}", f"- why: {'; '.join(r['why'])}",
                  f"- retrieved: {r['sources'][:8]}", f"- cause: {cause(r)}"]
            if r.get("judge"):
                L.append(f"- judge: {r['judge']}")
    disagree = [r for r in results if r.get("judge_disagrees")]
    if disagree:
        L += ["", f"## Judge vs rules disagreements ({len(disagree)}) — review by hand", ""]
        L += [f"- {r.get('id')}: rules={'correct' if r['correct'] else 'wrong'}, judge={r['judge']['verdict']} "
              f"({r['judge']['reason']})" for r in disagree]
    L += ["", "## Ranked fixes", ""]
    L += [f"{i}. {c} — {n} failure(s)" for i, (c, n) in enumerate(causes.most_common(), 1)] or ["(no failures)"]
    out = HERE / "report.md"
    out.write_text("\n".join(L) + "\n", "utf-8")
    return out


if __name__ == "__main__":
    print(write_report(int(sys.argv[1]) if len(sys.argv) > 1 else None))
