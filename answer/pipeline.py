"""Answer a question: understand -> retrieve -> generate (streamed) -> verify -> retry -> log.

Events (NDJSON over HTTP):
  {"type": "status", "text": ...}
  {"type": "sources", "passages": [...], "round": 1|2}
  {"type": "thinking", "text": ...}            only with think=True
  {"type": "token", "text": ...}
  {"type": "reset", "reason": ...}             the answer so far is replaced by a new attempt
  {"type": "checks", ...}
  {"type": "done", "answer": ..., "not_found": bool, "warnings": [...], "citations": [...], "timings": {...}}
"""
from __future__ import annotations

import time
from typing import Iterator

from psycopg.types.json import Jsonb

from answer import llm, verify
from answer.prompt import NOT_FOUND, build_messages
from answer.understand import understand
from core.config import settings
from retrieve.search import Options, retrieve, retrieve_wide

REWRITE_MODE = "regex"           # first round: "regex" (fast) or "llm"; the second round always uses the model


def _passages(hits) -> list[dict]:
    out = []
    for h in hits:
        r = dict(h.row)
        r["added_for"] = h.added_for
        r["rerank"] = round(h.rerank, 4) if h.rerank is not None else None
        r["exact"] = h.exact
        out.append(r)
    return out


def _public(p: dict, n: int) -> dict:
    keep = ("filename", "sha256", "pages", "bbox", "section_path", "clause_no", "kind", "status", "ref_key", "issued",
            "effective_from", "effective_to", "rerank", "exact", "added_for", "text", "id")
    return {"n": n, **{k: p.get(k) for k in keep}}


def _generate(question, passages, think, feedback=None, history=None) -> Iterator[dict]:
    msgs = build_messages(question, passages, feedback, history)
    yield from llm.chat_stream(msgs, think=think, num_ctx=settings.llm_num_ctx)


def ask(question: str, think: bool = False, history: list[dict] | None = None, opts: Options | None = None,
        log: bool = True, second_round: bool = True) -> Iterator[dict]:
    t0 = time.time()
    timings: dict[str, float] = {}
    opts = opts or Options()
    warnings: list[str] = []
    yield {"type": "status", "text": "searching"}
    u = understand(question, use_llm=REWRITE_MODE == "llm")
    r = retrieve(question, u, opts)
    timings["retrieve"] = round(time.time() - t0, 2)
    passages = _passages(r["hits"])
    rounds = [{"round": 1, "ids": [p["id"] for p in passages], "rerank": [p["rerank"] for p in passages],
               "candidates": [h.id for h in r["candidates"]], "queries": r["queries"]}]
    yield {"type": "sources", "round": 1, "passages": [_public(p, i) for i, p in enumerate(passages, 1)]}

    def attempt(feedback=None):
        text, stats = "", {}
        for ev in _generate(question, passages, think, feedback, history):
            if ev["type"] == "token":
                text += ev["text"]
                yield ev
            elif ev["type"] == "thinking":
                yield ev
            else:
                stats = ev
        return text, stats

    answer, stats = yield from attempt()
    timings["generate1"] = round(time.time() - t0, 2)

    # ---- "not found": search again, wider, before saying so
    if second_round and (answer.strip().startswith(NOT_FOUND) or not passages):
        yield {"type": "status", "text": "not found in the first search; searching more widely"}
        u2 = understand(question, use_llm=True)
        r2 = retrieve_wide(question, u2, opts)
        timings["retrieve2"] = round(time.time() - t0, 2)
        passages = _passages(r2["hits"])
        rounds.append({"round": 2, "ids": [p["id"] for p in passages], "rerank": [p["rerank"] for p in passages],
                       "candidates": [h.id for h in r2["candidates"]], "queries": r2["queries"]})
        yield {"type": "reset", "reason": "second search"}
        yield {"type": "sources", "round": 2, "passages": [_public(p, i) for i, p in enumerate(passages, 1)]}
        answer, stats = yield from attempt()
        timings["generate2"] = round(time.time() - t0, 2)

    not_found = answer.strip().startswith(NOT_FOUND)
    checks = verify.check(answer, passages, question)
    regenerated = False
    # ---- numbers must come from the cited passages; citations must exist
    if not not_found and (checks["unsupported_numbers"] or checks["bad_citations"] or not checks["has_citation"]
                          or checks["placeholder"]):
        problems = []
        if checks["placeholder"]:
            problems.append("the answer contains template text in angle brackets instead of an answer")
        if checks["unsupported_numbers"]:
            problems.append("the value(s) " + ", ".join(checks["unsupported_numbers"]) +
                            " do not appear in the passages you cited")
        if checks["bad_citations"]:
            problems.append("citation(s) " + ", ".join(f"[{c}]" for c in checks["bad_citations"]) +
                            f" do not exist (there are {len(passages)} passages)")
        if not checks["has_citation"]:
            problems.append("no statement has a citation [n]")
        yield {"type": "reset", "reason": "; ".join(problems)}
        answer, stats = yield from attempt("; ".join(problems) + ".")
        timings["regenerate"] = round(time.time() - t0, 2)
        regenerated = True
        not_found = answer.strip().startswith(NOT_FOUND)
        checks = verify.check(answer, passages, question)
        if checks["unsupported_numbers"]:
            answer = verify.remove_unsupported(answer, checks["unsupported_numbers"])
            warnings.append("Removed statements with values not found in the cited passages: "
                            + ", ".join(checks["unsupported_numbers"]))
            checks = {**verify.check(answer, passages, question), "removed": checks["unsupported_numbers"]}
            yield {"type": "reset", "reason": "unsupported values removed"}
            yield {"type": "token", "text": answer}
        if checks["bad_citations"]:
            answer = verify.strip_bad_citations(answer, checks["bad_citations"])
    held = _held_amendments(passages)
    if held:
        warnings.append("An amending notice (" + ", ".join(held) + ") has lines held for review, so the answer may "
                        "not reflect the latest amendment. Check the Review page.")
    if checks["cites_unverified"]:
        warnings.append("Passage(s) " + ", ".join(f"[{c}]" for c in checks["cites_unverified"]) +
                        " could not be verified against the document (quality gate warning).")
    if not_found:
        closest = sorted(passages, key=lambda p: -(p.get("rerank") or 0))[:3]
        checks["closest"] = [_public(p, passages.index(p) + 1) for p in closest]
    checks["regenerated"] = regenerated
    yield {"type": "checks", **{k: v for k, v in checks.items() if k != "closest"}}
    timings["total"] = round(time.time() - t0, 2)
    result = {"type": "done", "answer": answer, "not_found": not_found, "warnings": warnings,
              "citations": checks.get("citations", []), "closest": checks.get("closest"),
              "timings": timings, "llm": {k: v for k, v in stats.items() if k != "type"}}
    if log:
        result["log_id"] = _log(question, {"think": think, "rewrite": REWRITE_MODE}, rounds, answer, checks, timings)
    yield result


def _held_amendments(passages: list[dict]) -> list[str]:
    """Amending notices in the context whose documents still have held rows / lines / pages."""
    docs = {p["document_id"]: p for p in passages
            if (p.get("added_for") or "").startswith(("amends", "rescinds")) and p.get("document_id")}
    if not docs:
        return []
    try:
        from store import db
        with db.connect() as conn:
            rows = conn.execute("""SELECT DISTINCT d.id, d.filename, s.ref_key FROM review_queue q
                                   JOIN documents d ON d.id = q.document_id
                                   LEFT JOIN segments s ON s.document_id = d.id
                                   WHERE q.status = 'HOLD' AND d.id = ANY(%s)""", (list(docs),)).fetchall()
        return [f"SRO {ref} in {fn}" if ref else fn for _, fn, ref in rows]
    except Exception:
        return []


def _log(question, request, rounds, answer, checks, timings):
    try:
        from store import db
        with db.connect() as conn:
            return conn.execute(
                "INSERT INTO qa_log (question, request, retrieved, answer, checks, timings) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                (question, Jsonb(request), Jsonb(rounds), answer, Jsonb(checks, dumps=_dumps),
                 Jsonb(timings))).fetchone()[0]
    except Exception:
        return None


def _dumps(o):
    import json
    return json.dumps(o, ensure_ascii=False, default=str)


def answer_text(question: str, **kw) -> dict:
    """Non-streaming helper (tests, evaluation)."""
    out, sources = {}, []
    for ev in ask(question, **kw):
        if ev["type"] == "done":
            out = dict(ev)
        elif ev["type"] == "sources":
            sources = ev["passages"]
    out["sources"] = sources
    return out
