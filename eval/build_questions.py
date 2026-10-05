"""Build evaluation questions from the indexed documents (all local).

table_lookup   generated from table rows whose values are confirmed in the PDF text layer (verified=True)
rules / conditions / cross_lingual
               drafted by qwen3:8b from one passage; must_contain values must occur in the passage;
               verified=False until a person checks them (eval/review_questions.py)
unanswerable   hand-written list in eval/unanswerable.jsonl (verified when a person confirms)

    python -m eval.build_questions --split dev --n-table 24 --n-prose 24 --seed 1
Writes eval/questions_<split>.jsonl (appends new ids, keeps existing items and their verification).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pymupdf

from core.textnorm import BN_DIGITS, digits_ascii
from store import db

HERE = Path(__file__).parent
TO_BN = str.maketrans("0123456789", BN_DIGITS)


def _qid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def textlayer_values(path: str, page: int) -> set[str]:
    words = pymupdf.open(path)[page - 1].get_text("words")
    return {digits_ascii(w[4]).strip(",;:()") for w in words}


# ------------------------------------------------------------------ table lookups

CODE_COLS = re.compile(r"code|hs|heading|শিরোনাম|কোড", re.I)
DESC_COLS = re.compile(r"desc|বর্ণনা|goods|পণ্য", re.I)


def table_questions(conn, n: int, rng: random.Random) -> list[dict]:
    rows = conn.execute(
        """SELECT c.id, d.filename, d.sha256, c.pages[1], c.text, c.metadata, s.ref_text, s.ref_key
           FROM chunks c JOIN documents d ON d.id = c.document_id LEFT JOIN segments s ON s.id = c.segment_id
           WHERE c.kind = 'table_row' AND c.status = 'PASS' ORDER BY c.id""").fetchall()
    from core.config import settings
    meta_path = {}
    out = []
    rng.shuffle(rows)
    per_doc: dict[str, int] = {}
    for cid, fn, sha, page, text, md, ref_text, ref_key_ in rows:
        if len(out) >= n:
            break
        cells = md.get("cells") or {}
        code_col = next((h for h in cells if CODE_COLS.search(h) and re.search(r"\d{4}\.?\d{2}", cells[h] or "")
                         and not re.search(r"heading|শিরোনাম", h, re.I)), None)
        desc_col = next((h for h in cells if DESC_COLS.search(h) and re.search(r"[A-Za-zঀ-৿]{3}", cells[h] or "")), None)
        value_cols = [h for h in cells if h not in (code_col, desc_col) and re.fullmatch(r"[\d.,%]+", (cells[h] or "").strip())
                      and not CODE_COLS.search(h)]
        if not code_col or not desc_col or md.get("filled_from_above"):
            continue
        code = cells[code_col].split()[0]
        if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}|\d{8}", code):
            continue
        if per_doc.get(fn, 0) >= max(3, n // 2):
            continue
        if sha not in meta_path:
            meta_path[sha] = json.loads((settings.data_dir / "docs" / sha / "meta.json").read_text("utf-8"))["path"]
        tl = textlayer_values(meta_path[sha], page)
        desc = re.sub(r"\s+", " ", cells[desc_col]).strip(" :")
        notice = f" under SRO {ref_key_}" if ref_key_ else ""
        style = rng.choice(["dots", "plain", "bangla_digits", "name"])
        code_q = {"dots": code, "plain": code.replace(".", ""), "bangla_digits": code.translate(TO_BN),
                  "name": None}[style]
        if value_cols:
            vc = rng.choice(value_cols)
            val = cells[vc].strip()
            if digits_ascii(val) not in tl or digits_ascii(code) not in tl:
                continue                                           # not confirmed by the text layer
            lang = "bn" if style == "bangla_digits" else "en"
            if lang == "bn":
                q = f"এইচএস কোড {code_q} ({desc}) এর {vc} কত?"
            elif style == "name":
                q = f"What is the {vc} for {desc}{notice}?"
            else:
                q = f"What is the {vc} for HS code {code_q}{notice}?"
            must = [val]
            expected = f"{vc}: {val}"
        else:
            if digits_ascii(code) not in tl:
                continue
            lang = "en"
            q = (f"Is {desc} (HS {code_q}) included in {ref_text or fn}?" if code_q
                 else f"Which HS code is given for '{desc}' in {fn}?")
            must = [code] if not code_q else []
            expected = f"Yes: {code} {desc} is listed" if code_q else code
            if not must:
                must = [desc.split()[0]]
        out.append({"id": "tbl-" + _qid(cid, style), "question": q, "language": lang, "category": "table_lookup",
                    "expected_answer": expected, "must_contain": must, "must_not_contain": [],
                    "gold_sources": [{"file": fn, "page": page}], "gold_chunks": [cid], "answerable": True,
                    "verified": True, "verified_by": "text layer", "phrasing": style})
        per_doc[fn] = per_doc.get(fn, 0) + 1
    return out


# ------------------------------------------------------------------ prose (drafted by the local model)

DRAFT_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"}, "answer": {"type": "string"},
    "facts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}},
    "required": ["question", "answer", "facts"]}

DRAFT_PROMPT = """Write ONE question that a customs officer or importer might ask, which is answered by the passage
below. Write the question in {lang}. The question must be self-contained: name the notification or rules it is
about ({doc_name}) and the subject of the rule — never write "this notification" or "this clause". It must be
answerable from this passage alone and must not copy the passage word for word. Keep it under 30 words.
Return JSON: "question"; "answer": a one-sentence answer; "facts": 1 to 3 short exact substrings copied from the
passage that any correct answer must contain (prefer numbers, periods of time, amounts, names of forms or authorities).

Passage ({where}):
{text}"""


def prose_questions(conn, n: int, rng: random.Random, cross_share: float = 0.4) -> list[dict]:
    from answer import llm
    rows = conn.execute(
        """SELECT c.id, d.filename, c.pages, c.text, c.clause_no, c.section_path, c.lang, c.kind
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.kind = 'prose' AND c.status = 'PASS' AND c.n_tokens BETWEEN 120 AND 820
             AND c.text !~ '^গণপ্রজাতন্ত্রী'""").fetchall()
    rng.shuffle(rows)
    out = []
    for cid, fn, pages, text, clause, path, lang, kind in rows:
        if len(out) >= n:
            break
        cross = rng.random() < cross_share
        q_lang = ("English" if lang == "bn" else "Bangla") if cross else ("Bangla" if lang == "bn" else "English")
        try:
            raw = llm.chat([{"role": "user", "content": DRAFT_PROMPT.format(
                lang=q_lang, where=f"{fn}, clause {clause or '-'}", doc_name=(path or [fn])[0],
                text=text[:3000])}],
                fmt=DRAFT_SCHEMA, num_ctx=6144, num_predict=1200, timeout=600)
            d = json.loads(raw)
        except Exception as e:
            print("draft failed:", e)
            continue
        facts = [f.strip() for f in d.get("facts", []) if f.strip() and digits_ascii(f.strip()) in digits_ascii(text)]
        if not facts or not d.get("question"):
            continue
        cat = "cross_lingual" if cross else ("conditions" if re.search(r"শর্ত|condition", " ".join(path or []), re.I)
                                             else "rules_clause")
        out.append({"id": "prs-" + _qid(cid, q_lang), "question": d["question"].strip(),
                    "language": "bn" if q_lang == "Bangla" else "en", "category": cat,
                    "expected_answer": d.get("answer", "").strip(), "must_contain": facts, "must_not_contain": [],
                    "gold_sources": [{"file": fn, "page": p} for p in pages], "gold_chunks": [cid],
                    "answerable": True, "verified": False, "verified_by": None, "drafted_by": "qwen3:8b"})
        print(f"  drafted {len(out)}/{n}: {out[-1]['question'][:90]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-table", type=int, default=24)
    ap.add_argument("--n-prose", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    path = HERE / f"questions_{args.split}.jsonl"
    existing = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()] if path.exists() else []
    have = {x["id"] for x in existing}
    with db.connect() as conn:
        new = table_questions(conn, args.n_table, rng)
        if args.n_prose:
            new += prose_questions(conn, args.n_prose, rng)
    un = HERE / "unanswerable.jsonl"
    if un.exists():
        new += [json.loads(x) for x in un.read_text("utf-8").splitlines() if x.strip() and
                json.loads(x).get("split", "dev") == args.split]
    added = [x for x in new if x["id"] not in have]
    for x in added:
        x["split"] = args.split
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in existing + added), "utf-8")
    print(f"{len(added)} new questions -> {path} ({len(existing) + len(added)} total)")


if __name__ == "__main__":
    main()
