"""Hybrid retrieval.

  1. exact identifier match (codes, notice references, dates; any digit form / separator)
  2. vector search (pgvector cosine, HNSW) for every query rewrite
  3. keyword search: PostgreSQL full text ('simple') + trigram word similarity on the normalised text
  4. reciprocal rank fusion; exact identifier hits always rank above fuzzy hits
  5. amendments: chunks of notices that amend / rescind a retrieved notice are always added
  6. re-ranker (cross-encoder) picks the final context within the token budget

Every stage can be switched off for the ablation (Options).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from core.config import settings
from core.textnorm import digits_ascii, norm_identifier, search_form
from ingest.links import find_dates, ref_key
from store import db, embed

RRF_K = 60


@dataclass
class Options:
    candidates: int = 50
    per_list: int = 30
    top_k: int = 6
    context_tokens: int = 2500           # in answering-model (qwen3) tokens, estimated by llm_tokens()
    vector: bool = True
    keyword: bool = True
    identifiers: bool = True
    amendments: bool = True
    rerank: bool = True
    embed_model: str | None = None        # None = settings.embed_model (column chunks.embedding)
    min_rerank_keep: int = 2


@dataclass
class Hit:
    id: int
    score: float = 0.0
    sources: dict = field(default_factory=dict)     # list name -> rank
    exact: str | None = None                        # identifier kind when an exact identifier matched
    row: dict | None = None
    rerank: float | None = None
    added_for: str | None = None                    # e.g. "amends 169/2024"


NOTICE_GROUP = re.compile(
    r"(?:S\.?\s?R\.?\s?O\.?s?|এস\.?\s?আর\.?\s?ও\.?)\s*(?:নং|No\.?|Nos\.?)?[\s:\-–]*"
    r"([0-9০-৯]{1,4}(?:\s*/\s*[0-9০-৯]{4})?(?:\s*(?:-|–|to|and|or|,|ও|এবং|থেকে|হইতে)\s*[0-9০-৯]{1,4}"
    r"(?:\s*/\s*[0-9০-৯]{4})?)*)", re.I)


def notice_keys(question: str) -> dict[str, str]:
    """'SRO 213/2024 ... SRO 210-212' -> REF:213/2024, REF:210/2024, REF:211/2024, REF:212/2024.
    A number without a year takes the year of the other notices named in the question (or any year)."""
    t = digits_ascii(question)
    groups = [m.group(1) for m in NOTICE_GROUP.finditer(t)]
    years = re.findall(r"\b\d{1,4}\s*/\s*(\d{4})\b", " ".join(groups))
    default_year = max(set(years), key=years.count) if years else None
    keys = {}
    for g in groups:
        parts = re.split(r"\s*(,|and|or|ও|এবং)\s*", g)
        for part in parts:
            m = re.fullmatch(r"(\d{1,4})(?:\s*/\s*(\d{4}))?(?:\s*(?:-|–|to|থেকে|হইতে)\s*(\d{1,4})(?:\s*/\s*(\d{4}))?)?",
                             part.strip())
            if not m:
                continue
            a, ya, b, yb = m.groups()
            year = ya or yb or default_year
            nums = range(int(a), int(b) + 1) if b and 0 < int(b) - int(a) <= 20 else [int(a)] + ([int(b)] if b else [])
            for n in nums:
                keys[f"REF:{n}/{year}" if year else f"REFNUM:{n}"] = "ref"
    return keys


def identifier_keys(ids: list[str]) -> dict[str, str]:
    """User identifiers -> normalised keys in the identifiers table (key -> kind)."""
    keys = {}
    for x in ids:
        rk = ref_key(re.sub(r"^\D*", "", digits_ascii(x)))
        if rk and "/" in digits_ascii(x):
            keys["REF:" + rk] = "ref"
            continue
        for _, d in find_dates(x):
            keys["DATE:" + d.isoformat()] = "date"
        if re.fullmatch(r"\s*[0-9০-৯]{1,4}\s*[-–]\s*[0-9০-৯]{1,4}\s*", x):
            continue                                   # "210-212": a range of notice numbers, not a code
        n = norm_identifier(x)
        n = re.sub(r"^(HS|HSCODE|SRO|NO)", "", n)
        if re.fullmatch(r"\d{4,10}", n):
            keys[n] = "code" if len(n) >= 6 else "prefix"
    return keys


def _ids_search(conn, keys: dict[str, str], limit: int):
    if not keys:
        return []
    nums = [k.split(":")[1] for k in keys if k.startswith("REFNUM:")]
    if nums:                                           # notice number without a year: any year
        for (k,) in conn.execute("SELECT DISTINCT normalised FROM identifiers WHERE normalised LIKE ANY(%s)",
                                 ([f"REF:{n}/%" for n in nums],)).fetchall():
            keys = {**keys, k: "ref"}
        keys = {k: v for k, v in keys.items() if not k.startswith("REFNUM:")}
    rows = conn.execute(
        """SELECT i.chunk_id, i.kind, i.normalised FROM identifiers i JOIN chunks c ON c.id = i.chunk_id
           WHERE i.normalised = ANY(%s) ORDER BY (i.kind = 'code') DESC, (i.kind = 'ref') DESC LIMIT %s""",
        (list(keys), limit * 4)).fetchall()
    out, seen = [], set()
    for cid, kind, n in rows:
        if cid not in seen:
            seen.add(cid)
            out.append((cid, kind))
    return out[:limit * 2]


# main clause numbers only: "উপ-বিধি ১" / "sub-rule 1" are sub-clauses and would match rule 1 of every document
CLAUSE_Q = re.compile(r"(?<!উপ-)(?<!উপ)(?<!sub-)(?<!sub)(?:বিধি|ধারা|অনুচ্ছেদ|rule|section|clause|article|para(?:graph)?)"
                      r"\s*[-–]?\s*([০-৯0-9]{1,3})", re.I)


def clause_numbers(question: str) -> list[str]:
    """'বিধি ১২', 'rule 12', 'section 7' -> ['12', '7'] (ASCII)."""
    return list(dict.fromkeys(digits_ascii(m.group(1)) for m in CLAUSE_Q.finditer(question)))


def _clauses(conn, nums: list[str], vec, limit: int):
    """Chunks whose main clause number is one of nums, most similar to the question first."""
    if not nums:
        return []
    return [r[0] for r in conn.execute(
        """SELECT id FROM chunks
           WHERE translate(split_part(coalesce(clause_no, ''), '(', 1), '০১২৩৪৫৬৭৮৯', '0123456789') = ANY(%s)
              OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(coalesce(metadata->'clauses', '[]'::jsonb)) x
                         WHERE translate(split_part(x, '(', 1), '০১২৩৪৫৬৭৮৯', '0123456789') = ANY(%s))
           ORDER BY embedding <=> %s::vector LIMIT %s""", (nums, nums, vec, limit)).fetchall()]


def _vector(conn, vec, limit: int, model: str | None):
    exact = settings.vector_search == "exact"
    conn.execute("SET LOCAL enable_indexscan = off" if exact else "SET LOCAL hnsw.ef_search = 1000")
    try:
        if model and model != settings.embed_model:
            return [r[0] for r in conn.execute(
                "SELECT chunk_id FROM chunk_vectors WHERE model = %s ORDER BY embedding <=> %s::vector LIMIT %s",
                (model, vec, limit)).fetchall()]
        return [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                                           (vec, limit)).fetchall()]
    finally:
        if exact:
            conn.execute("RESET enable_indexscan")


def _keyword(conn, query: str, limit: int):
    q = search_form(query)
    # split on spaces and punctuation only: Python's \w excludes Bangla vowel signs and virama ("ইউডি" -> "ইউড")
    words = [w.strip(".-/") for w in re.split(r"[\s,;:!?()\[\]{}\"'“”‘’।|]+", q)]
    words = [w for w in words if len(w) > 1]
    if not words:
        return []
    tsq = " | ".join("'" + w.replace("'", "") + "'" for w in words[:20])
    fts = [r[0] for r in conn.execute(
        """SELECT id FROM chunks WHERE tsv @@ to_tsquery('simple', %s)
           ORDER BY ts_rank_cd(tsv, to_tsquery('simple', %s)) DESC LIMIT %s""", (tsq, tsq, limit)).fetchall()]
    conn.execute("SET LOCAL pg_trgm.word_similarity_threshold = 0.45")
    trg = [r[0] for r in conn.execute(
        """SELECT id FROM chunks WHERE %s <%% text_search
           ORDER BY word_similarity(%s, text_search) DESC LIMIT %s""", (q, q, limit)).fetchall()]
    return fts, trg


def _rrf(lists: dict[str, list[int]], hits: dict[int, Hit]):
    for name, ids in lists.items():
        for rank, cid in enumerate(ids):
            h = hits.setdefault(cid, Hit(cid))
            h.sources[name] = rank
            h.score += 1.0 / (RRF_K + rank + 1)


def _load_rows(conn, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    rows = conn.execute(
        """SELECT c.id, c.document_id, d.filename, c.segment_id, s.ref_key, s.issued, s.effective_from, s.effective_to,
                  c.seq, c.pages, c.bbox, c.section_path, c.clause_no, c.kind, c.text, c.lang, c.status, c.n_tokens,
                  c.metadata, d.sha256
           FROM chunks c JOIN documents d ON d.id = c.document_id LEFT JOIN segments s ON s.id = c.segment_id
           WHERE c.id = ANY(%s)""", (ids,)).fetchall()
    cols = ["id", "document_id", "filename", "segment_id", "ref_key", "issued", "effective_from", "effective_to", "seq",
            "pages", "bbox", "section_path", "clause_no", "kind", "text", "lang", "status", "n_tokens", "metadata",
            "sha256"]
    out = {}
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("issued", "effective_from", "effective_to"):
            d[k] = d[k].isoformat() if d[k] else None
        out[d["id"]] = d
    return out


def _amendments(conn, hits: dict[int, Hit], per_link: int = 3) -> dict[int, str]:
    """Chunks of notices that amend/rescind a retrieved notice, preferring chunks that cite it."""
    seg_ids = list({h.row["segment_id"] for h in hits.values() if h.row and h.row.get("segment_id")})
    if not seg_ids:
        return {}
    links = conn.execute(
        """SELECT l.from_segment, l.kind, l.to_reference FROM doc_links l
           WHERE l.to_segment = ANY(%s) AND l.kind IN ('amends', 'rescinds')""", (seg_ids,)).fetchall()
    added = {}
    for from_seg, kind, ref in links:
        rows = conn.execute(
            """SELECT c.id FROM chunks c LEFT JOIN identifiers i ON i.chunk_id = c.id AND i.normalised = %s
               WHERE c.segment_id = %s ORDER BY (i.chunk_id IS NOT NULL) DESC, c.seq LIMIT %s""",
            ("REF:" + ref, from_seg, per_link)).fetchall()
        for (cid,) in rows:
            added.setdefault(cid, f"{kind} {ref}")
    return added


def retrieve(question: str, understood: dict | None = None, opts: Options | None = None, conn=None) -> dict:
    """Returns {'hits': [Hit...] (final context, best first), 'candidates': [...], 'timings': {...}}."""
    opts = opts or Options()
    t0 = time.time()
    understood = understood or {"queries": [question], "identifiers": [], "language": None}
    queries = list(dict.fromkeys([question] + understood.get("queries", [])))
    timings = {}
    own = conn is None
    ctx = db.connect() if own else None
    conn = ctx.__enter__() if own else conn
    try:
        hits: dict[int, Hit] = {}
        lists: dict[str, list[int]] = {}
        exact: dict[int, str] = {}
        if opts.identifiers:
            keys = {**identifier_keys(understood.get("identifiers", [])), **notice_keys(question)}
            for cid, kind in _ids_search(conn, keys, opts.per_list):
                exact.setdefault(cid, kind)
            lists["ids"] = list(exact)
        timings["ids"] = round(time.time() - t0, 3)
        if opts.vector:
            for i, q in enumerate(queries):
                vec = embed.embed_query(q, opts.embed_model)
                lists[f"vec{i}"] = _vector(conn, vec, opts.per_list, opts.embed_model)
                if i == 0 and opts.identifiers:
                    lists["clause"] = _clauses(conn, clause_numbers(question), vec, opts.per_list)
        timings["vector"] = round(time.time() - t0, 3)
        if opts.keyword:
            for i, q in enumerate(queries):
                r = _keyword(conn, q, opts.per_list)
                if r:
                    lists[f"fts{i}"], lists[f"trgm{i}"] = r
        timings["keyword"] = round(time.time() - t0, 3)
        _rrf(lists, hits)
        for cid, kind in exact.items():
            hits[cid].exact = kind
        # exact code/reference hits first (by RRF within them), then the rest
        rank_key = lambda h: (0 if h.exact in ("code", "ref") else 1 if h.exact else 2, -h.score)
        cands = sorted(hits.values(), key=rank_key)[:opts.candidates]
        rows = _load_rows(conn, [h.id for h in cands])
        for h in cands:
            h.row = rows.get(h.id)
        cands = [h for h in cands if h.row]
        if opts.amendments:
            extra = _amendments(conn, {h.id: h for h in cands})
            new = [cid for cid in extra if cid not in {h.id for h in cands}]
            rows = _load_rows(conn, new)
            for cid in new:
                if cid in rows:
                    cands.append(Hit(cid, 0.0, {}, None, rows[cid], None, extra[cid]))
            for h in cands:
                if h.id in extra and not h.added_for:
                    h.added_for = extra[h.id]
        timings["candidates"] = round(time.time() - t0, 3)
        cards = notice_cards(conn, notice_keys(question)) if opts.identifiers else []
        # the cards replace the notices' first pages, which were only matched for naming the notice
        card_segs = {c.row["segment_id"] for c in cards}
        cands_for_ctx = [h for h in cands if not (h.exact == "ref" and h.row.get("segment_id") in card_segs
                                                  and h.row.get("seq", 1) == 0)]
        final = cards + select_context(question, cands_for_ctx,
                                       Options(**{**opts.__dict__,
                                                  "context_tokens": max(800, opts.context_tokens - 120 * len(cards))}))
        final = add_table_context(conn, final)
        timings["rerank"] = round(time.time() - t0, 3)
    finally:
        if own:
            ctx.__exit__(None, None, None)
    return {"hits": final, "candidates": cands, "timings": timings, "queries": queries}


def select_context(question: str, cands: list[Hit], opts: Options) -> list[Hit]:
    """Re-rank and keep the best within top_k and the token budget. Exact code/reference hits and
    amendment chunks of kept notices are never dropped in favour of fuzzy hits."""
    if not cands:
        return []
    if opts.rerank:
        from retrieve import rerank
        # score the best fused candidates (exact hits and amendment chunks always included)
        pool = cands[:settings.rerank_candidates] + [h for h in cands[settings.rerank_candidates:]
                                                       if h.exact or h.added_for]
        scores = rerank.score(question, [passage_text(h.row) for h in pool])
        for h, s in zip(pool, scores):
            h.rerank = s
        order = sorted(pool, key=lambda h: -(h.rerank if h.rerank is not None else -99))
    else:
        order = list(cands)
    must = [h for h in cands if h.exact in ("code", "ref")][: max(2, opts.top_k // 2)]
    picked, used = [], 0
    for h in must + [h for h in order if h not in must]:
        if len(picked) >= opts.top_k:
            break
        if h in picked:
            continue
        n = llm_tokens(passage_text(h.row))
        if used + n > opts.context_tokens and len(picked) >= opts.min_rerank_keep:
            continue
        picked.append(h)
        used += n
    # amendments of notices that are really relevant (exact hit or a clear re-ranker score), not of every passage
    segs = {h.row.get("ref_key") for h in picked
            if h.exact or h.rerank is None or h.rerank >= 0.2}
    # one chunk per amending notice (the first one found, which cites the amended notice), within the budget
    seen_amending = set()
    for h in cands:
        if h.added_for and h.added_for.split()[-1] in segs and h not in picked:
            key = h.row.get("segment_id") or h.row.get("document_id")
            n = llm_tokens(passage_text(h.row))
            if key in seen_amending or used + n > opts.context_tokens * 1.2:
                continue
            seen_amending.add(key)
            picked.append(h)
            used += n
    return picked


def llm_tokens(text: str) -> int:
    """Estimate of qwen3 tokens, measured on this corpus: ~1.18 tokens per Bangla character, ~0.39 per other."""
    bn = sum(1 for ch in text if "\u0980" <= ch <= "\u09ff")
    return int(bn * 1.18 + (len(text) - bn) * 0.39) + 20


def _short(sentence: str, keep: int = 160) -> str:
    """The part of a long sentence that carries the date ('... ৬ জুন, ২০২৪ খ্রিস্টাব্দ তারিখে কার্যকর হইবে।')."""
    t = sentence.strip()
    return t if len(t) <= keep else "…" + t[-keep:]


def notice_cards(conn, keys: dict[str, str]) -> list[Hit]:
    """One small passage per notice named in the question: reference, title, dates and the sentences that set
    them, quoted from the document. Several notices fit in the context where their full first pages would not."""
    refs = [k[4:] for k in keys if k.startswith("REF:")]
    nums = [k.split(":")[1] + "/%" for k in keys if k.startswith("REFNUM:")]
    if not refs and not nums:
        return []
    rows = conn.execute(
        """SELECT s.id, s.ref_key, s.ref_text, s.title, s.issued, s.effective_from, s.effective_to, s.effective_text,
                  s.expiry_text, s.first_page, d.id, d.filename, d.sha256
           FROM segments s JOIN documents d ON d.id = s.document_id
           WHERE s.ref_key = ANY(%s) OR s.ref_key LIKE ANY(%s) ORDER BY s.ref_key""", (refs, nums or [""])).fetchall()
    out = []
    for (sid, ref, ref_text, title, issued, eff, until, eff_text, exp_text, page, doc_id, fn, sha) in rows:
        lines = [f"Notice {ref_text or 'SRO ' + ref}" + (f": {title}" if title else "")]
        if issued:
            lines.append(f"Issued: {issued.isoformat()}")
        if eff or eff_text:
            lines.append(f"Effective from: {eff.isoformat() if eff else 'see text'}"
                         + (f' — the notice says: "{_short(eff_text)}"' if eff_text else ""))
        else:
            lines.append("Effective date: not stated in a form the system recognised")
        if until or exp_text:
            lines.append(f"Effective until: {until.isoformat() if until else 'see text'}"
                         + (f' — the notice says: "{_short(exp_text)}"' if exp_text else ""))
        text = "\n".join(lines)
        row = {"id": -sid, "document_id": doc_id, "filename": fn, "segment_id": sid, "ref_key": ref,
               "issued": issued.isoformat() if issued else None, "effective_from": eff.isoformat() if eff else None,
               "effective_to": until.isoformat() if until else None, "seq": 0, "pages": [page] if page else [],
               "bbox": {}, "section_path": [], "clause_no": None, "kind": "notice", "text": text, "lang": "mixed",
               "status": "PASS", "n_tokens": 100, "metadata": {"card": True}, "sha256": sha}
        out.append(Hit(-sid, 0.0, {}, "ref", row, None, "notice card"))
    return out


def add_table_context(conn, picked: list[Hit], per_table: int = 1) -> list[Hit]:
    """A table row alone does not say what applies to it ("TABLE-1: 8448.33.00 Ring Travellers"); the notice's
    clause that refers to the table does ("goods in TABLE-1 are exempt from CD above 1% ..."). For every table in
    the context, add the earliest prose chunk of the same notice that names the table's title."""
    have = {h.id for h in picked}
    seen = set()
    extra = []
    for h in picked:
        r = h.row
        if r.get("kind") != "table_row" or not r.get("segment_id"):
            continue
        title = ((r.get("metadata") or {}).get("table_title") or (r.get("section_path") or [""])[-1]).strip()
        if len(title) < 3 or (r["segment_id"], title) in seen:
            continue
        seen.add((r["segment_id"], title))
        pat = re.escape(search_form(title)).replace(r"\ ", r"\s*").replace(r"\-", r"\s*[-–]?\s*")
        pat += "(?![0-9০-৯])"                      # TABLE-1 is not TABLE-10
        rows = conn.execute(
            """SELECT id FROM chunks WHERE segment_id = %s AND kind = 'prose' AND text_search ~ %s
               ORDER BY seq LIMIT %s""", (r["segment_id"], pat, per_table)).fetchall()
        new = [cid for (cid,) in rows if cid not in have]
        loaded = _load_rows(conn, new)
        for cid in new:
            if cid in loaded:
                have.add(cid)
                extra.append(Hit(cid, 0.0, {}, None, loaded[cid], None, f"explains {title}"))
    return picked + extra


def passage_text(row: dict) -> str:
    path = " / ".join(row.get("section_path") or [])
    return f"{path}\n{row['text']}" if path else row["text"]


def retrieve_wide(question: str, understood: dict, opts: Options | None = None) -> dict:
    """Second round (before ever answering "not found"): more candidates, every rewrite, identifier-only
    search, neighbouring chunks of the best hits, and notices linked in either direction."""
    base = opts or Options()
    wide = Options(**{**base.__dict__, "candidates": 100, "per_list": 60})
    with db.connect() as conn:
        r = retrieve(question, understood, Options(**{**wide.__dict__, "rerank": False}), conn=conn)
        cands = r["candidates"]
        have = {h.id for h in cands}
        # identifier-only search with every identifier-looking token of all rewrites
        from answer.understand import regex_identifiers
        ids = set(understood.get("identifiers", []))
        for q in r["queries"]:
            ids.update(regex_identifiers(q))
        extra: dict[int, str] = {}
        for cid, kind in _ids_search(conn, identifier_keys(list(ids)), 60):
            extra.setdefault(cid, f"identifier ({kind})")
        # neighbours of the best fused candidates (same document, previous / next chunk)
        top = cands[:10]
        if top:
            for cid, why in conn.execute(
                    """SELECT n.id, 'next to ' || c.id FROM chunks c JOIN chunks n
                       ON n.document_id = c.document_id AND n.seq IN (c.seq - 1, c.seq + 1)
                       WHERE c.id = ANY(%s)""", ([h.id for h in top],)).fetchall():
                extra.setdefault(cid, why)
        # notices linked to / from the candidates' notices
        segs = list({h.row["segment_id"] for h in top if h.row and h.row.get("segment_id")})
        if segs:
            for cid, kind, ref in conn.execute(
                    """SELECT c.id, l.kind, l.to_reference FROM doc_links l
                       JOIN chunks c ON c.segment_id IN (l.from_segment, l.to_segment)
                       WHERE (l.to_segment = ANY(%s) OR l.from_segment = ANY(%s)) AND c.segment_id <> ALL(%s)
                       ORDER BY c.seq LIMIT 40""", (segs, segs, segs)).fetchall():
                extra.setdefault(cid, f"linked notice ({kind} {ref})")
        new = [cid for cid in extra if cid not in have]
        rows = _load_rows(conn, new)
        for cid in new:
            if cid in rows:
                cands.append(Hit(cid, 0.0, {}, None, rows[cid], None, None))
        final = select_context(question, cands, Options(**{**base.__dict__, "rerank": True}))
        final = add_table_context(conn, final)
    return {"hits": final, "candidates": cands, "queries": r["queries"], "timings": r["timings"]}
