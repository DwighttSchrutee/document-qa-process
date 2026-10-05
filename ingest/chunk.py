"""Structure-aware chunking (generic: no domain schema).

Tables
  one chunk per data row, rendered with its column headers ("HS Code: 8714.10.90 | CD: 25 | ...");
  table title and section path repeated in every chunk; headers carried across pages when a table
  continues; dash-indented hierarchy ("-", "--", "---") resolved so "--- Other" carries its parents.
Prose
  split on structure: headings, section markers (TABLE-n, শর্তাবলি/Conditions, পরিশিষ্ট/Appendix,
  তফসিল/Schedule, ফরম/Form), notice headers, numbered clauses (১।, 1., (১), (ক), (a), (i)).
  Whole clauses are packed up to `max_tokens`; a clause is never cut unless it alone is longer,
  then it is split at sub-clauses, then at sentences (with a one-sentence overlap).
Every chunk carries: pages, bbox per page, section path, clause number, segment (notice) dates,
language, quality status (worst status of the rows/lines it contains; HOLD/REJECTED lines are left out).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from core.textnorm import clean, digits_ascii, language, search_form
from ingest import gate as gate_mod
from ingest.links import SegmentTracker, identifier_tokens

MAX_TOKENS = 450          # XLM-R tokens; Bangla costs ~1.2 qwen3 tokens per character, so prompts stay small
MIN_TOKENS = 300
MARGIN = {"PageHeader", "PageFooter", "Picture", "Figure"}

BN_NUM = "[০-৯0-9]"
CLAUSE_PATTERNS = [   # (level, regex) — checked in order
    (1, re.compile(rf"^\s*({BN_NUM}{{1,3}})\s*[।\.]\s*(?![০-৯0-9])")),                    # ১। / 1.
    (1, re.compile(r"^\s*(?:Article|Section|Rule|ধারা|বিধি|অনুচ্ছেদ)\s+([০-৯0-9]{1,3})", re.I)),
    (2, re.compile(rf"^\s*\(({BN_NUM}{{1,3}})\)")),                                       # (১) / (1)
    (3, re.compile(r"^\s*\(([ক-হ][া-ৌ]?|[a-z]{1,2}|[ivxlc]{1,5})\)")),                   # (ক) / (a) / (iv)
]
SECTION_MARKERS = re.compile(
    r"^[\s\[(]*(?:TABLE|Table|টেবিল|সারণি)\s*[-–:]?\s*[0-9০-৯IVX]+|^[\s\[(]*(?:শ(?:র্)?তাবল[িী]|শর্তসমূহ|Conditions?)\s*[:।]?\s*$"
    r"|^[\s\[(]*(?:পরিশিষ্ট|Appendix|Annex(?:ure)?|তফসিল|Schedule|ফরম|Form|ছক)(?:\b|[-–\s:]|$)", re.I)
FORM_MARKER = re.compile(r"^[\s\[(]*(?:ফরম|Form|ছক)(?:\b|[-–\s:]|$)", re.I)
# letterhead lines at the top of a notice: kept as text, never used as section headings
LETTERHEAD = re.compile(r"গণপ্রজাতন্ত্রী|মন্ত্রণালয়|বিভাগ$|রাজস্ব\s+বোর্ড|^\(?কাস্টমস|^প্রজ্ঞাপন|^তারিখ|Government\s+of"
                        r"|Ministry|Division$|Board\s+of\s+Revenue|^NOTIFICATION|^Dated?", re.I)


@lru_cache(maxsize=1)
def _tokenizer():
    """XLM-R tokenizer (same family as bge-m3 / bge-reranker-v2-m3), loaded offline."""
    try:
        from transformers import AutoTokenizer
        from core.config import settings
        return AutoTokenizer.from_pretrained(settings.reranker_model)
    except Exception:
        return None


def n_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is None:                       # rough fallback: Bangla ~2 chars/token, English ~4
        return int(len(text) / (2 if language(text) == "bn" else 4)) + 1
    return len(tok(text, add_special_tokens=False)["input_ids"])


@dataclass
class Unit:
    """One line of prose or one table row, with its place on the page and its quality status."""
    text: str
    page: int
    bbox: list | None
    status: str
    label: str
    block: int
    index: int
    segment: int = 0
    section_path: tuple = ()


@dataclass
class Chunk:
    kind: str
    text: str
    embed_text: str
    units: list[Unit]
    section_path: list[str]
    clause_no: str | None = None
    segment: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def pages(self) -> list[int]:
        return sorted({u.page for u in self.units})

    @property
    def bbox(self) -> dict:
        out: dict[str, list] = {}
        for u in self.units:
            if not u.bbox:
                continue
            k = str(u.page)
            b = out.get(k)
            out[k] = list(u.bbox) if b is None else [min(b[0], u.bbox[0]), min(b[1], u.bbox[1]),
                                                     max(b[2], u.bbox[2]), max(b[3], u.bbox[3])]
        return out

    @property
    def status(self) -> str:
        return gate_mod.worst(*(u.status for u in self.units))


def clause_of(text: str) -> tuple[int, str] | None:
    for level, rx in CLAUSE_PATTERNS:
        m = rx.match(text)
        if m:
            return level, m.group(1)
    return None


# ------------------------------------------------------------------ tables

def _is_headerish(row: list[str]) -> bool:
    cells = [c for c in row if c.strip()]
    return bool(cells) and sum(1 for c in cells if re.search(r"\d", digits_ascii(c))) <= len(cells) // 3


def _dash_level(text: str) -> int:
    """'--- Other' -> 3. A cell that is only a dash ('-' = nil) is not a hierarchy marker."""
    m = re.match(r"^\s*([-–—]+(?:\s*[-–—])*)\s*(?=\S*[^\W\d_])", text)
    return len(re.sub(r"\s", "", m.group(1))) if m else 0


class TableState:
    """Carries headers and description hierarchy from one table (part) to the next."""

    def __init__(self):
        self.headers: list[str] | None = None
        self.n_cols = 0
        self.title = ""
        self.stack: list[tuple[int, str]] = []
        self.desc_col: int | None = None
        self.last_row: list[str] | None = None
        self.text_col: int | None = None          # main description column
        self.group: tuple[dict, str] | None = None  # (key cells, description) of the current group header row


def table_chunks(block: dict, page_no: int, gate: dict, state: TableState | None, title: str,
                 section_path: tuple, segment: int, continues: bool) -> tuple[list[Chunk], TableState]:
    rows = block["table"]["rows"]
    if not rows:
        return [], state or TableState()
    n_cols = max(len(r) for r in rows)
    bboxes = block["table"].get("row_bboxes") or [None] * len(rows)
    hdr = block["table"].get("header_rows", 0)
    if hdr == 0 and len(rows) > 1 and _is_headerish(rows[0]) and not _is_headerish(rows[1]):
        hdr = 1
    # a table part with no header row and the same column count continues the previous table
    if not (state and state.n_cols == n_cols and (continues or hdr == 0)):
        state = TableState()
        state.title = title
    if hdr:
        names = []
        for c in range(n_cols):
            parts = [rows[r][c].strip() for r in range(hdr) if c < len(rows[r]) and rows[r][c].strip()
                     and not re.fullmatch(r"\(?[0-9০-৯]{1,2}\)?", rows[r][c].strip())]   # "(2)": column numbers
            names.append(" / ".join(dict.fromkeys(parts)) or f"Column {c + 1}")
        if state.headers and [h.casefold() for h in names] == [h.casefold() for h in state.headers]:
            pass                                    # repeated header on a continuation page
        state.headers = names
    if not state.headers:
        state.headers = [f"Column {c + 1}" for c in range(n_cols)]
    state.n_cols = n_cols
    data = list(range(hdr, len(rows)))
    if state.desc_col is None:                      # column holding dash-indented descriptions
        dashed = [sum(1 for i in data if c < len(rows[i]) and _dash_level(rows[i][c]) > 0) for c in range(n_cols)]
        best = max(range(n_cols), key=lambda c: dashed[c]) if dashed else 0
        state.desc_col = best if dashed and dashed[best] >= 2 else -1

    if state.text_col is None:                       # the column with the longest texts describes the items
        avg = [sum(len(rows[i][c]) for i in data if c < len(rows[i])) / max(1, len(data)) for c in range(n_cols)]
        state.text_col = max(range(n_cols), key=lambda c: avg[c]) if avg else 0

    chunks = []
    for i in data:
        row = [clean(c) for c in rows[i]] + [""] * (n_cols - len(rows[i]))
        # merged cells: carry a group key down (e.g. heading 73.26 over codes 7326.xx.xx), only when its
        # digits are a prefix of another cell in this row — otherwise nothing is filled in
        filled_from_above = []
        for c in range(n_cols):
            prev = state.last_row[c] if state.last_row and c < len(state.last_row) else ""
            pd = re.sub(r"\D", "", digits_ascii(prev))
            if not row[c].strip() and len(pd) >= 2:
                others = [re.sub(r"\D", "", digits_ascii(x)) for k, x in enumerate(row) if k != c]
                if any(len(o) > len(pd) and o.startswith(pd) for o in others):
                    row[c] = prev
                    filled_from_above.append(state.headers[c] if c < len(state.headers) else c)
        if any(x.strip() for x in row):
            state.last_row = row
        st = gate_mod.unit_status(gate, ("row", block["id"], i)) if gate else "PASS"
        if st in ("HOLD", "REJECTED") or not any(c.strip() for c in row):
            continue
        nums = [digits_ascii(c).strip("() ") for c in row if c.strip()]
        if len(nums) >= 3 and all(x.isdigit() for x in nums) and \
                [int(x) for x in nums] == list(range(int(nums[0]), int(nums[0]) + len(nums))):
            continue
        parents = []
        dc = state.desc_col
        if dc is not None and dc >= 0:
            desc = row[dc]
            lvl = _dash_level(desc)
            filled = sum(1 for c in row if c.strip())
            if lvl == 0 and filled <= 2 and desc.strip():
                state.stack = [(0, desc.strip())]   # heading row resets the hierarchy
            else:
                while state.stack and state.stack[-1][0] >= lvl and not (lvl == 0 and state.stack[-1][0] == 0 and
                                                                           len(state.stack) == 1 and filled > 2):
                    state.stack.pop()
                parents = [d for _, d in state.stack]
                if desc.strip():
                    state.stack.append((lvl, desc.strip()))
        # group header rows: "69.07 | All HS Code | Ceramic Tiles | | " heads the unlabelled rows below it
        # ("| | 600 mm X 600 mm | m2 | 6.00"); those rows carry the header's key cells and description
        tc = state.text_col
        keys, vals = list(range(tc)), list(range(tc + 1, n_cols))
        group_from = None
        if keys and row[tc].strip():
            has_key = any(row[c].strip() for c in keys)
            has_val = any(row[c].strip() for c in vals)
            if has_key and not has_val:
                state.group = ({c: row[c] for c in keys if row[c].strip()}, row[tc])
            elif has_key:
                state.group = None
            elif state.group and has_val and not parents:
                gkeys, gdesc = state.group
                for c, v in gkeys.items():
                    row[c] = v
                row[tc] = f"{gdesc} > {row[tc]}"
                group_from = gdesc
        cells = []
        for c, v in enumerate(row):
            if not v.strip():
                continue
            if c == dc and parents:
                v = " > ".join(parents + [v])
            cells.append(f"{state.headers[c]}: {v}")
        text = " | ".join(cells)
        u = Unit(text, page_no, bboxes[i] if i < len(bboxes) else None, st, "TableRow", block["id"], i, segment,
                 section_path)
        spath = list(section_path) + ([state.title] if state.title and state.title not in section_path else [])
        ctx = " / ".join(x for x in spath if x)
        chunks.append(Chunk("table_row", text, f"{ctx}\n{text}" if ctx else text, [u], spath, None, segment,
                            {"table_title": state.title, "headers": state.headers, "row": i, "block": block["id"],
                             "parents": parents, "cells": dict(zip(state.headers, row)),
                             **({"filled_from_above": filled_from_above} if filled_from_above else {}),
                             **({"group_header": group_from} if group_from else {})}))
    return chunks, state


# ------------------------------------------------------------------ prose

def _split_sentences(u: Unit, max_tokens: int) -> list[Unit]:
    """A single over-long line -> pieces at sentence ends, each overlapping the previous by one sentence."""
    sents = [x for x in re.split(r"(?<=[।;])\s+|(?<=[a-z0-9)]\.)\s+(?=[A-Z(])", u.text) if x]
    if len(sents) < 2:
        return [u]
    out, cur = [], []
    for s in sents:
        if cur and n_tokens(" ".join(cur + [s])) > max_tokens:
            out.append(" ".join(cur))
            cur = [cur[-1]]                               # one-sentence overlap
        cur.append(s)
    out.append(" ".join(cur))
    return [Unit(t, u.page, u.bbox, u.status, u.label, u.block, u.index, u.segment, u.section_path) for t in out]


def _pack(units: list[Unit], section_path: tuple, segment: int, kind: str, max_tokens: int) -> list[Chunk]:
    """Units -> clauses -> pieces that fit -> chunks of whole clauses up to max_tokens."""
    text_of = lambda us: "\n".join(u.text for u in us)
    # 1. clauses: a line starting with a clause marker opens a clause; other lines continue the current one
    clauses: list[dict] = []
    main_no = None
    for u in units:
        c = clause_of(u.text)
        if c:
            level, no = c
            if level == 1:
                main_no = no
            label = no if level == 1 or not main_no else f"{main_no}({no})"
            clauses.append({"level": level, "no": label, "units": [u]})
        elif clauses:
            clauses[-1]["units"].append(u)
        else:
            clauses.append({"level": 0, "no": None, "units": [u]})
    # 2. a main clause owns the sub-clauses after it
    groups: list[list[dict]] = []
    for c in clauses:
        if c["level"] <= 1 or not groups:
            groups.append([c])
        else:
            groups[-1].append(c)
    # 3. pieces: whole groups; an over-long group splits at sub-clauses, then sentences.
    #    Every piece after the first of a split group carries the group's first line as context.
    pieces: list[dict] = []
    for g in groups:
        us = [u for c in g for u in c["units"]]
        nos = [c["no"] for c in g if c["no"]]
        if n_tokens(text_of(us)) <= max_tokens:
            pieces.append({"units": us, "no": g[0]["no"], "nos": nos, "cont": None})
            continue
        head = g[0]["units"][0].text[:160]
        parts, cur, cur_nos = [], [], []
        for c in g:
            for u in c["units"]:
                for v in (_split_sentences(u, max_tokens) if n_tokens(u.text) > max_tokens else [u]):
                    if cur and n_tokens(text_of(cur + [v])) > max_tokens:
                        parts.append((cur, cur_nos))
                        cur, cur_nos = [], []
                    cur.append(v)
            if c["no"]:
                cur_nos.append(c["no"])
        if cur:
            parts.append((cur, cur_nos))
        for k, (pu, pn) in enumerate(parts):
            pieces.append({"units": pu, "no": (pn or [g[0]["no"]])[0], "nos": pn, "cont": head if k else None})
    # 4. pack consecutive pieces while they fit (a continued piece always starts a new chunk)
    chunks: list[Chunk] = []
    cur: list[dict] = []
    for pc in pieces:
        if cur and (pc["cont"] or n_tokens(text_of([u for x in cur for u in x["units"]] + pc["units"])) > max_tokens):
            chunks.append(_prose_chunk(cur, section_path, segment, kind))
            cur = []
        cur.append(pc)
    if cur:
        chunks.append(_prose_chunk(cur, section_path, segment, kind))
    return chunks


def _prose_chunk(pieces: list[dict], section_path, segment, kind) -> Chunk:
    units = [u for p in pieces for u in p["units"]]
    text = "\n".join(u.text for u in units)
    head = f"(continued: {pieces[0]['cont']})\n" if pieces[0]["cont"] else ""
    ctx = " / ".join(section_path)
    nos = list(dict.fromkeys(n for p in pieces for n in p["nos"]))
    return Chunk(kind, head + text, (f"{ctx}\n" if ctx else "") + head + text, units, list(section_path),
                 pieces[0]["no"], segment, {"clauses": nos})


# ------------------------------------------------------------------ document

def chunk_document(pages: list[dict], max_tokens: int = MAX_TOKENS) -> tuple[list[Chunk], list[dict]]:
    """pages: page records in order (with 'gate'). Returns (chunks, segments)."""
    seg = SegmentTracker()
    chunks: list[Chunk] = []
    prose: list[Unit] = []
    path: list[str] = []          # section path within the current segment
    seg_title: dict[int, str] = {}
    kind = "prose"
    tstate: TableState | None = None
    last_was_table = False
    last_text = ""
    recent: list[str] = []      # last few prose lines (table titles)
    current_ref = None

    def flush(before_table: bool = False):
        nonlocal prose
        if before_table and prose and all(u.label == "Heading" for u in prose):
            prose = []                 # they are in the table rows' section path already
        if prose:
            sp = tuple(x for x in (seg_title.get(prose[0].segment), *prose[0].section_path) if x)
            chunks.extend(_pack(prose, sp, prose[0].segment, kind, max_tokens))
        prose = []

    for page in pages:
        g = page.get("gate")
        if g and g["status"] in ("HOLD", "REJECTED"):
            continue
        pn = page["page_no"]
        for b in page.get("blocks", []):
            if b.get("label") in MARGIN or b.get("kind") not in ("text", "table"):
                continue
            if b["kind"] == "table":
                flush(before_table=True)
                s_idx = seg.cur["index"]
                sp = tuple(x for x in (seg_title.get(s_idx), *path) if x)
                continues = last_was_table or (tstate is not None and not last_text)
                short = [t for t in recent[-3:] if len(t) < 200 and not LETTERHEAD.search(t)]
                title = short[-1] if short and not continues else (tstate.title if continues and tstate else
                                                                   (path[-1] if path else ""))
                tc, tstate = table_chunks(b, pn, g, tstate, title, sp, s_idx, continues)
                for row_text in (" ".join(r) for r in b["table"]["rows"][:1]):
                    seg.feed(row_text, pn)
                if kind == "form":
                    for c in tc:
                        c.kind = "form"
                chunks.extend(tc)
                last_was_table, last_text, recent = True, "", []
                continue
            lines = b.get("lines") or b.get("text", "").split("\n")
            lboxes = b.get("line_bboxes") or [None] * len(lines)
            for i, ln in enumerate(lines):
                ln = clean(ln).strip()
                if not ln:
                    continue
                before = seg.cur["index"]
                s_idx = seg.feed(ln, pn)
                if s_idx != before:                      # possible new notice: start a new chunk
                    flush()
                if seg.cur["ref_key"] and seg.cur["ref_key"] != current_ref:
                    current_ref = seg.cur["ref_key"]     # a new notice is confirmed: new section context
                    path, kind, tstate = [], "prose", None
                if seg.cur["ref_text"] and s_idx not in seg_title:
                    seg_title[s_idx] = seg.cur["ref_text"]
                st = gate_mod.unit_status(g, ("line", b["id"], i)) if g else "PASS"
                if st in ("HOLD", "REJECTED"):
                    continue
                is_heading = (b.get("label") == "SectionHeader" and len(ln) < 150 and not clause_of(ln)
                              and not LETTERHEAD.search(ln))
                if is_heading or SECTION_MARKERS.match(ln):
                    if any(u.label != "Heading" for u in prose):
                        flush()            # consecutive headings stay together with the text after them
                    level_marker = bool(SECTION_MARKERS.match(ln))
                    if level_marker:            # TABLE-n / Conditions / Appendix / Form: a new section
                        path = [ln[:120]]
                        kind = "form" if FORM_MARKER.match(ln) else "prose"
                    else:                   # plain heading: replaces earlier plain headings, nests under a marker
                        path = [x for x in path if SECTION_MARKERS.match(x)][-1:] + [ln[:120]]
                    last_text = ln
                    recent.append(ln)
                    last_was_table = False
                    # headings are kept as their own small unit too, so a heading-only question finds them
                    prose.append(Unit(ln, pn, lboxes[i] if i < len(lboxes) else b.get("bbox"), st, "Heading",
                                      b["id"], i, s_idx, tuple(path)))
                    continue
                prose.append(Unit(ln, pn, lboxes[i] if i < len(lboxes) else b.get("bbox"), st, b.get("label") or "Text",
                                  b["id"], i, s_idx, tuple(path)))
                recent.append(ln)
                last_text = ln
                last_was_table = False
    flush()
    segments = seg.finish()
    by_idx = {s["index"]: s for s in segments}
    for c in chunks:
        c.segment = seg.remap.get(c.segment, c.segment)
        s = by_idx.get(c.segment, {})
        if s.get("ref_text") and (not c.section_path or c.section_path[0] != s["ref_text"]):
            c.section_path = [s["ref_text"]] + c.section_path
            c.embed_text = s["ref_text"] + " / " + c.embed_text
        c.metadata.update({k: s.get(k) for k in ("ref_key", "issued", "effective_from", "effective_to")})
    return chunks, segments


def chunk_record(c: Chunk) -> dict:
    """Plain dict for storage."""
    return {"kind": c.kind, "text": c.text, "embed_text": c.embed_text, "text_search": search_form(c.text),
            "pages": c.pages, "bbox": c.bbox, "section_path": c.section_path, "clause_no": c.clause_no,
            "lang": language(c.text), "status": c.status, "segment": c.segment, "metadata": c.metadata,
            "identifiers": identifier_tokens(c.text), "n_tokens": n_tokens(c.embed_text)}
