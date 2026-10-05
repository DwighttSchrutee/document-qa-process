"""Merge Surya OCR with a reference reader, block by block.

Surya reads Bangla well but can invent numbers; the PDF text layer has garbled Bangla but exact
digits, codes and English. So per block:

  1. split both readings into runs: Bangla-letter runs vs non-Bangla runs (numbers, Latin, symbols);
  2. align the run sequences (difflib, no autojunk). Bangla runs match any Bangla run; non-Bangla
     runs match on their digit-normalised value;
  3. non-Bangla run that differs        -> replaced by the reference   (CORRECTION)
     reference value Surya missed        -> inserted                    (RESTORED)
     Surya value with no counterpart     -> kept, flagged               (UNCONFIRMED)
  4. a RESTORED and an UNCONFIRMED with the same value anywhere on the page cancel: that is a
     reordering, not an error.

mode="textlayer": reference = PDF text layer (trusted).
mode="vote":      reference = second OCR engine (Tesseract) on scanned pages. Nothing is copied
                  from it; every disagreement or Surya-only value becomes UNCONFIRMED.

Bangla text is always Surya's. Legacy ANSI-font words in the text layer are never copied, except
digits, which are converted to Bangla digits.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from core.textnorm import BN_DIGITS, BN_LETTER, digits_ascii
from ingest.blocks import html_lines, html_table, is_table

_RUN_RE = re.compile(f"([{BN_LETTER}]+)|(\\s+)|([^\\s{BN_LETTER}]+)")
_SIG_RE = re.compile(r"[0-9০-৯A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9০-৯]")
_EDGE_PUNCT = re.compile(r"^[^0-9A-Za-z০-৯]+|[^0-9A-Za-z০-৯%]+$")
SKIP_LABELS = {"Picture", "Figure", "Image", "Diagram", "Blank-Page"}


@dataclass(eq=False)
class Run:
    text: str
    kind: str                 # bn | num | lat | punct | space
    loc: tuple                # where it lives in the block: ("line", i) or ("cell", r, c)
    bbox: tuple | None = None  # reference runs only, PDF points
    flag: str | None = None    # UNCONFIRMED on OCR runs
    restored: bool = False
    cancelled: bool = False
    derived: bool = False      # copy of a spanned cell: kept for structure, never compared

    @property
    def core(self) -> str:
        return _EDGE_PUNCT.sub("", self.text) or self.text

    @property
    def key(self) -> str:
        if self.kind == "bn":
            return "§bn"
        if self.kind == "dash":
            return "-" * len(self.text)
        return digits_ascii(self.core)


def split_runs(text: str, loc: tuple, bbox=None) -> list[Run]:
    runs = []
    for m in _RUN_RE.finditer(text):
        bn, sp, other = m.groups()
        if bn:
            runs.append(Run(bn, "bn", loc, bbox))
        elif sp:
            runs.append(Run(sp, "space", loc, bbox))
        elif _SIG_RE.search(other):
            runs.append(Run(other, "num" if _DIGIT_RE.search(other) else "lat", loc, bbox))
        elif re.fullmatch(r"[-–—]+", other):
            runs.append(Run(other, "dash", loc, bbox))     # "---": hierarchy level in tables, or a nil value
        else:
            runs.append(Run(other, "punct", loc, bbox))
    return runs


SIG_KINDS = ("bn", "num", "lat", "dash")


def _sig(runs):
    return [r for r in runs if r.kind in SIG_KINDS and not r.derived]


# ------------------------------------------------------------------ block content

def block_runs(block: dict) -> tuple[list[Run], dict]:
    """Runs of one Surya block in reading order plus the structure needed to rebuild it."""
    runs: list[Run] = []
    if is_table(block):
        tab = html_table(block.get("html", ""))
        copies = {tuple(x) for x in tab["copies"]}
        for r, row in enumerate(tab["rows"]):
            for c, cell in enumerate(row):
                cell_runs = split_runs(cell, ("cell", r, c))
                for x in cell_runs:
                    x.derived = (r, c) in copies
                runs += cell_runs
                runs.append(Run(" ", "space", ("cell", r, c)))
        return runs, {"kind": "table", "shape": (len(tab["rows"]), max((len(r) for r in tab["rows"]), default=0)),
                      "header_rows": tab["header_rows"]}
    lines = html_lines(block.get("html", ""))
    for i, ln in enumerate(lines):
        runs += split_runs(ln, ("line", i))
        runs.append(Run(" ", "space", ("line", i)))
    return runs, {"kind": "text", "n_lines": len(lines)}


def rebuild(runs: list[Run], struct: dict) -> dict:
    parts: dict[tuple, list[str]] = {}
    for r in runs:
        if r.restored and r.cancelled:
            continue
        text = r.text
        if r.restored:   # make sure an inserted value is separated from its neighbours
            text = f" {text} "
        parts.setdefault(r.loc, []).append(text)
    join = lambda loc: re.sub(r"\s+", " ", "".join(parts.get(loc, []))).strip()
    boxes: dict = {}
    for r in runs:
        if r.bbox and not (r.restored and r.cancelled):
            k = r.loc[1]
            b = boxes.get(k)
            boxes[k] = list(r.bbox) if b is None else [min(b[0], r.bbox[0]), min(b[1], r.bbox[1]),
                                                       max(b[2], r.bbox[2]), max(b[3], r.bbox[3])]
    box_of = lambda k: [round(v, 1) for v in boxes[k]] if k in boxes else None
    # row/line numbers may be fractional: rows restored from the text layer sit between OCR rows
    if struct["kind"] == "table":
        n_cols = struct["shape"][1]
        row_ids = sorted({loc[1] for loc in parts if loc[0] == "cell"} | set(range(struct["shape"][0])))
        rows = [[join(("cell", r, c)) for c in range(n_cols)] for r in row_ids]
        restored_rows = [i for i, r in enumerate(row_ids) if r != int(r)]
        text = "\n".join(" | ".join(row) for row in rows)
        return {"kind": "table", "table": {"rows": rows, "header_rows": struct["header_rows"],
                                           "restored_rows": restored_rows, "row_ids": row_ids,
                                           "row_bboxes": [box_of(r) for r in row_ids]}, "text": text}
    line_ids = sorted({loc[1] for loc in parts if loc[0] == "line"} | set(range(struct["n_lines"])))
    lines = [join(("line", i)) for i in line_ids]
    return {"kind": "text", "lines": lines, "line_ids": line_ids, "line_bboxes": [box_of(i) for i in line_ids],
            "text": "\n".join(lines)}


# ------------------------------------------------------------------ reference words

def reference_runs(words) -> list[Run]:
    """Reference words (already inside one block) -> runs in geometric reading order."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2)
    heights = sorted(w.bbox[3] - w.bbox[1] for w in ws)
    tol = max(1.0, heights[len(heights) // 2] * 0.5)
    lines, cur, cur_y = [], [], None
    for w in ws:
        y = (w.bbox[1] + w.bbox[3]) / 2
        if cur and abs(y - cur_y) > tol:
            lines.append(cur)
            cur = []
        cur.append(w)
        cur_y = sum((x.bbox[1] + x.bbox[3]) / 2 for x in cur) / len(cur)
    lines.append(cur)
    runs = []
    for i, line in enumerate(lines):
        for w in sorted(line, key=lambda w: w.bbox[0]):
            text = w.text
            if getattr(w, "legacy", False):
                if not re.fullmatch(r"[0-9]+", text):
                    continue                         # legacy-font letters: never copied
                text = text.translate(str.maketrans("0123456789", BN_DIGITS))
            runs += [r for r in split_runs(text, ("ref", i), tuple(w.bbox)) if r.kind != "space"]
    return runs


# ------------------------------------------------------------------ alignment

# Bangla digits that Tesseract (ben+eng) returns as the Latin digit they look like
_LOOKALIKE = str.maketrans({"০": "0", "৪": "8", "৭": "9"})


def _lookalike(text: str) -> str:
    """'৪৭' -> '89': how Tesseract reads these Bangla digits; other digits must match exactly."""
    return digits_ascii(text.translate(_LOOKALIKE)) if re.search(r"[০৪৭]", text) else "\x00"


def _shape(key: str) -> str:
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"[0-9]", "9", key))


def _pair_cost(a: Run, b: Run) -> float:
    """Cost of treating b as the correct reading of a. > 1.5 (two gaps) means 'never pair'."""
    ratio = difflib.SequenceMatcher(None, a.key, b.key).ratio()
    if a.kind == b.kind == "num":
        if _shape(a.key) == _shape(b.key):
            return 0.5 * (1.0 - ratio)                 # same format, e.g. 87033121 vs 87033211, 0 vs 5
        if not re.search(r"[A-Za-z]", a.key + b.key):
            # different format: only a near-identical value (a dropped/extra digit, 5603119 vs 56031190) is a
            # misreading; 3.00 vs 1704.90.90 is a reading-order difference, never a correction
            return 1.0 - ratio if ratio >= 0.8 else 2.0
        return 2.0 if ratio < 0.6 else 1.0 - ratio     # letters involved: only near-identical
    if a.kind == b.kind == "lat":
        return 2.0 if ratio < 0.4 else 1.0 - ratio
    if a.kind == b.kind == "dash":
        return 0.2                                     # "--" read for "---"
    return 1.0 - ratio if ratio >= 0.85 else 2.0       # e.g. "industry" vs "industry1" (footnote mark)


def _align_replace(ocr: list[Run], ref: list[Run]):
    """Order-preserving pairing of two short lists (edit distance with similarity cost)."""
    gap = 0.75
    n, m = len(ocr), len(ref)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + _pair_cost(ocr[i - 1], ref[j - 1]),
                          D[i - 1][j] + gap, D[i][j - 1] + gap)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + _pair_cost(ocr[i - 1], ref[j - 1]):
            pairs.append((ocr[i - 1], ref[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + gap:
            pairs.append((ocr[i - 1], None)); i -= 1
        else:
            pairs.append((None, ref[j - 1])); j -= 1
    return pairs[::-1]


@dataclass
class BlockMerge:
    runs: list[Run]
    struct: dict
    changes: list[dict] = field(default_factory=list)
    n_nonbn: int = 0


def _absorb_splits(o_part: list[Run], t_part: list[Run]):
    """Drop Latin words that differ only in spacing or line-break hyphenation:
    "Artificial Staple" vs "Artificialstaple", "re-factory" vs "re-" ... "factory"."""
    o, t = list(o_part), list(t_part)
    fold = lambda rs: re.sub(r"[-\u00ad]", "", "".join(r.key for r in rs)).casefold()
    window = 12
    while True:
        hit = None
        for a, b in ((o, t), (t, o)):
            singles = {}
            for j, x in enumerate(b):
                if x.kind == "lat":
                    singles.setdefault(fold([x]), j)
            for i in range(len(a)):
                if a[i].kind != "lat":
                    continue
                for k in range(i + 1, min(len(a), i + window)):
                    if a[k].kind != "lat":
                        continue
                    j = singles.get(fold([a[i], a[k]]))
                    if j is not None:
                        hit = (a, (i, k), b, j)
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            return o, t
        a, (i, k), b, j = hit
        del a[k]
        del a[i]
        del b[j]


def _column_ranges(col_x: dict) -> dict:
    """Table column -> (x0, x1) covered by the reference words that matched it."""
    return {c: (min(a for a, _ in xs), max(b for _, b in xs)) for c, xs in col_x.items() if xs}


def _column_for(ranges: dict, bbox) -> int:
    """Column whose x-range contains the word centre (narrowest wins); else the nearest range."""
    cx = (bbox[0] + bbox[2]) / 2
    inside = [c for c, (a, b) in ranges.items() if a - 1 <= cx <= b + 1]
    if inside:
        return min(inside, key=lambda c: ranges[c][1] - ranges[c][0])
    return min(ranges, key=lambda c: min(abs(cx - ranges[c][0]), abs(cx - ranges[c][1])))


def _restore_missing_lines(runs, struct, ref_all, inserts, col_x, bm):
    """Reference lines that Surya skipped entirely come back as whole rows (tables) or lines (text).

    Words are assigned to table columns by x-position, learned from the words that did match.
    Lines need >= 2 non-Bangla values in a row of missing lines; a lone value stays a plain insert.
    Returns the inserts that were not used.
    """
    by_line: dict[int, list[Run]] = {}
    for r in ref_all:
        by_line.setdefault(r.loc[1], []).append(r)
    pending = {id(t): (a, t) for a, t in inserts}
    missing = [i for i, rs in sorted(by_line.items())
               if any(r.kind in ("num", "lat") for r in rs)
               and all(id(r) in pending for r in rs if r.kind in ("num", "lat"))]
    segments, cur = [], []
    for i in missing:
        if cur and i != cur[-1] + 1:
            segments.append(cur)
            cur = []
        cur.append(i)
    if cur:
        segments.append(cur)
    ranges = _column_ranges(col_x)
    is_table = struct["kind"] == "table"
    for seg_no, seg in enumerate(segments):
        values = [r for i in seg for r in by_line[i] if r.kind in ("num", "lat")]
        if len(values) < 2 or (is_table and not ranges):
            continue
        anchor = pending[id(values[0])][0]
        base = anchor.loc[1] if anchor is not None else -1
        first_col = min(ranges, key=lambda c: ranges[c][0]) if ranges else 0
        row = None
        for k, i in enumerate(seg):
            line_runs = sorted(by_line[i], key=lambda r: r.bbox[0] if r.bbox else 0)
            if is_table:
                col_of = {id(r): _column_for(ranges, r.bbox) for r in line_runs}
                if row is None or any(col_of[id(r)] == first_col for r in line_runs):
                    row = base + (seg_no * 100 + k + 1) / 100000
                locs = {id(r): ("cell", row, col_of[id(r)]) for r in line_runs}
            else:
                row = base + (seg_no * 100 + k + 1) / 100000
                locs = {id(r): ("line", row) for r in line_runs}
            for r in line_runs:
                new = Run(r.text, r.kind, locs[id(r)], r.bbox, restored=True)
                runs.append(new)
                runs.append(Run(" ", "space", locs[id(r)]))
                bm.changes.append(dict(kind="RESTORED", value_kind=r.kind, ocr=None, ref=r.text, run=new,
                                       bbox=r.bbox, reason="line missing in OCR"))
                pending.pop(id(r), None)
    return [pending[k] for k in [id(t) for _, t in inserts] if k in pending]


def _join_hyphenated(ref_sig: list[Run], ocr_sig: list[Run], window: int = 40) -> list[Run]:
    """Reference words broken at a line end ("lub-" ... "blending") are joined when OCR has them as one."""
    fold = lambda t: re.sub(r"[-\u00ad]", "", t).casefold()
    targets = {fold(o.key) for o in ocr_sig if o.kind == "lat" and "-" in o.text}
    if not targets:
        return ref_sig
    out, used = [], set()
    for i, r in enumerate(ref_sig):
        if i in used:
            continue
        if r.kind == "lat" and r.text.endswith(("-", "\u00ad")):
            for k in range(i + 1, min(len(ref_sig), i + window)):
                n = ref_sig[k]
                if k not in used and n.kind == "lat" and fold(r.key + n.key) in targets:
                    r = Run(r.text + n.text, "lat", r.loc, r.bbox)
                    used.add(k)
                    break
        out.append(r)
    return out


def merge_block(block: dict, ref_words, mode: str = "textlayer") -> BlockMerge:
    runs, struct = block_runs(block)
    ocr_sig = _sig(runs)
    ref_all = reference_runs(ref_words)
    ref_sig = _join_hyphenated(_sig(ref_all), ocr_sig)
    bm = BlockMerge(runs, struct, n_nonbn=sum(r.kind in ("num", "lat") for r in ocr_sig))

    sm = difflib.SequenceMatcher(None, [r.key for r in ocr_sig], [r.key for r in ref_sig], autojunk=False)
    inserts: list[tuple[Run | None, Run]] = []     # (anchor OCR run, reference run to insert after it)
    col_x: dict[int, list[float]] = {}             # table column -> x centres of matched reference words

    line_row: dict[int, float] = {}                # reference line -> OCR table row it matched

    def matched(o: Run, t: Run):
        if t.bbox:
            o.bbox = t.bbox
        if o.loc[0] == "cell" and t.bbox:
            col_x.setdefault(o.loc[2], []).append((t.bbox[0], t.bbox[2]))
            line_row.setdefault(t.loc[1], o.loc[1])

    def correction(o: Run, t: Run):
        if mode == "vote":
            if _lookalike(o.core) == t.core:
                return                  # Tesseract read Bangla ৪/৭/০ as Latin 8/9/0: the engines agree
            o.flag = "UNCONFIRMED"
            bm.changes.append(dict(kind="UNCONFIRMED", value_kind=o.kind, ocr=o.text, ref=t.text,
                                   reason="engines disagree", run=o, bbox=t.bbox))
            return
        matched(o, t)
        new = o.text.replace(o.core, t.core, 1) if o.core in o.text else t.core
        bm.changes.append(dict(kind="CORRECTION", value_kind=t.kind, ocr=o.text, ref=t.text, run=o, bbox=t.bbox))
        o.text, o.kind = new, t.kind

    def unconfirmed(o: Run):
        o.flag = "UNCONFIRMED"
        bm.changes.append(dict(kind="UNCONFIRMED", value_kind=o.kind, ocr=o.text, ref=None, run=o, bbox=None))

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for o, t in zip(ocr_sig[i1:i2], ref_sig[j1:j2]):
                matched(o, t)
            continue
        o_part = [r for r in ocr_sig[i1:i2] if r.kind != "bn"]
        t_part = [r for r in ref_sig[j1:j2] if r.kind != "bn"]
        o_part, t_part = _absorb_splits(o_part, t_part)
        anchor = ocr_sig[i1 - 1] if i1 > 0 else None
        for o, t in _align_replace(o_part, t_part):
            if o is not None and t is not None:
                correction(o, t)
                anchor = o
            elif o is not None:
                unconfirmed(o)
                anchor = o
            elif mode != "vote":         # a second OCR engine is never trusted to add values
                inserts.append((anchor, t))

    if inserts:
        inserts = _restore_missing_lines(runs, struct, ref_all, inserts, col_x, bm)

    # remaining single values: insert right after their anchor run (or after earlier insertions there)
    last_after: dict[int, Run] = {}
    ranges = _column_ranges(col_x)
    for anchor, t in inserts:
        loc = anchor.loc if anchor else (runs[0].loc if runs else ("line", 0))
        if loc[0] == "cell" and ranges and t.bbox:    # table: column by x, row by the line's other words
            row = line_row.get(t.loc[1], loc[1])
            col = _column_for(ranges, t.bbox)
            if (row, col) != (loc[1], loc[2]):
                loc = ("cell", row, col)
                anchor = None
                ends = [r for r in runs if r.loc == loc]
                if ends:
                    anchor = ends[-1]
        new = Run(t.text, t.kind, loc, t.bbox, restored=True)
        after = last_after.get(id(anchor), anchor)
        pos = next(i for i, r in enumerate(runs) if r is after) + 1 if after is not None else 0
        runs.insert(pos, new)
        last_after[id(anchor)] = new
        bm.changes.append(dict(kind="RESTORED", value_kind=t.kind, ocr=None, ref=t.text, run=new, bbox=t.bbox))
    return bm


# ------------------------------------------------------------------ page

def _assign_words(blocks: list[dict], words, scale: tuple[float, float]):
    """Each reference word goes to the block (in PDF points) containing its centre; else unassigned."""
    sx, sy = scale
    rects = []
    for b in blocks:
        xs = [p[0] for p in b["polygon"]]
        ys = [p[1] for p in b["polygon"]]
        rects.append((min(xs) * sx - 2, min(ys) * sy - 2, max(xs) * sx + 2, max(ys) * sy + 2))
    per_block = [[] for _ in blocks]
    unassigned = []
    for w in words:
        cx, cy = (w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2
        best, best_area = None, 0.0
        for k, (x0, y0, x1, y1) in enumerate(rects):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best_area:      # innermost block wins
                    best, best_area = k, area
        (per_block[best] if best is not None else unassigned).append(w)
    return per_block, unassigned, rects


def merge_page(surya_page: dict, words, page_size: tuple[float, float], mode: str = "textlayer") -> dict:
    """Merge one page. Returns {'blocks': [...], 'changes': [...], 'metrics': {...}}."""
    iw, ih = surya_page["image_bbox"][2], surya_page["image_bbox"][3]
    scale = (page_size[0] / iw, page_size[1] / ih)
    sblocks = sorted(surya_page["blocks"], key=lambda b: b.get("reading_order", 0))
    per_block, unassigned, rects = _assign_words(sblocks, words, scale)

    merged: list[tuple[dict, BlockMerge | None, tuple]] = []
    for b, ws, rect in zip(sblocks, per_block, rects):
        if b.get("skipped") or b.get("label") in SKIP_LABELS:
            merged.append((b, None, rect))
            continue
        merged.append((b, merge_block(b, ws, mode), rect))

    # page-level cancellation: an UNCONFIRMED OCR value and a RESTORED (or unassigned) reference value with the
    # same value are one value read in a different order — but only when the reference word lies in the same
    # row / line band as the OCR value. Otherwise a value would be moved to the wrong row.
    bands: dict[tuple[int, object], tuple[float, float]] = {}
    for k, (_, bm, _) in enumerate(merged):
        if not bm:
            continue
        for r in bm.runs:
            if r.bbox and not r.restored:
                key = (k, r.loc[1])
                y0, y1 = bands.get(key, (r.bbox[1], r.bbox[3]))
                bands[key] = (min(y0, r.bbox[1]), max(y1, r.bbox[3]))
    # a table row extends down to where the next row starts (multi-line cells); the last row to the block bottom
    for k, (_, bm, rect) in enumerate(merged):
        if not bm or bm.struct["kind"] != "table":
            continue
        rows = sorted(((key[1], v) for key, v in bands.items() if key[0] == k), key=lambda x: x[1][0])
        for i, (rid, (y0, y1)) in enumerate(rows):
            nxt = rows[i + 1][1][0] if i + 1 < len(rows) else rect[3]
            bands[(k, rid)] = (y0, max(y1, nxt))
    owner = {id(c["run"]): k for k, (_, bm, _) in enumerate(merged) if bm for c in bm.changes}

    def near(u_run, ref_bbox) -> bool:
        band = bands.get((owner.get(id(u_run)), u_run.loc[1]))
        if band is None or ref_bbox is None:
            return False                       # the OCR row matched nothing in the text layer: no evidence
        tol = max(3.0, (band[1] - band[0]) * 0.5)
        cy = (ref_bbox[1] + ref_bbox[3]) / 2
        return band[0] - tol <= cy <= band[1] + tol

    restored = [c for _, bm, _ in merged if bm for c in bm.changes if c["kind"] == "RESTORED"]
    unconf = [c for _, bm, _ in merged if bm for c in bm.changes if c["kind"] == "UNCONFIRMED" and c["ref"] is None]
    loose = [r for r in _sig(reference_runs(unassigned)) if r.kind in ("num", "lat")] if mode == "textlayer" else []
    pool: dict[str, list] = {}
    for c in restored:
        pool.setdefault(c["run"].key, []).append(("restored", c, c["run"].bbox))
    for r in loose:
        pool.setdefault(r.key, []).append(("loose", r, r.bbox))
    n_cancel = 0
    for u in unconf:
        cands = pool.get(u["run"].key) or []
        pick = next((i for i, (_, _, bb) in enumerate(cands) if near(u["run"], bb)), None)
        if pick is None:
            continue
        what, item, _ = cands.pop(pick)
        u["run"].flag = None
        u["cancelled"] = True
        if what == "restored":
            item["run"].cancelled = True
            item["cancelled"] = True
        else:
            item.cancelled = True
        n_cancel += 1

    out_blocks, changes = [], []
    for k, (b, bm, rect) in enumerate(merged):
        rec = {"id": k, "label": b.get("label"), "bbox": [round(v, 1) for v in rect],
               "confidence": b.get("confidence"), "html": b.get("html", "")}
        if bm is None:
            rec.update(kind="skipped", text="")
        else:
            rec.update(rebuild(bm.runs, bm.struct))
            for c in bm.changes:
                if c.get("cancelled"):
                    continue
                changes.append({"kind": c["kind"], "value_kind": c["value_kind"], "ocr": c["ocr"],
                                "ref": c["ref"], "block": k, "loc": list(c["run"].loc),
                                "bbox": [round(v, 1) for v in c["bbox"]] if c["bbox"] else rec["bbox"],
                                **({"reason": c["reason"]} if "reason" in c else {})})
        out_blocks.append(rec)

    n_nonbn = sum(bm.n_nonbn for _, bm, _ in merged if bm)
    count = lambda kind, vk=None: sum(1 for c in changes if c["kind"] == kind and (vk is None or c["value_kind"] == vk))
    loose_left = [r for r in loose if not r.cancelled]
    metrics = {
        "blocks": len(out_blocks),
        "nonbn_tokens": n_nonbn,
        "corrections": count("CORRECTION"),
        "corrections_num": count("CORRECTION", "num"),
        "restored": count("RESTORED"),
        "unconfirmed": count("UNCONFIRMED"),
        "unconfirmed_num": count("UNCONFIRMED", "num"),
        "cancelled_pairs": n_cancel,
        "unassigned_ref_values": len(loose_left),
        "correction_rate": round(count("CORRECTION") / n_nonbn, 4) if n_nonbn else 0.0,
    }
    return {"blocks": out_blocks, "changes": changes, "metrics": metrics,
            "unassigned_ref": [{"text": r.text, "bbox": list(r.bbox)} for r in loose_left]}
