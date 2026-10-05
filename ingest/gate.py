"""Quality gate between extraction and indexing.

Every page and every table row / text line gets a status:
  PASS  index
  WARN  index, but answers citing it say "could not be verified against the document"
  HOLD  not indexed until a person reviews it
  REJECTED  (by a reviewer) never indexed

Units: ("row", block, i) for table rows, ("line", block, i) for text lines, ("block", block) when a
value cannot be placed more precisely, ("page",) for page-level problems.

Review decisions (data/reviews/<sha>.json) are applied after every evaluation, so they survive
re-processing.

    python -m ingest.gate            # re-gate every ingested page (no OCR), print the upload reports
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from core.config import settings
from ingest import validators, wordlist

ORDER = {"PASS": 0, "WARN": 1, "HOLD": 2, "REJECTED": 3}
MARGIN_LABELS = {"PageHeader", "PageFooter"}     # page numbers, running heads: never hold a page for these


def worst(*statuses: str) -> str:
    return max(statuses, key=ORDER.get, default="PASS")


def _unit_of(block: dict, loc) -> tuple:
    if not loc:
        return ("block", block["id"])
    if loc[0] == "cell" and block.get("table"):
        ids = block["table"].get("row_ids") or list(range(len(block["table"]["rows"])))
        return ("row", block["id"], ids.index(loc[1]) if loc[1] in ids else 0)
    if loc[0] == "line" and block.get("lines") is not None:
        ids = block.get("line_ids") or list(range(len(block["lines"])))
        return ("line", block["id"], ids.index(loc[1]) if loc[1] in ids else 0)
    return ("block", block["id"])


def unit_text(page: dict, unit: tuple) -> str:
    if unit[0] == "page":
        return ""
    b = page["blocks"][unit[1]]
    if unit[0] == "row":
        return " | ".join(b["table"]["rows"][unit[2]])
    if unit[0] == "line":
        return b["lines"][unit[2]]
    return b.get("text", "")


def unit_bbox(page: dict, unit) -> list | None:
    """Position of a row / line on the page (PDF points); falls back to its block."""
    if unit[0] == "page" or unit[1] >= len(page.get("blocks", [])):
        return None
    b = page["blocks"][unit[1]]
    box = None
    if unit[0] == "row" and b.get("table"):
        bx = b["table"].get("row_bboxes") or []
        box = bx[unit[2]] if unit[2] < len(bx) else None
    elif unit[0] == "line":
        bx = b.get("line_bboxes") or []
        box = bx[unit[2]] if unit[2] < len(bx) else None
    if box is None and b.get("bbox") and b["bbox"] and unit[0] in ("row", "line"):
        # estimate the row band from neighbouring rows with known positions
        bx = (b["table"].get("row_bboxes") if unit[0] == "row" and b.get("table") else b.get("line_bboxes")) or []
        prev = next((bx[i] for i in range(unit[2] - 1, -1, -1) if i < len(bx) and bx[i]), None)
        nxt = next((bx[i] for i in range(unit[2] + 1, len(bx)) if bx[i]), None)
        y0 = prev[3] if prev else b["bbox"][1]
        y1 = nxt[1] if nxt else b["bbox"][3]
        if y1 > y0:
            box = [b["bbox"][0], y0, b["bbox"][2], y1]
    return box or b.get("bbox")


def _in_words(value: str, text: str) -> bool:
    """'৬০' in '“৬০ (ষাট)”': the document states the number twice, in digits and in words."""
    from core.numwords import confirmed_numbers
    from core.textnorm import digits_ascii
    v = re.sub(r"\D", "", digits_ascii(value))
    return bool(v) and v in confirmed_numbers(text)


def cell_type(v: str) -> str:
    """code | number | word — coarse type of a table cell (only consistency within a column matters)."""
    t = re.sub(r"\s+", " ", v).strip()
    a = re.sub(r"[০-৯]", "0", t)
    if re.fullmatch(r"\d{4}\.\d{2}(\.\d{2,4})?( \d{4}\.\d{2}(\.\d{2,4})?)*", a):
        return "code"
    if re.fullmatch(r"[-–]|[\d,]+(\.\d+)?\s*%?(\s*/-)?", a):
        return "number"
    return "word"                    # units and descriptions are one family ("kg", "Felt", "Sofa")


def evaluate(page: dict) -> dict:
    """Return the gate record for one extracted page (does not modify the page)."""
    s = settings
    items: list[dict] = []
    page_reasons: list[dict] = []
    m = page.get("metrics", {})
    blocks = page.get("blocks", [])

    def item(status, unit, check, detail, value=None, bbox=None):
        if bbox is None:
            bbox = unit_bbox(page, unit)
        items.append({"status": status, "unit": list(unit), "check": check, "detail": detail, "value": value,
                      "bbox": bbox})

    def page_reason(status, check, detail):
        page_reasons.append({"status": status, "check": check, "detail": detail})

    # ---- value checks (from the merge / two-engine vote)
    for c in page.get("changes", []):
        b = blocks[c["block"]] if c.get("block") is not None and c["block"] < len(blocks) else None
        if b is None:
            continue
        unit = _unit_of(b, c.get("loc"))
        if c["kind"] == "UNCONFIRMED" and not c.get("ref"):
            c = {**c, "bbox": None}            # no reference position: use the row / line position
        if c["kind"] == "UNCONFIRMED":
            if b.get("label") in MARGIN_LABELS:
                item("WARN", unit, "unconfirmed", "unconfirmed value in page margin", c["ocr"], c.get("bbox"))
            elif c["value_kind"] == "num" and _in_words(c["ocr"], unit_text(page, unit)):
                item("PASS", unit, "confirmed_by_words", "number repeated in words next to it", c["ocr"],
                     c.get("bbox"))
            elif c["value_kind"] == "num":
                item("HOLD", unit, "unconfirmed", c.get("reason") or "number not in the text layer", c["ocr"],
                     c.get("bbox"))
            else:
                item("WARN", unit, "unconfirmed", c.get("reason") or "word not in the text layer", c["ocr"],
                     c.get("bbox"))
        elif c["kind"] == "RESTORED" and c["value_kind"] == "bn":
            item("WARN", unit, "restored_bangla", "Bangla restored from the text layer (may be garbled)", c["ref"],
                 c.get("bbox"))

    # ---- generic (and optional domain) format validators, per line / row
    vals = validators.active(s.domain_validators)
    for b in blocks:
        if b.get("kind") == "table":
            units = [(("row", b["id"], i), " | ".join(r)) for i, r in enumerate(b["table"]["rows"])]
        elif b.get("kind") == "text":
            units = [(("line", b["id"], i), ln) for i, ln in enumerate(b.get("lines", []))]
        else:
            continue
        for unit, text in units:
            for name, fn in vals.items():
                for value, problem in fn(text):
                    item("WARN", unit, f"validator:{name}", problem, value)

    # ---- table checks: a column that is almost always filled is empty in this row
    for b in blocks:
        if b.get("kind") != "table":
            continue
        rows = b["table"]["rows"][b["table"].get("header_rows", 0):]
        data_rows = [r for r in rows if sum(1 for c in r if c.strip()) >= 2]     # skip heading-like rows
        if len(data_rows) < 5:
            continue
        hdr = b["table"].get("header_rows", 0)
        for col in range(len(rows[0]) if rows else 0):
            filled = sum(1 for r in data_rows if r[col].strip())
            if filled / len(data_rows) < s.gate_column_fill or filled == len(data_rows):
                continue
            for i, r in enumerate(rows):
                if r in data_rows and not r[col].strip():
                    item("WARN", ("row", b["id"], hdr + i), "table:empty_cell", f"column {col + 1} is empty")

    # ---- table checks: cells that do not fit their column's type (a shifted row)
    for b in blocks:
        if b.get("kind") != "table":
            continue
        hdr = b["table"].get("header_rows", 0)
        rows = b["table"]["rows"]
        data = [(i, r) for i, r in enumerate(rows) if i >= hdr and sum(1 for c in r if c.strip()) >= 3]
        if len(data) < 5:
            continue
        n_cols = max(len(r) for _, r in data)
        dominant = {}
        for c in range(n_cols):
            types = [cell_type(r[c]) for _, r in data if c < len(r) and r[c].strip()]
            if len(types) >= 5:
                top = max(set(types), key=types.count)
                if types.count(top) / len(types) >= 0.8:
                    dominant[c] = top
        if len(dominant) < 2:
            continue
        for i, r in data:
            bad = [c for c, t in dominant.items() if c < len(r) and r[c].strip() and cell_type(r[c]) != t]
            if len(bad) >= 2:
                item("HOLD", ("row", b["id"], i), "table:shifted",
                     f"{len(bad)} cells do not match their column (columns {', '.join(str(c + 1) for c in bad)})")
            elif bad:
                item("WARN", ("row", b["id"], i), "table:column_type", f"column {bad[0] + 1} has an unusual value")

    # ---- page checks
    if page.get("method") == "blank":
        pass
    else:
        ncorr, nonbn = m.get("corrections_num", 0), m.get("nonbn_tokens", 0)
        if nonbn and ncorr >= s.gate_min_corrections and ncorr / nonbn > s.gate_num_correction_rate:
            page_reason("HOLD", "correction_rate", f"{ncorr} of {nonbn} numbers corrected ({ncorr / nonbn:.1%})")
        held_units = {tuple(x["unit"]) for x in items if x["status"] == "HOLD"}
        n_units = sum(len(b["table"]["rows"]) if b.get("kind") == "table" else len(b.get("lines") or [])
                      for b in blocks if b.get("label") not in MARGIN_LABELS)
        if held_units and len(held_units) >= s.gate_page_held_units_min and \
                len(held_units) >= s.gate_page_held_share * max(1, n_units):
            n_unconf = sum(1 for x in items if x["check"] == "unconfirmed" and x["status"] == "HOLD")
            page_reason("HOLD", "unconfirmed", f"{len(held_units)} of {n_units} rows/lines held "
                                               f"({n_unconf} unconfirmed numbers)")
        conf = m.get("ocr_confidence")
        if conf is not None and conf < s.gate_min_confidence:
            page_reason("WARN", "ocr_confidence", f"OCR confidence {conf:.3f}")
        if m.get("ocr_mode") == "block_fallback":
            page_reason("WARN", "ocr_fallback", "OCR fell back to block mode")
        text = "\n".join(b.get("text", "") for b in blocks if b.get("label") not in MARGIN_LABELS)
        valid, n = wordlist.validity(text)
        if valid is not None and n >= s.gate_bangla_min_words:
            if valid < s.gate_bangla_valid_hold:
                page_reason("HOLD", "bangla_words", f"only {valid:.0%} of {n} Bangla words are known words")
            elif valid < s.gate_bangla_valid_warn:
                page_reason("WARN", "bangla_words", f"{valid:.0%} of {n} Bangla words are known words")
        if (m.get("unassigned_ref_values") or 0) >= s.gate_unassigned_warn:
            page_reason("WARN", "unassigned", f"{m['unassigned_ref_values']} text-layer values outside OCR blocks")
        if page.get("page_class") in ("digital", "scanned") and len(text.strip()) < s.gate_min_text_chars:
            page_reason("HOLD", "no_text", "page has visible content but almost no extracted text")
        if valid is not None:
            m["bangla_valid"] = round(valid, 3)

    status = worst(*(r["status"] for r in page_reasons))
    if status == "PASS" and any(x["status"] != "PASS" for x in items):
        status = "WARN"            # page is indexed; its flagged rows carry their own status
    return {"status": status, "reasons": page_reasons, "items": _dedupe(items)}


def _dedupe(items):
    seen, out = set(), []
    for x in items:
        k = (tuple(x["unit"]), x["check"], x["value"])
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# ------------------------------------------------------------------ reviews

def reviews_path(sha: str) -> Path:
    return settings.data_dir / "reviews" / f"{sha}.json"


def load_reviews(sha: str) -> dict:
    p = reviews_path(sha)
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def review_key(page_no: int, unit, value=None) -> str:
    return f"{page_no}:{'/'.join(map(str, unit))}:{value or ''}"


def save_review(sha: str, page_no: int, unit, decision: str, value=None, corrected=None, note=None) -> dict:
    assert decision in ("accept", "correct", "reject")
    revs = load_reviews(sha)
    rec = {"page": page_no, "unit": list(unit), "value": value, "decision": decision,
           "corrected": corrected, "note": note, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    revs[review_key(page_no, unit, value)] = rec
    p = reviews_path(sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(revs, ensure_ascii=False, indent=1), "utf-8")
    return rec


def apply_reviews(page: dict, gate: dict, revs: dict) -> dict:
    """Apply reviewer decisions to a page (in place) and its gate record."""
    pn = page["page_no"]
    for r in [r for r in revs.values() if r["page"] == pn]:
        unit = tuple(r["unit"])
        if unit == ("page",):
            if r["decision"] == "reject":
                gate["status"] = "REJECTED"
            elif gate["status"] in ("HOLD", "WARN"):
                gate["status"] = "PASS" if not any(i["status"] != "PASS" for i in gate["items"]) else "WARN"
                for x in gate["reasons"]:
                    x["status"], x["reviewed"] = "PASS", r["decision"]
            continue
        if r["decision"] == "correct" and r.get("value") and r.get("corrected") is not None:
            _replace_in_unit(page, unit, r["value"], r["corrected"])
        for x in gate["items"]:
            if tuple(x["unit"]) == unit and (r.get("value") is None or x["value"] == r["value"]):
                x["status"] = "REJECTED" if r["decision"] == "reject" else "PASS"
                x["reviewed"] = r["decision"]
        if r["decision"] == "reject" and r.get("value") is None:
            gate["items"].append({"status": "REJECTED", "unit": list(unit), "check": "review",
                                  "detail": "rejected by reviewer", "value": None, "reviewed": "reject"})
    return gate


def _replace_in_unit(page, unit, old, new):
    b = page["blocks"][unit[1]]
    if unit[0] == "row":
        row = b["table"]["rows"][unit[2]]
        for c, cell in enumerate(row):
            if old in cell:
                row[c] = cell.replace(old, new, 1)
                break
        b["text"] = "\n".join(" | ".join(r) for r in b["table"]["rows"])
    elif unit[0] == "line":
        b["lines"][unit[2]] = b["lines"][unit[2]].replace(old, new, 1)
        b["text"] = "\n".join(b["lines"])
    else:
        b["text"] = b["text"].replace(old, new, 1)


def unit_status(gate: dict, unit) -> str:
    """Status of one row / line / block for indexing (page HOLD/REJECTED overrides)."""
    if gate["status"] in ("HOLD", "REJECTED"):
        return gate["status"]
    unit = list(unit)
    st = [x["status"] for x in gate["items"]
          if x["unit"] == unit or (x["unit"][0] == "block" and len(unit) > 1 and x["unit"][1] == unit[1])]
    return worst(*st) if st else "PASS"


# ------------------------------------------------------------------ documents

def gate_document(sha: str) -> dict:
    """(Re-)gate every page of an ingested document; write gate results into the page files."""
    ddir = settings.data_dir / "docs" / sha
    meta = json.loads((ddir / "meta.json").read_text("utf-8"))
    revs = load_reviews(sha)
    counts = {"PASS": 0, "WARN": 0, "HOLD": 0, "REJECTED": 0}
    held_pages, held_units, warn_units = [], 0, 0
    files = sorted((ddir / "pages").glob("p*.json"))
    for f in files:
        page = json.loads(f.read_text("utf-8"))
        page.pop("gate", None)
        if "original_blocks" in page:          # undo earlier reviewer corrections before re-applying
            page["blocks"] = page.pop("original_blocks")
        orig = json.loads(json.dumps(page["blocks"]))
        g = evaluate(page)
        if any(r["decision"] == "correct" for r in revs.values() if r["page"] == page["page_no"]):
            page["original_blocks"] = orig
        apply_reviews(page, g, revs)
        page["gate"] = g
        f.write_text(json.dumps(page, ensure_ascii=False), "utf-8")
        counts[g["status"]] += 1
        if g["status"] == "HOLD":
            held_pages.append(page["page_no"])
        held_units += len({tuple(x["unit"]) for x in g["items"] if x["status"] == "HOLD"})
        warn_units += len({tuple(x["unit"]) for x in g["items"] if x["status"] == "WARN"})
    expected = meta.get("pages")
    doc = {
        "pages_total": expected or len(files),
        "pages_processed": len(files),
        "all_pages_processed": expected is None or len(files) == expected,
        "status_counts": counts,
        "held_pages": held_pages,
        "held_rows_or_lines": held_units,
        "warn_rows_or_lines": warn_units,
        "pass_share": round(counts["PASS"] / max(1, len(files)), 3),
    }
    doc["summary"] = upload_summary(doc)
    (ddir / "gate.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")
    return doc


def upload_summary(doc: dict) -> str:
    c = doc["status_counts"]
    held = f" ({', '.join(f'p{p}' for p in doc['held_pages'][:10])}{', ...' if len(doc['held_pages']) > 10 else ''})" \
        if doc["held_pages"] else ""
    s = f"{doc['pages_processed']} pages: {c['PASS']} passed, {c['WARN']} warnings, {c['HOLD']} held{held}"
    if c["REJECTED"]:
        s += f", {c['REJECTED']} rejected"
    if doc["held_rows_or_lines"]:
        s += f"; {doc['held_rows_or_lines']} rows/lines held for review"
    if not doc["all_pages_processed"]:
        s += f" — INCOMPLETE: {doc['pages_total'] - doc['pages_processed']} pages not processed"
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--remerge", action="store_true",
                    help="rebuild finished documents from the OCR cache first (no OCR is run)")
    args = ap.parse_args()
    for meta in sorted((settings.data_dir / "docs").glob("*/meta.json")):
        sha = meta.parent.name
        info = json.loads(meta.read_text("utf-8"))
        if not (meta.parent / "report.json").exists():
            continue                                  # still being ingested
        if args.remerge:
            from ingest import pipeline
            pipeline.ingest_file(Path(info["path"]), sha, log=lambda *_: None, index=False)
        doc = gate_document(sha)
        print(f"{info['filename']:34} {doc['summary']}")


if __name__ == "__main__":
    main()
