"""Merge + quality gate on real pages (cached Surya output + text layer in tests/fixtures).

Calibration from the brief: the bad rate-sheet page must be HOLD, good pages must PASS.
At 192 dpi Surya reads p101 well (6 code errors, all corrected); the bad case is p101 OCR'd at
150 dpi (37 wrong values, 29 invented ones), kept as `tariff_p101_150dpi`.
"""
import json
import re
from pathlib import Path

import pytest

from ingest import extract, gate
from ingest.textlayer import Word

FX = Path(__file__).parent / "fixtures"


def load(name):
    fx = json.loads((FX / f"{name}.json").read_text("utf-8"))
    words = [Word(w["text"], tuple(w["bbox"]), w["font"], w["legacy"], w["line"]) for w in fx["words"]]
    page = type("P", (), {"rect": type("R", (), {"width": fx["size"][0], "height": fx["size"][1]})()})()
    return extract._pdf_page(1, page, words, fx["page_class"], fx["surya"]), words


def hs_codes(text):
    return sorted(c.replace(".", "") for c in re.findall(r"(?<![\d.])(\d{4}\.\d{2}\.\d{2}|\d{8})(?![\d.])", text))


@pytest.mark.parametrize("name", ["tariff_p1", "tariff_p61", "tariff_p101", "nbr01_p2", "nbr06_p2", "nbr02_p1",
                                  "nbr02_p6"])
def test_good_pages_pass(name):
    rec, _ = load(name)
    g = gate.evaluate(rec)
    assert g["status"] in ("PASS", "WARN"), g["reasons"]
    assert not [x for x in g["items"] if x["status"] == "HOLD"]


def test_bad_page_is_held():
    rec, _ = load("tariff_p101_150dpi")
    g = gate.evaluate(rec)
    assert g["status"] == "HOLD"
    assert {r["check"] for r in g["reasons"]} >= {"correction_rate"}


@pytest.mark.parametrize("name", ["tariff_p1", "tariff_p61", "tariff_p101", "nbr01_p2", "nbr06_p2", "nbr02_p6"])
def test_merged_codes_equal_text_layer(name):
    rec, words = load(name)
    merged = "\n".join(b.get("text", "") for b in rec["blocks"])
    assert hs_codes(merged) == hs_codes(" ".join(w.text for w in words))


def test_missing_rows_restored_as_rows():
    rec, _ = load("tariff_p1")
    table = next(b for b in rec["blocks"] if b["kind"] == "table")["table"]
    assert len(table["restored_rows"]) == 6
    for i in table["restored_rows"]:
        row = table["rows"][i]
        assert re.fullmatch(r"\d{8}", row[0]) and row[-1]          # code in column 1, TTI in the last column


def test_review_decisions_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.settings, "data_dir", tmp_path)
    rec, _ = load("tariff_p101_150dpi")
    rec["page_no"] = 1
    g = gate.evaluate(rec)
    held = next(x for x in g["items"] if x["status"] == "HOLD")
    gate.save_review("abc", 1, held["unit"], "reject")
    gate.save_review("abc", 1, ["page"], "accept")
    g = gate.apply_reviews(rec, gate.evaluate(rec), gate.load_reviews("abc"))
    assert g["status"] == "WARN"
    assert gate.unit_status(g, held["unit"]) == "REJECTED"


def test_few_bad_rows_do_not_hold_the_page():
    """nbr_06 p8 pattern: 6 unconfirmed numbers in 5 rows of a ~65-row page -> only those rows are held."""
    rec, _ = load("tariff_p61")
    rec["page_no"] = 1
    table = next(b for b in rec["blocks"] if b["kind"] == "table")
    rec["changes"] = [{"kind": "UNCONFIRMED", "value_kind": "num", "ocr": v, "ref": None, "block": table["id"],
                       "loc": ["cell", r, 4], "bbox": None} for r, v in ((5, "1.11"), (6, "2.22"), (7, "3.33"),
                                                                         (8, "4.44"), (8, "5.55"), (9, "6.66"))]
    g = gate.evaluate(rec)
    assert g["status"] == "WARN" and not [r for r in g["reasons"] if r["status"] == "HOLD"]
    assert len({tuple(x["unit"]) for x in g["items"] if x["status"] == "HOLD"}) == 5
