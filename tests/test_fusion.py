"""Merge rules on synthetic blocks (no Surya needed). Page = 1000x1000 image px = 500x500 PDF pt."""
from ingest.fusion import merge_page, split_runs
from ingest.textlayer import Word

PAGE = (500.0, 500.0)


def surya(*blocks):
    return {"image_bbox": [0, 0, 1000, 1000],
            "blocks": [{"polygon": poly, "label": label, "html": html, "reading_order": i, "confidence": 0.9}
                       for i, (label, html, poly) in enumerate(blocks)]}


def box(x0, y0, x1, y1):            # image px
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def words(*items, y=20.0, legacy=False):
    """Words laid out left to right on one line (PDF points)."""
    out, x = [], 10.0
    for t in items:
        out.append(Word(t, (x, y, x + 8 * len(t), y + 10), "f", legacy, 1))
        x += 8 * len(t) + 4
    return out


def test_runs_split_bangla_and_numbers():
    kinds = [(r.text, r.kind) for r in split_runs("নং ১৬৯-আইন/২০২৪ CD 25%", ("line", 0)) if r.kind != "space"]
    assert kinds == [("নং", "bn"), ("১৬৯-", "num"), ("আইন", "bn"), ("/২০২৪", "num"), ("CD", "lat"), ("25%", "num")]


def test_correction_replaces_wrong_code():
    sp = surya(("Table", "<table><tr><td>87033131</td><td>25</td></tr></table>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("87033211", "25"), PAGE)
    assert m["blocks"][0]["table"]["rows"] == [["87033211", "25"]]
    assert [(c["kind"], c["ocr"], c["ref"]) for c in m["changes"]] == [("CORRECTION", "87033131", "87033211")]


def test_bangla_kept_from_ocr_digits_from_textlayer():
    # text layer Bangla is garbled ("তাবরখ"), but its digits are right
    sp = surya(("Text", "<p>তারিখ: ২৯ মে, ২০২৪</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("তাবরখ:", "২৮", "মম,", "২০২৪"), PAGE)
    assert m["blocks"][0]["text"] == "তারিখ: ২৮ মে, ২০২৪"
    assert m["metrics"]["corrections"] == 1


def test_restored_value_is_inserted():
    sp = surya(("Text", "<p>CD 25 VAT</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("CD", "25", "VAT", "15"), PAGE)
    assert m["blocks"][0]["text"] == "CD 25 VAT 15"
    assert [c["kind"] for c in m["changes"]] == ["RESTORED"]


def test_unconfirmed_value_is_flagged_not_removed():
    sp = surya(("Text", "<p>CD 25 SD 45</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("CD", "25", "SD"), PAGE)
    assert m["blocks"][0]["text"] == "CD 25 SD 45"
    assert [(c["kind"], c["ocr"]) for c in m["changes"]] == [("UNCONFIRMED", "45")]


def test_reordering_cancels_on_the_same_line():
    # Surya's block 2 polygon is too narrow: the text-layer "15" on the same line falls outside every block
    sp = surya(("Text", "<p>VAT</p>", box(0, 0, 1000, 100)), ("Text", "<p>rate 15</p>", box(0, 200, 400, 300)))
    ws = words("VAT", y=20) + words("rate", y=120) + [Word("15", (300, 120, 316, 130), "f", False, 3)]
    m = merge_page(sp, ws, PAGE)
    assert m["changes"] == []
    assert m["blocks"][1]["text"] == "rate 15"


def test_same_value_in_another_row_is_not_a_reordering():
    # the code exists on the page, but in a different row: Surya's placement stays UNCONFIRMED
    html = ("<table><tr><td>9028.30.10<br/>9028.30.90</td><td>KWH meter</td></tr>"
            "<tr><td>9028.30.90</td><td>Single phase</td></tr></table>")
    sp = surya(("Table", html, box(0, 0, 1000, 400)))
    ws = [Word("9028.30.10", (10, 10, 60, 18), "f", False, 1), Word("KWH", (100, 10, 120, 18), "f", False, 1),
          Word("meter", (122, 10, 150, 18), "f", False, 1), Word("9028.30.90", (10, 22, 60, 30), "f", False, 2),
          Word("Single", (100, 80, 130, 88), "f", False, 3), Word("phase", (132, 80, 160, 88), "f", False, 3)]
    m = merge_page(sp, ws, PAGE)
    assert [(c["kind"], c["ocr"]) for c in m["changes"] if c["kind"] == "UNCONFIRMED"] == [("UNCONFIRMED", "9028.30.90")]


def test_legacy_font_text_never_copied_but_digits_are():
    sp = surya(("Text", "<p>শর্তাবলি ১</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("kZ©vewj", "1", legacy=True), PAGE)
    assert m["blocks"][0]["text"] == "শর্তাবলি ১"
    assert m["changes"] == []


def test_vote_mode_never_copies():
    sp = surya(("Text", "<p>HS 87033131 CD 25</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("HS", "87033211", "CD"), PAGE, mode="vote")
    assert m["blocks"][0]["text"] == "HS 87033131 CD 25"
    assert sorted((c["kind"], c["ocr"]) for c in m["changes"]) == [("UNCONFIRMED", "25"), ("UNCONFIRMED", "87033131")]


def test_table_header_rows_and_cells():
    html = ("<table><tr><th>HS Code</th><th>CD</th></tr>"
            "<tr><td>87141090</td><td>25</td></tr></table>")
    sp = surya(("Table", html, box(0, 0, 1000, 200)))
    ws = words("HS", "Code", "CD", y=10) + words("87141090", "25", y=40)
    m = merge_page(sp, ws, PAGE)
    t = m["blocks"][0]["table"]
    assert t["header_rows"] == 1 and t["rows"][1] == ["87141090", "25"]
    assert m["changes"] == []


def test_missing_table_row_is_restored_as_a_row():
    html = ("<table><tr><td>01012100</td><td>Horses</td><td>5</td></tr>"
            "<tr><td>01013000</td><td>Asses</td><td>5</td></tr></table>")
    sp = surya(("Table", html, box(0, 0, 1000, 400)))

    def row(code, desc, rate, y):
        return [Word(code, (10, y, 60, y + 8), "f", False, 1), Word(desc, (100, y, 150, y + 8), "f", False, 1),
                Word(rate, (300, y, 310, y + 8), "f", False, 1)]
    ws = row("01012100", "Horses", "5", 10) + row("01012900", "Other", "10", 30) + row("01013000", "Asses", "5", 50)
    m = merge_page(sp, ws, PAGE)
    assert m["blocks"][0]["table"]["rows"] == [["01012100", "Horses", "5"], ["01012900", "Other", "10"],
                                               ["01013000", "Asses", "5"]]
    assert m["blocks"][0]["table"]["restored_rows"] == [1]
    assert {c["reason"] for c in m["changes"]} == {"line missing in OCR"}


def test_latin_spacing_difference_is_not_a_correction():
    sp = surya(("Text", "<p>Artificial Staple fibres 25</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("Artificialstaple", "fibres", "25"), PAGE)
    assert m["blocks"][0]["text"] == "Artificial Staple fibres 25"
    assert m["changes"] == []


def test_rowspan_copies_are_not_flagged():
    html = ("<table><tr><td rowspan='2'>04.09</td><td>0409.00.10</td></tr>"
            "<tr><td>0409.00.90</td></tr></table>")
    sp = surya(("Table", html, box(0, 0, 1000, 200)))
    ws = [Word("04.09", (10, 15, 40, 23), "f", False, 1),
          Word("0409.00.10", (100, 10, 150, 18), "f", False, 1), Word("0409.00.90", (100, 30, 150, 38), "f", False, 2)]
    m = merge_page(sp, ws, PAGE)
    assert m["changes"] == []
    assert m["blocks"][0]["table"]["rows"] == [["04.09", "0409.00.10"], ["04.09", "0409.00.90"]]


def test_number_is_never_replaced_by_a_word():
    sp = surya(("Text", "<p>rate 0</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("rate", "industry1"), PAGE)
    assert "0" in m["blocks"][0]["text"].split()


def test_hyphenated_line_break_matches():
    sp = surya(("Text", "<p>lub-blending plant 25</p>", box(0, 0, 1000, 100)))
    ws = words("lub-", "25", y=10) + words("blending", "plant", y=30)
    m = merge_page(sp, ws, PAGE)
    assert m["changes"] == [] and m["blocks"][0]["text"] == "lub-blending plant 25"


def test_stacked_cell_reading_order_is_not_a_correction():
    # one cell holds stacked codes; the text layer reads line by line, so "3.00" comes before the
    # later codes. Surya's correct reading must stay untouched.
    html = ("<table><tr><td>17.04</td><td>1704.10.10<br/>1704.10.90</td><td>Sugar</td><td>kg</td><td>3.00</td></tr>"
            "</table>")
    sp = surya(("Table", html, box(0, 0, 1000, 200)))
    ws = [Word("17.04", (10, 10, 40, 18), "f", False, 1), Word("1704.10.10", (60, 10, 110, 18), "f", False, 1),
          Word("Sugar", (130, 10, 160, 18), "f", False, 1), Word("kg", (200, 10, 210, 18), "f", False, 1),
          Word("3.00", (300, 10, 320, 18), "f", False, 1), Word("1704.10.90", (60, 22, 110, 30), "f", False, 2)]
    m = merge_page(sp, ws, PAGE)
    assert m["blocks"][0]["table"]["rows"] == [["17.04", "1704.10.10 1704.10.90", "Sugar", "kg", "3.00"]]
    assert m["changes"] == []


def test_vote_mode_accepts_tesseract_lookalike_digits():
    sp = surya(("Text", "<p>ধারা ৪৭ এবং ১২৫</p>", box(0, 0, 1000, 100)))
    m = merge_page(sp, words("ধারা", "89", "এবং", "125"), PAGE, mode="vote")
    assert m["changes"] == []
    m = merge_page(sp, words("ধারা", "88", "এবং", "125"), PAGE, mode="vote")     # 88 is not how ৪৭ looks
    assert [(c["kind"], c["ocr"]) for c in m["changes"]] == [("UNCONFIRMED", "৪৭")]
