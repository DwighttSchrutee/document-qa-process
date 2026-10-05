"""Chunking: table headers repeated, hierarchy, page continuation, clause boundaries, statuses, links."""
from ingest.chunk import chunk_document, chunk_record, clause_of
from ingest.links import identifier_tokens


def text_block(i, label, lines):
    return {"id": i, "label": label, "kind": "text", "lines": lines, "text": "\n".join(lines),
            "bbox": [0, 0, 100, 100], "line_bboxes": [[0, k * 10, 100, k * 10 + 9] for k in range(len(lines))]}


def table_block(i, rows, header_rows=1):
    return {"id": i, "label": "Table", "kind": "table", "bbox": [0, 0, 500, 500], "text": "",
            "table": {"rows": rows, "header_rows": header_rows,
                      "row_bboxes": [[0, k * 10, 500, k * 10 + 9] for k in range(len(rows))]}}


def page(n, blocks, gate=None):
    return {"page_no": n, "blocks": blocks, "gate": gate or {"status": "PASS", "reasons": [], "items": []}}


def test_table_rows_carry_headers_title_and_hierarchy():
    p1 = page(1, [
        text_block(0, "Text", ["TABLE-1"]),
        table_block(1, [["Heading", "H.S. Code", "Description", "CD"],
                        ["87.14", "", "Parts and accessories of vehicles", ""],
                        ["", "8714.10", "- Of motorcycles", ""],
                        ["", "8714.10.10", "-- Saddles", "10"],
                        ["", "8714.10.90", "-- Other", "25"]]),
    ])
    # page 2: the table continues without a header row
    p2 = page(2, [table_block(0, [["", "8714.20.00", "- Of carriages for disabled persons", "0"]], header_rows=0)])
    chunks, _ = chunk_document([p1, p2])
    rows = [c for c in chunks if c.kind == "table_row"]
    other = next(c for c in rows if "8714.10.90" in c.text)
    assert other.text == ("Heading: 87.14 | H.S. Code: 8714.10.90 | Description: Parts and accessories of vehicles "
                          "> - Of motorcycles > -- Other | CD: 25")
    assert "TABLE-1" in other.section_path
    cont = next(c for c in rows if "8714.20.00" in c.text)
    assert cont.text.startswith("Heading: 87.14 | H.S. Code: 8714.20.00 | Description: Parts and accessories")
    assert cont.pages == [2] and "CD: 0" in cont.text


def test_clauses_are_never_cut_and_numbers_tracked():
    lines = ["১। শিরোনাম।- (১) এই বিধিমালা সাধারণ বন্ড বিধিমালা, ২০২৪ নামে অভিহিত হইবে।",
             "(২) এই বিধিমালা ৬ জুন, ২০২৪ তারিখে কার্যকর হইবে।",
             "২। সংজ্ঞা।- এই বিধিমালায়,-",
             "(ক) “আইন” অর্থ কাস্টমস আইন, ২০২৩;",
             "(খ) “বন্ড” অর্থ সাধারণ বন্ড।"]
    chunks, _ = chunk_document([page(1, [text_block(0, "Text", lines)])], max_tokens=60)
    texts = [c.text for c in chunks]
    # every clause start is at a line start of some chunk, and no line is split
    for ln in lines:
        assert any(ln in t.split("\n") for t in texts)
    c2 = next(c for c in chunks if "সংজ্ঞা" in c.text)
    assert c2.clause_no == "২" and "২(খ)" in c2.metadata["clauses"]


def test_clause_markers():
    assert clause_of("১। শিরোনাম") == (1, "১")
    assert clause_of("12. Definitions") == (1, "12")
    assert clause_of("(৩) কোনো") == (2, "৩")
    assert clause_of("(ক) “আইন”") == (3, "ক")
    assert clause_of("(iv) where") == (3, "iv")
    assert clause_of("2024 সনের") is None


def test_held_rows_are_left_out_warn_rows_marked():
    g = {"status": "WARN", "reasons": [],
         "items": [{"status": "HOLD", "unit": ["row", 0, 1], "check": "unconfirmed", "value": "45"},
                   {"status": "WARN", "unit": ["row", 0, 2], "check": "table:empty_cell", "value": None}]}
    p = page(1, [table_block(0, [["Code", "SD"], ["87033211", "45"], ["87033212", "20"], ["87033213", "20"]])], g)
    chunks, _ = chunk_document([p])
    assert [c.text for c in chunks] == ["Code: 87033212 | SD: 20", "Code: 87033213 | SD: 20"]
    assert [c.status for c in chunks] == ["WARN", "PASS"]


def test_notices_dates_and_amendment_links():
    lines = ["গণপ্রজাতন্ত্রী বাংলাদেশ সরকার", "প্রজ্ঞাপন", "তারিখ: ০৮ জুন, ২০২৬ খ্রিষ্টাব্দ",
             "এস.আর.ও. নং ১৪৯-আইন/২০২৬/০৪/কাস্টমস।- সরকার, এস.আর.ও. নং ১৬৯-আইন/২০২৪/২১/কাস্টমস এর নিম্নরূপ সংশোধন করিল।",
             "১। এই প্রজ্ঞাপন জারির তারিখ হইতে কার্যকর হইবে।"]
    chunks, segs = chunk_document([page(1, [text_block(0, "Text", lines)])])
    assert len(segs) == 1
    s = segs[0]
    assert (s["ref_key"], s["issued"], s["effective_from"]) == ("149/2026", "2026-06-08", "2026-06-08")
    assert [(r["key"], r["kind"]) for r in s["references"]] == [("169/2024", "amends")]
    assert chunks[0].section_path[0].startswith("এস.আর.ও. নং ১৪৯")


def test_identifiers_normalised():
    ids = {(n, k) for n, _, k in identifier_tokens("HS ৮৭১৪.১০.৯০ and 87141090 under SRO 169-Law/2024/21 dated 29 May, 2024")}
    assert ("87141090", "code") in ids and ("8714", "prefix") in ids
    assert ("REF:169/2024", "ref") in ids and ("DATE:2024-05-29", "date") in ids


def test_chunk_record_fields():
    chunks, _ = chunk_document([page(3, [text_block(0, "Text", ["১। Rates apply to HS 8714.10.90."])])])
    r = chunk_record(chunks[0])
    assert r["pages"] == [3] and r["bbox"]["3"] == [0, 0, 100, 9]
    assert "8714.10.90" in r["text"] and r["text_search"] == r["text"].casefold().replace("১", "1")


def test_group_header_rows_label_their_children():
    p = page(1, [table_block(0, [["Heading", "HS Code", "Description", "Unit", "Minimum Value"],
                                 ["69.07", "All HS Code", "Ceramic Tiles", "", ""],
                                 ["", "", "600 mm X 600 mm", "m2", "6.00"],
                                 ["", "", "200 mm X 200 mm", "m2", "4.75"],
                                 ["69.10", "6910.10.00", "Sanitary ware", "u", "30.00"],
                                 ["", "", "Other", "u", "20.00"]])])
    rows = [c.text for c in chunk_document([p])[0]]
    assert rows[1] == ("Heading: 69.07 | HS Code: All HS Code | Description: Ceramic Tiles > 600 mm X 600 mm | "
                       "Unit: m2 | Minimum Value: 6.00")
    assert "Ceramic" not in rows[4]          # a row with values ends the group


def test_immediately_means_effective_on_issue():
    lines = ["গণপ্রজাতন্ত্রী বাংলাদেশ সরকার", "তারিখ: ২৯ মে, ২০২৪ খ্রিষ্টাব্দ",
             "এস.আর.ও. নং ২১৩-আইন/২০২৪/৬৫/কাস্টমস।- সরকার ...", "(৩) ইহা অবিলম্বে কার্যকর হইবে।"]
    _, segs = chunk_document([page(1, [text_block(0, "Text", lines)])])
    assert (segs[0]["issued"], segs[0]["effective_from"]) == ("2024-05-29", "2024-05-29")
