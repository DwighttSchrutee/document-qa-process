"""Routing and extraction for non-PDF types; scanned-page detection (no OCR run here)."""
import pymupdf
import pytest

from ingest import extract, textlayer


def test_word_paragraphs_headings_tables(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("শর্তাবলি", level=1)
    d.add_paragraph("১। এই প্রজ্ঞাপন জারির তারিখ হইতে কার্যকর হইবে।")
    t = d.add_table(rows=2, cols=3)
    for c, v in enumerate(["HS Code", "CD", "SD"]):
        t.rows[0].cells[c].text = v
    for c, v in enumerate(["8714.10.90", "25", "0"]):
        t.rows[1].cells[c].text = v
    path = tmp_path / "rules.docx"
    d.save(path)
    (page,) = list(extract.extract(path, "x"))
    labels = [b["label"] for b in page["blocks"]]
    assert labels == ["SectionHeader", "Text", "Table"]
    assert page["blocks"][2]["table"]["rows"] == [["HS Code", "CD", "SD"], ["8714.10.90", "25", "0"]]
    assert page["method"] == "direct"


def test_excel_one_page_per_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rates"
    ws.append(["HSCODE", "CD", "VAT"])
    ws.append(["87141090", 25, 15])
    wb.create_sheet("Empty")
    path = tmp_path / "rates.xlsx"
    wb.save(path)
    pages = list(extract.extract(path, "x"))
    assert len(pages) == 1
    assert pages[0]["blocks"][1]["table"]["rows"] == [["HSCODE", "CD", "VAT"], ["87141090", "25", "15"]]


def test_csv_and_markdown(tmp_path):
    (tmp_path / "a.csv").write_text("code,rate\n0101.21.00,5\n", "utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nFirst paragraph.\n\nSecond.\n", "utf-8")
    (p,) = list(extract.extract(tmp_path / "a.csv", "x"))
    assert p["blocks"][1]["table"]["rows"][1] == ["0101.21.00", "5"]
    (q,) = list(extract.extract(tmp_path / "b.md", "x"))
    assert [(b["label"], b["text"]) for b in q["blocks"]] == [("SectionHeader", "Title"), ("Text", "First paragraph."),
                                                              ("Text", "Second.")]


def test_unsupported_type(tmp_path):
    with pytest.raises(ValueError):
        extract.file_kind(tmp_path / "x.pptx")


def test_page_classification(tmp_path):
    src = pymupdf.open()
    p = src.new_page()
    p.insert_text((72, 72), "Digital page with a real text layer, CD 25 VAT 15. " * 3, fontsize=11)
    src.new_page()                                     # blank
    # scanned: an image of the first page, no text layer
    pix = src[0].get_pixmap(dpi=100)
    q = src.new_page()
    q.insert_image(q.rect, pixmap=pix)
    path = tmp_path / "mixed.pdf"
    src.save(path)
    d = pymupdf.open(path)
    assert [textlayer.classify(d[i]) for i in range(3)] == ["digital", "blank", "scanned"]
