"""End-to-end: ingest two small PDFs into a separate database and data folder, then answer questions.

Needs Surya (llama-server), Ollama (qwen3:8b, bge-m3), the re-ranker and PostgreSQL, so it only runs with
DQA_INTEGRATION=1 (`make test-integration`). OCR results already in data/cache are reused.
"""
import hashlib
import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("DQA_INTEGRATION") != "1", reason="set DQA_INTEGRATION=1")

FILES = ["nbr_08.pdf", "nbr_07.pdf"]
QUESTIONS = [
    ("রপ্তানিমুখী শিল্প প্রতিষ্ঠানের আমদানিকৃত পণ্য ওয়্যারহাউসে সর্বোচ্চ কত মাস সংরক্ষণ করা যায়?", ["24"]),
    ("How long may an export-oriented shipbuilding company keep imported goods in its warehouse?", ["48"]),
    ("সাধারণ বন্ড কত টাকার নন-জুডিসিয়াল স্ট্যাম্পে সম্পাদন করতে হয়?", ["2000"]),
]
UNANSWERABLE = "What is the licence fee for a bonded warehouse under these rules?"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import psycopg
    from core.config import settings
    data = tmp_path_factory.mktemp("data")
    real = settings.data_dir
    for f in FILES:                                   # reuse OCR already done for these files
        sha = hashlib.sha256((settings.docs_dir / f).read_bytes()).hexdigest()
        src = real / "cache" / "surya" / sha
        if src.exists():
            shutil.copytree(src, data / "cache" / "surya" / sha)
    with psycopg.connect("host=127.0.0.1 port=5433 user=dqa dbname=dqa", autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS dqa_test")
        c.execute("CREATE DATABASE dqa_test")
    os.environ["DQA_DSN"] = "host=127.0.0.1 port=5433 user=dqa dbname=dqa_test"
    from store import db
    db.DSN = os.environ["DQA_DSN"]
    db.init_schema()
    settings.data_dir = data
    yield data
    settings.data_dir = real
    from ingest import surya_ocr
    surya_ocr.stop()


def test_ingest_and_answer(env):
    from answer.pipeline import answer_text
    from core.textnorm import digits_ascii
    from ingest import pipeline
    from core.config import settings
    for f in FILES:
        path = settings.docs_dir / f
        rep = pipeline.ingest_file(path, pipeline.sha256(path), log=lambda *_: None)
        assert rep["pages_done"] == rep["pages"] and "index" in rep, rep.get("gate")
    from ingest import surya_ocr
    surya_ocr.stop()                                  # free the GPU for answering
    for q, must in QUESTIONS:
        r = answer_text(q)
        ans = digits_ascii(r["answer"]).replace(",", "")
        assert not r["not_found"], (q, r["answer"])
        assert all(m in ans for m in must), (q, r["answer"])
        assert r["citations"], r["answer"]
    r = answer_text(UNANSWERABLE)
    assert r["not_found"], r["answer"]
