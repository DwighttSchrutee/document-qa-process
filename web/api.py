"""FastAPI backend. Bound to 127.0.0.1 (see Makefile `serve`).

    .venv/bin/uvicorn web.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import io
import json
import re
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path

import pymupdf
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import ROOT, settings
from ingest import extract, gate, pipeline

app = FastAPI(title="Document Q&A (local)", docs_url="/api/docs", openapi_url="/api/openapi.json")
DOCS = settings.data_dir / "docs"
UPLOADS = settings.data_dir / "uploads"


# ------------------------------------------------------------------ helpers

def _meta(sha: str) -> dict:
    p = DOCS / sha / "meta.json"
    if not p.exists():
        raise HTTPException(404, "unknown document")
    return json.loads(p.read_text("utf-8"))


def _page(sha: str, page_no: int) -> dict:
    p = DOCS / sha / "pages" / f"p{page_no:04d}.json"
    if not p.exists():
        raise HTTPException(404, "page not extracted")
    return json.loads(p.read_text("utf-8"))


def _after_change(sha: str):
    """Re-gate and (when the index exists) re-index a document after a review decision."""
    gate.gate_document(sha)
    from store import indexer
    indexer.index_document(sha, log=lambda *_: None)


# ------------------------------------------------------------------ ingestion jobs (single worker)

JOBS: dict[str, dict] = {}
_job_q: "queue.Queue[str]" = queue.Queue()


def _worker():
    while True:
        job_id = _job_q.get()
        job = JOBS[job_id]
        job["status"] = "running"
        job["started"] = time.time()
        try:
            def log(msg):
                job["log"].append(msg)
                if "page" in msg:
                    job["progress"] = msg.strip()
            job["report"] = pipeline.ingest_file(Path(job["path"]), job["sha"], log=log)
            job["status"] = "done"
            job["indexed"] = "index" in job["report"]
        except Exception as e:                       # report, keep the worker alive
            job["status"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            job["log"].append(traceback.format_exc())
        finally:
            job["finished"] = time.time()
            try:
                from ingest import surya_ocr
                if _job_q.empty():
                    surya_ocr.stop()                  # free GPU memory for answering
            except Exception:
                pass


threading.Thread(target=_worker, daemon=True).start()


@app.post("/documents")
async def upload(file: UploadFile):
    try:
        extract.file_kind(Path(file.filename))
    except ValueError as e:
        raise HTTPException(415, str(e))
    UPLOADS.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS / f".{time.time_ns()}_{Path(file.filename).name}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    sha = pipeline.sha256(tmp)
    if (DOCS / sha / "report.json").exists():
        tmp.unlink()
        return {"sha": sha, "duplicate": True, "filename": _meta(sha)["filename"],
                "message": "identical file already ingested"}
    dest = UPLOADS / sha / Path(file.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(dest)
    job_id = sha[:12]
    if job_id not in JOBS or JOBS[job_id]["status"] == "failed":
        JOBS[job_id] = {"id": job_id, "sha": sha, "path": str(dest), "filename": dest.name, "status": "queued",
                        "log": [], "progress": "", "queued": time.time()}
        _job_q.put(job_id)
    return {"sha": sha, "job": job_id, "duplicate": False}


@app.get("/jobs")
def jobs():
    return [{k: v for k, v in j.items() if k != "log"} | {"log_tail": j["log"][-5:]} for j in JOBS.values()]


@app.get("/documents")
def documents():
    out = []
    for m in sorted(DOCS.glob("*/meta.json")):
        meta = json.loads(m.read_text("utf-8"))
        g = m.parent / "gate.json"
        meta["gate"] = json.loads(g.read_text("utf-8")) if g.exists() else None
        meta["ingesting"] = not (m.parent / "report.json").exists()
        out.append(meta)
    return out


@app.get("/documents/{sha}/report")
def report(sha: str):
    _meta(sha)
    rep = DOCS / sha / "report.json"
    if not rep.exists():
        raise HTTPException(409, "ingestion not finished")
    r = json.loads(rep.read_text("utf-8"))
    g = DOCS / sha / "gate.json"
    r["gate"] = json.loads(g.read_text("utf-8")) if g.exists() else None
    statuses = {}
    for f in (DOCS / sha / "pages").glob("p*.json"):
        pg = json.loads(f.read_text("utf-8"))
        statuses[pg["page_no"]] = {"status": (pg.get("gate") or {}).get("status"),
                                   "reasons": (pg.get("gate") or {}).get("reasons", [])}
    for row in r["per_page"]:
        row.update(statuses.get(row["page"], {}))
    return r


# ------------------------------------------------------------------ pages and images

def _render(sha: str, page_no: int, dpi: int) -> bytes:
    cache = settings.data_dir / "images" / sha / f"p{page_no:04d}_{dpi}.png"
    if cache.exists():
        return cache.read_bytes()
    meta = _meta(sha)
    if meta["kind"] != "pdf":
        raise HTTPException(404, "no page image for this file type")
    doc = pymupdf.open(meta["path"])
    if not 1 <= page_no <= doc.page_count:
        raise HTTPException(404, "no such page")
    png = doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(png)
    return png


@app.get("/pages/{sha}/{page_no}")
def page(sha: str, page_no: int):
    return _page(sha, page_no)


@app.get("/pages/{sha}/{page_no}/image")
def page_image(sha: str, page_no: int, dpi: int = 110):
    return Response(_render(sha, page_no, max(36, min(dpi, 300))), media_type="image/png")


@app.get("/pages/{sha}/{page_no}/crop")
def page_crop(sha: str, page_no: int, bbox: str, dpi: int = 160, pad: float = 6.0):
    """bbox = x0,y0,x1,y1 in PDF points."""
    meta = _meta(sha)
    if meta["kind"] != "pdf":
        raise HTTPException(404, "no page image for this file type")
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(400, "bbox must be x0,y0,x1,y1")
    doc = pymupdf.open(meta["path"])
    pg = doc[page_no - 1]
    clip = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) & pg.rect
    png = pg.get_pixmap(dpi=max(36, min(dpi, 300)), clip=clip).tobytes("png")
    return Response(png, media_type="image/png")


# ------------------------------------------------------------------ review

@app.get("/review/queue")
def review_queue(include_warn: bool = False):
    out = []
    wanted = {"HOLD", "WARN"} if include_warn else {"HOLD"}
    for m in sorted(DOCS.glob("*/meta.json")):
        meta = json.loads(m.read_text("utf-8"))
        for f in sorted((m.parent / "pages").glob("p*.json")):
            pg = json.loads(f.read_text("utf-8"))
            g = pg.get("gate")
            if not g:
                continue
            if g["status"] in wanted:
                for r in g["reasons"]:
                    if r["status"] in wanted:
                        out.append({"sha": meta["sha256"], "filename": meta["filename"], "page": pg["page_no"],
                                    "unit": ["page"], "status": r["status"], "check": r["check"],
                                    "detail": r["detail"], "value": None, "text": ""})
            for x in g["items"]:
                if x["status"] in wanted:
                    out.append({"sha": meta["sha256"], "filename": meta["filename"], "page": pg["page_no"],
                                "unit": x["unit"], "status": x["status"], "check": x["check"],
                                "detail": x["detail"], "value": x["value"],
                                "text": gate.unit_text(pg, tuple(x["unit"]))[:300]})
    return out


class Decision(BaseModel):
    unit: list
    decision: str               # accept | correct | reject
    value: str | None = None
    corrected: str | None = None
    note: str | None = None


@app.post("/review/{sha}/{page_no}")
def review(sha: str, page_no: int, d: Decision, background: BackgroundTasks):
    _meta(sha)
    if d.decision not in ("accept", "correct", "reject"):
        raise HTTPException(400, "decision must be accept, correct or reject")
    if d.decision == "correct" and (not d.value or d.corrected is None):
        raise HTTPException(400, "a correction needs the original value and the corrected value")
    rec = gate.save_review(sha, page_no, d.unit, d.decision, d.value, d.corrected, d.note)
    _after_change(sha)
    return {"saved": rec, "page": _page(sha, page_no)}


@app.get("/review/{sha}")
def reviews(sha: str):
    return gate.load_reviews(sha)


# ------------------------------------------------------------------ questions

class AskRequest(BaseModel):
    question: str
    think: bool = False
    history: list[dict] = []


@app.post("/ask")
def ask(req: AskRequest):
    """Streaming NDJSON: status, sources, thinking, token, reset, checks, done."""
    from answer import pipeline as answer_pipeline
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    busy = any(j["status"] in ("queued", "running") for j in JOBS.values())

    def gen():
        if busy:
            yield json.dumps({"type": "status", "text": "a document is being ingested; answers are slower until "
                                                        "it finishes"}) + "\n"
        try:
            for ev in answer_pipeline.ask(q, think=req.think, history=req.history[-6:]):
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "text": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class TestQuestion(BaseModel):
    question: str
    expected_answer: str = ""
    must_contain: list[str] = []
    must_not_contain: list[str] = []
    answerable: bool = True
    gold_sources: list[dict] = []
    category: str = "user"
    split: str = "user"


@app.post("/eval/questions")
def add_test_question(t: TestQuestion):
    """Save a question asked in the chat as a verified test item (eval/questions_<split>.jsonl)."""
    import hashlib
    from core.textnorm import language
    if t.answerable and not t.must_contain:
        raise HTTPException(400, "give at least one value the answer must contain")
    split = re.sub(r"[^a-z0-9_]", "", t.split.lower()) or "user"
    item = {"id": "usr-" + hashlib.sha1(t.question.encode("utf-8")).hexdigest()[:10], "question": t.question.strip(),
            "language": "bn" if language(t.question) in ("bn", "mixed") else "en", "category": t.category,
            "expected_answer": t.expected_answer, "must_contain": t.must_contain,
            "must_not_contain": t.must_not_contain, "gold_sources": t.gold_sources, "answerable": t.answerable,
            "verified": True, "verified_by": "user", "split": split, "added": time.strftime("%Y-%m-%d %H:%M")}
    path = ROOT / "eval" / f"questions_{split}.jsonl"
    lines = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()] if path.exists() else []
    lines = [x for x in lines if x["id"] != item["id"]] + [item]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), "utf-8")
    return {"saved": item, "count": len(lines), "file": str(path.relative_to(ROOT))}


@app.get("/eval/latest")
def eval_latest():
    from store import db
    with db.connect() as conn:
        rows = conn.execute("SELECT id, at, name, config, metrics FROM eval_runs ORDER BY at DESC LIMIT 10").fetchall()
    return [{"id": r[0], "at": r[1].isoformat(), "name": r[2], "config": r[3], "metrics": r[4]} for r in rows]


@app.get("/eval/run/{run_id}")
def eval_run(run_id: int):
    from store import db
    with db.connect() as conn:
        run = conn.execute("SELECT id, at, name, config, metrics FROM eval_runs WHERE id = %s", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "no such run")
        res = conn.execute("SELECT question_id, result FROM eval_results WHERE run_id = %s", (run_id,)).fetchall()
    return {"id": run[0], "at": run[1].isoformat(), "name": run[2], "config": run[3], "metrics": run[4],
            "results": [{"id": q, **r} for q, r in res]}


# ------------------------------------------------------------------ frontend

STATIC = ROOT / "web" / "static"
app.mount("/vendor", StaticFiles(directory=ROOT / "web" / "vendor"), name="vendor")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
