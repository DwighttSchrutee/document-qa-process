"""Load an ingested document (data/docs/<sha>) into PostgreSQL: pages, changes, notices, chunks,
identifiers, links, review queue. Re-indexing replaces everything for that document.

    python -m store.indexer              # every finished document
    python -m store.indexer NAME_OR_SHA  # one document
"""
from __future__ import annotations

import json
import sys
import time

from psycopg.types.json import Jsonb

from core.config import settings
from ingest.chunk import chunk_document, chunk_record
from store import db, embed

DOCS = settings.data_dir / "docs"


def _load(sha: str):
    d = DOCS / sha
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    pages = [json.loads(f.read_text("utf-8")) for f in sorted((d / "pages").glob("p*.json"))]
    g = d / "gate.json"
    gate = json.loads(g.read_text("utf-8")) if g.exists() else None
    return meta, pages, gate


MIME = {"pdf": "application/pdf", "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "sheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text": "text/plain"}


def index_document(sha: str, log=print) -> dict:
    t0 = time.time()
    meta, pages, gate = _load(sha)
    chunks, segments = chunk_document(pages)
    recs = [chunk_record(c) for c in chunks]
    with db.connect() as conn:
        vecs = embed.embed_documents([r["embed_text"] for r in recs], conn=conn)
        first = next((s for s in segments if s.get("issued")), {})
        status = "incomplete" if gate and not gate["all_pages_processed"] else \
            "has_held" if gate and (gate["held_pages"] or gate["held_rows_or_lines"]) else "ok"
        doc_id = conn.execute(
            """INSERT INTO documents (sha256, filename, mime, pages, doc_date, effective_from, effective_to, status,
                                      report, indexed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
               ON CONFLICT (sha256) DO UPDATE SET filename = EXCLUDED.filename, pages = EXCLUDED.pages,
                 doc_date = EXCLUDED.doc_date, effective_from = EXCLUDED.effective_from,
                 effective_to = EXCLUDED.effective_to, status = EXCLUDED.status, report = EXCLUDED.report,
                 indexed_at = now()
               RETURNING id""",
            (sha, meta["filename"], MIME.get(meta["kind"]), meta.get("pages") or len(pages), first.get("issued"),
             first.get("effective_from"), first.get("effective_to"), status, Jsonb(gate or {}))).fetchone()[0]
        for t in ("chunks", "pages", "segments", "doc_links", "review_queue"):
            col = "from_document" if t == "doc_links" else "document_id"
            conn.execute(f"DELETE FROM {t} WHERE {col} = %s", (doc_id,))

        for p in pages:
            g = p.get("gate") or {}
            m = p.get("metrics") or {}
            page_id = conn.execute(
                "INSERT INTO pages (document_id, page_no, method, status, confidence, metrics) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                (doc_id, p["page_no"], p["method"], g.get("status"), m.get("ocr_confidence"), Jsonb(m))).fetchone()[0]
            ch = p.get("changes") or []
            if ch:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO extraction_changes (page_id, kind, value_kind, ocr_value, textlayer_value, bbox) "
                        "VALUES (%s,%s,%s,%s,%s,%s)",
                        [(page_id, c["kind"], c.get("value_kind"), c.get("ocr"), c.get("ref"), Jsonb(c.get("bbox")))
                         for c in ch])
            for r in g.get("reasons", []):
                if r["status"] in ("HOLD", "WARN"):
                    conn.execute("INSERT INTO review_queue (document_id, page_no, unit, status, check_name, detail) "
                                 "VALUES (%s,%s,%s,%s,%s,%s)",
                                 (doc_id, p["page_no"], Jsonb(["page"]), r["status"], r["check"], r["detail"]))
            for x in g.get("items", []):
                if x["status"] in ("HOLD", "WARN"):
                    conn.execute("INSERT INTO review_queue (document_id, page_no, unit, status, check_name, detail, "
                                 "value) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                 (doc_id, p["page_no"], Jsonb(x["unit"]), x["status"], x["check"], x["detail"],
                                  x.get("value")))

        seg_ids = {}
        for s in segments:
            seg_ids[s["index"]] = conn.execute(
                "INSERT INTO segments (document_id, seg_index, ref_key, ref_text, title, issued, effective_from,"
                " effective_to, effective_text, expiry_text, first_page, last_page)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (doc_id, s["index"], s["ref_key"], s["ref_text"], s.get("title"), s["issued"], s["effective_from"],
                 s["effective_to"], s.get("effective_text"), s.get("expiry_text"), s["first_page"],
                 s["last_page"])).fetchone()[0]
            for r in s["references"]:
                conn.execute("INSERT INTO doc_links (from_document, from_segment, to_reference, kind, page, evidence) "
                             "VALUES (%s,%s,%s,%s,%s,%s)",
                             (doc_id, seg_ids[s["index"]], r["key"], r["kind"], r["page"], r["evidence"]))

        ids_rows = []
        for seq, (r, v) in enumerate(zip(recs, vecs)):
            cid = conn.execute(
                """INSERT INTO chunks (document_id, segment_id, seq, pages, bbox, section_path, clause_no, kind, text,
                                       embed_text, text_search, lang, status, n_tokens, embedding, metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (doc_id, seg_ids.get(r["segment"]), seq, r["pages"], Jsonb(r["bbox"]), r["section_path"],
                 r["clause_no"], r["kind"], r["text"], r["embed_text"], r["text_search"], r["lang"], r["status"],
                 r["n_tokens"], v, Jsonb(r["metadata"]))).fetchone()[0]
            ids_rows += [(n, o, k, cid) for n, o, k in r["identifiers"]]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO identifiers (normalised, original, kind, chunk_id) VALUES (%s,%s,%s,%s)",
                            ids_rows)
        resolve_links(conn)
    with db.connect(autocommit=True) as conn:
        conn.execute("VACUUM ANALYZE chunks")      # repairs the HNSW graph for the rows just replaced
    out = {"document_id": doc_id, "chunks": len(recs), "segments": len(segments), "identifiers": len(ids_rows),
           "seconds": round(time.time() - t0, 1)}
    log(f"indexed {meta['filename']}: {out}")
    return out


def resolve_links(conn):
    """Point every link at the notice it names, in whichever indexed document holds it: by reference number,
    or — when the amending text names the notice by its title — by that title."""
    conn.execute("""UPDATE doc_links l SET to_segment = s.id, to_document = s.document_id
                    FROM segments s WHERE s.ref_key = l.to_reference AND s.document_id <> l.from_document""")
    norm = "regexp_replace(translate({0}, '০১২৩৪৫৬৭৮৯', '0123456789'), '[[:space:][:punct:]–—‘’“”]', '', 'g')"
    conn.execute(f"""UPDATE doc_links l SET to_segment = s.id, to_document = s.document_id,
                         to_reference = coalesce(l.to_reference, s.ref_key)
                     FROM segments s
                     WHERE (l.to_segment IS NULL OR l.to_reference IS NULL) AND s.title IS NOT NULL
                       AND length(s.title) > 10 AND s.document_id <> l.from_document
                       AND position({norm.format('s.title')} in {norm.format('l.evidence')}) > 0""")
    conn.execute("""UPDATE doc_links l SET to_segment = NULL, to_document = NULL
                    WHERE to_segment IS NOT NULL AND NOT EXISTS (SELECT 1 FROM segments s WHERE s.id = l.to_segment)""")


def maintain():
    """Re-indexing deletes and re-inserts rows; pgvector's HNSW graph loses recall with many dead rows
    (a best match was missed), so clean up and rebuild the vector index."""
    with db.connect(autocommit=True) as conn:
        conn.execute("VACUUM ANALYZE chunks")
        conn.execute("REINDEX INDEX chunks_embedding")


def main():
    db.init_schema()
    want = sys.argv[1] if len(sys.argv) > 1 else None
    for m in sorted(DOCS.glob("*/meta.json")):
        meta = json.loads(m.read_text("utf-8"))
        if not (m.parent / "report.json").exists():
            continue
        if want and want not in (meta["filename"], meta["sha256"]):
            continue
        index_document(meta["sha256"])
    maintain()


if __name__ == "__main__":
    main()
