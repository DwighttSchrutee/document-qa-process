-- Document Q&A schema (PostgreSQL 16 + pgvector + pg_trgm). Applied by store/db.py (idempotent).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id             bigserial PRIMARY KEY,
    sha256         text UNIQUE NOT NULL,
    filename       text NOT NULL,
    mime           text,
    pages          int,
    uploaded_at    timestamptz DEFAULT now(),
    indexed_at     timestamptz,
    doc_date       date,
    effective_from date,
    effective_to   date,
    status         text,                -- gate summary status: ok | has_held | incomplete
    report         jsonb
);

CREATE TABLE IF NOT EXISTS pages (
    id          bigserial PRIMARY KEY,
    document_id bigint REFERENCES documents(id) ON DELETE CASCADE,
    page_no     int NOT NULL,
    method      text,                   -- merge | surya | surya+tesseract | direct | blank
    status      text,                   -- PASS | WARN | HOLD | REJECTED
    confidence  real,
    metrics     jsonb,
    image_path  text,
    UNIQUE (document_id, page_no)
);

CREATE TABLE IF NOT EXISTS extraction_changes (
    id              bigserial PRIMARY KEY,
    page_id         bigint REFERENCES pages(id) ON DELETE CASCADE,
    kind            text,               -- CORRECTION | RESTORED | UNCONFIRMED
    value_kind      text,
    ocr_value       text,
    textlayer_value text,
    bbox            jsonb,
    resolved_by     text,
    resolved_value  text
);

-- notices inside a document (a compilation holds many)
CREATE TABLE IF NOT EXISTS segments (
    id             bigserial PRIMARY KEY,
    document_id    bigint REFERENCES documents(id) ON DELETE CASCADE,
    seg_index      int,
    ref_key        text,                -- "169/2024"
    ref_text       text,
    issued         date,
    effective_from date,
    effective_to   date,
    first_page     int,
    last_page      int
);
CREATE INDEX IF NOT EXISTS segments_ref ON segments (ref_key);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint REFERENCES documents(id) ON DELETE CASCADE,
    segment_id   bigint REFERENCES segments(id) ON DELETE SET NULL,
    seq          int,                   -- order within the document
    pages        int[],
    bbox         jsonb,                 -- {"page": [x0, y0, x1, y1]} in PDF points
    section_path text[],
    clause_no    text,
    kind         text,                  -- prose | table_row | heading | form
    text         text,                  -- what the answer model sees / is cited
    embed_text   text,                  -- text + context used for the embedding
    text_search  text,                  -- normalised (NFC, ASCII digits, casefold)
    lang         text,
    status       text,                  -- PASS | WARN  (HOLD / REJECTED are never stored)
    n_tokens     int,
    embedding    vector(1024),
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text_search, ''))) STORED,
    metadata     jsonb
);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_trgm ON chunks USING gin (text_search gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (document_id, seq);

-- alternative embeddings for the ablation (same 1024 dims)
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id  bigint REFERENCES chunks(id) ON DELETE CASCADE,
    model     text,
    embedding vector(1024),
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS identifiers (
    normalised text NOT NULL,
    original   text,
    kind       text,                    -- code | prefix | ref | date
    chunk_id   bigint REFERENCES chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS identifiers_norm ON identifiers (normalised);

CREATE TABLE IF NOT EXISTS doc_links (
    id            bigserial PRIMARY KEY,
    from_document bigint REFERENCES documents(id) ON DELETE CASCADE,
    from_segment  bigint REFERENCES segments(id) ON DELETE CASCADE,
    to_reference  text,                 -- ref_key of the target notice
    to_document   bigint REFERENCES documents(id) ON DELETE SET NULL,
    to_segment    bigint REFERENCES segments(id) ON DELETE SET NULL,
    kind          text,                 -- amends | rescinds | refers
    page          int,
    evidence      text
);
CREATE INDEX IF NOT EXISTS doc_links_to ON doc_links (to_segment);

CREATE TABLE IF NOT EXISTS review_queue (
    id          bigserial PRIMARY KEY,
    document_id bigint REFERENCES documents(id) ON DELETE CASCADE,
    page_no     int,
    unit        jsonb,
    status      text,
    check_name  text,
    detail      text,
    value       text
);

CREATE TABLE IF NOT EXISTS embed_cache (
    model     text,
    text_hash text,
    embedding vector,
    PRIMARY KEY (model, text_hash)
);

CREATE TABLE IF NOT EXISTS qa_log (
    id         bigserial PRIMARY KEY,
    at         timestamptz DEFAULT now(),
    question   text,
    request    jsonb,
    retrieved  jsonb,
    answer     text,
    checks     jsonb,
    timings    jsonb
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id       bigserial PRIMARY KEY,
    at       timestamptz DEFAULT now(),
    name     text,
    config   jsonb,
    metrics  jsonb
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id      bigint REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id text,
    result      jsonb
);

-- added later (idempotent upgrades)
ALTER TABLE segments ADD COLUMN IF NOT EXISTS title text;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS effective_text text;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS expiry_text text;
