# Document Q&A — local, Bangla + English

Question answering over uploaded documents (PDF, scanned PDF, Word, Excel/CSV, text) with citations.
Everything runs on this Mac: no document text, question or answer is sent to any external service.

Pipeline: upload → extract (Surya OCR merged with the PDF text layer) → quality gate (+ review screen) →
structure-aware chunks → PostgreSQL/pgvector → hybrid retrieval + re-ranker → qwen3:8b answer with citations →
number/citation checks and a second search before "not found". Evaluation tooling in `eval/`.

## Licence warning — Surya model weights

Surya's code is Apache 2.0, but the **model weights** (`datalab-to/surya-ocr-2-gguf`) use a modified
OpenRAIL-M licence: free for research, personal use, and organisations under **US$5M funding/revenue**.
Above that a commercial licence from Datalab is needed. Check this for PeriscopeLabs **and** for each
client that runs the system before delivering it. This project does not decide it.

## Setup (once, needs internet)

```
make setup        # .venv, Tesseract 5 + tessdata_best (ben, eng), PostgreSQL 16 + pgvector, word list
make models       # qwen3:8b, bge-m3, qwen3-embedding:4b (Ollama), BAAI/bge-reranker-v2-m3 (Hugging Face)
```

Everything is installed inside this folder (`.venv/`, `tools/`, `models/`, `data/`) except the model
caches (`~/.ollama`, `~/.cache/huggingface`). Surya and `llama-server` are reused from
`~/Downloads/bench` (paths in `core/config.py`, override with `DQA_LLAMA_SERVER`).

Disk: tools ~0.6 GB, Python env ~1 GB, models ~10 GB (qwen3:8b 5.2, bge-m3 1.2, qwen3-embedding:4b 2.5,
reranker 2.3, Surya 1.4).

## Offline / privacy

- `core/config.py` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
  `DO_NOT_TRACK=1` on import. Downloads are only possible with `DQA_ALLOW_DOWNLOADS=1` (used by `make models`).
- All servers listen on 127.0.0.1 only: FastAPI (8000), PostgreSQL (5433), Ollama (11434, its default),
  llama-server (random local port chosen by Surya).
- The web UI loads nothing from the internet (Vue is vendored in `web/vendor/`).
- To verify: turn Wi-Fi off and run `make test`, `make bench`, then ask questions in the UI.

## Running

```
make ingest       # OCR + merge + gate for every file in ~/Downloads/bench/docs (resumable)
make regate       # rebuild all pages from the OCR cache with current code, re-run the gate (no OCR)
make serve        # http://127.0.0.1:8000
make test         # unit tests (merge and gate tests use cached Surya output in tests/fixtures)
make bench        # OCR regression bench (13 pages) + gate calibration
```

Ingestion speed: 20–50 s per page (Surya on the M3 GPU). Results are cached per page in
`data/cache/surya/`, so an interrupted run resumes where it stopped.

## How extraction works

| Input | Method |
|---|---|
| Digital PDF page | Surya OCR + PDF text-layer merge (`ingest/fusion.py`) |
| Scanned page (no/little text layer, mostly image) | Surya + Tesseract `tessdata_best` as second reader; a number is accepted only when both agree |
| Word | python-docx: headings, paragraphs, tables |
| Excel / CSV | one page per sheet, first row = header |
| Text / Markdown | paragraphs, `#` headings |

Why the merge: the PDFs' Bangla text layer is wrong (Nikosh font with a broken glyph-to-Unicode table:
"তারিখ" → "তাবরখ"), but its digits, codes and English are exact. Surya reads Bangla very well but
sometimes invents codes and skips rows. The merge keeps Surya's Bangla and takes every number, code
and English word from the text layer:

- CORRECTION: Surya value differs → text-layer value used
- RESTORED: text-layer value Surya missed → inserted; whole missing rows come back as table rows
- UNCONFIRMED: Surya value with no text-layer counterpart → kept but flagged (quality gate)

OCR bench (13 pages, answer key = text layer):

| | HS codes | invented | numbers | rate rows exact | Bangla CER |
|---|---|---|---|---|---|
| Surya alone | 93.8% | 12 | 97.0% | 171/192 | 0.65% |
| Surya + merge | 100% | 0 | 100% | 192/192 | 0.65% |

## Quality gate

Every page and every table row / text line is PASS (indexed), WARN (indexed, answers citing it say it
could not be verified) or HOLD (not indexed until reviewed in the Review screen). Thresholds are in
`core/config.py` (`gate_*`):

| Check | Level |
|---|---|
| unconfirmed number in a row / line | row HOLD |
| ≥ 5 unconfirmed numbers on a page | page HOLD |
| numeric corrections > 2% of non-Bangla tokens (and ≥ 3) | page HOLD |
| page has visible content but < 20 characters extracted | page HOLD |
| < 60% / < 80% of Bangla words in the word list (≥ 30 words) | page HOLD / WARN |
| OCR confidence < 0.90, OCR fell back to block mode, ≥ 5 text-layer values outside OCR blocks | page WARN |
| malformed date, reference, code; percentage > 100 (generic validators) | row WARN |
| empty cell in a column that is ≥ 90% filled | row WARN |

Calibration (`make bench`): tariff p1, p61, p101 and the other bench pages PASS; tariff p101 OCR'd at
150 dpi (37 wrong values, 29 invented) is HOLD. Note: the brief expected p101 itself to be the bad page;
that was true at 150 dpi, not at the 192 dpi used now (6 code errors, all corrected).

The Bangla word list (40,752 words) is read directly from Tesseract's `ben.traineddata` dictionary
(`ingest/wordlist.py`), plus words from hand-verified pages.

Review decisions (accept / correct / reject, per value, row or page) are stored in
`data/reviews/<sha256>.json` and re-applied whenever a document is re-processed.

Domain validators are optional plug-ins (`ingest/validators.py`, `DQA_DOMAIN_VALIDATORS=hs_codes`).

## Chunking (`ingest/chunk.py`, `ingest/links.py`)

Generic structure only; no document-type schema.

- **Table rows**: one chunk per row, rendered with column headers (`HS Code: 3305.10.00 | Description of
  Goods: Shampoos | Unit: kg | Minimum Value USD/Unit: 4.00`), with table title and section path. Headers
  carry across pages (a header-less table part with the same column count continues the previous table).
  Dash hierarchy is resolved (`-- Turkeys : > --- Other`). Merged group cells (e.g. a heading over several
  codes) are carried down only when the value is a digit prefix of another cell in the row; such cells
  are listed in `metadata.filled_from_above`. Column-numbering rows ("(1) (2) (3)") are skipped.
- **Prose**: split at headings, section markers (TABLE-n, শর্তাবলি/Conditions, পরিশিষ্ট/Appendix,
  তফসিল/Schedule, ফরম/ছক/Form) and notice headers. Inside a section, whole clauses (১।, 1., (১), (ক),
  (a), (iv)) are packed up to 800 tokens (XLM-R tokenizer, same as bge-m3). A clause is cut only when it is
  longer than 800 tokens on its own: at sub-clauses first, then at sentences with a one-sentence
  overlap; the continuation repeats the clause's first line.
- **Notices** inside a file (a compilation holds many) are found from the letterhead / issuing reference.
  Each gets issue date, effective-from / effective-to (from "কার্যকর হইবে", "পর্যন্ত বলবৎ", "come into force",
  "valid until"; never earlier than the issue date) and references to other notices with the kind
  amends / rescinds / refers. These become `segments` and `doc_links` rows.
- **Status**: HOLD / REJECTED rows and lines are left out; a chunk containing a WARN line is WARN.
- **Identifiers**: codes (`8714.10.90` = `87141090` = `৮৭১৪.১০.৯০`), their 4- and 6-digit prefixes, notice
  references (`REF:169/2024`), dates (`DATE:2024-05-29`).

## Database (`store/`)

PostgreSQL 16 + pgvector 0.7 + pg_trgm on 127.0.0.1:5433 (`data/pg`). Schema: `store/schema.sql`
(documents, pages, extraction_changes, segments, chunks with `vector(1024)` HNSW index, `tsv` GIN index
(`simple` config) and trigram index, identifiers, doc_links, review_queue, embed_cache, qa_log, eval tables).

`make index` (re)loads every finished document. Embeddings (Ollama `bge-m3`, 1024 dims) are cached by text
hash, so re-indexing after a review takes seconds. Ingestion indexes automatically when the database runs.

## Retrieval (`retrieve/search.py`, `retrieve/rerank.py`)

1. Question understanding (`answer/understand.py`): identifiers by regex (instant). qwen3:8b rewrites
   (3–4 queries incl. the other language) are used in the second round; set `REWRITE_MODE = "llm"` in
   `answer/pipeline.py` to use them in the first round too (adds ~5–10 s).
2. Exact identifier match: `8714.10.90`, `87141090`, `৮৭১৪.১০.৯০` hit the same chunks; `8714` matches every code
   in that heading; `SRO 169/2024` / `এস.আর.ও. নং ১৬৯-আইন/২০২৪/...` match the notice reference.
3. Vector search (bge-m3, HNSW cosine) per query + full text (`simple`) + trigram word similarity.
4. Reciprocal rank fusion; exact code/reference hits always rank above fuzzy hits and are never dropped.
5. Amendments: chunks of notices that amend or rescind a retrieved notice are always added.
6. Re-ranker `BAAI/bge-reranker-v2-m3` (fp16 on the Apple GPU) keeps the best 8 within 6,000 tokens.

## Answering (`answer/`)

- qwen3:8b via Ollama, `think: false` by default (the chat has a "reasoning" toggle), `num_ctx` 16384,
  system prompt in `answer/prompt.py` (brief section 7). Passages carry document, page, notice, issue and
  effective dates, section, clause and an UNVERIFIED mark for quality-gate warnings.
- Checks after generation (`answer/verify.py`):
  - every number/code/date/percentage in the answer must appear (digit-normalised, as a whole token) in a
    cited passage; otherwise the answer is regenerated once with the problem stated, and if it still fails
    the sentences with unsupported values are removed and a warning is shown;
  - every `[n]` must exist; an answer without citations is regenerated;
  - citing an unverified passage adds a warning.
- "Not found" is never the first reply: a second search (100 candidates, LLM rewrites in both languages,
  identifier-only search, neighbouring chunks, linked notices) runs first. If that also fails, the reply says
  so and lists the closest passages.
- Every question is logged in `qa_log` (queries, candidates, final passages, re-ranker scores, answer, checks,
  timings).

## API and UI

`make serve` → http://127.0.0.1:8000

| Endpoint | |
|---|---|
| `POST /documents` | upload; ingestion runs in a background job (one at a time) |
| `GET /documents`, `GET /documents/{sha}/report`, `GET /jobs` | documents, per-page quality report, job progress |
| `GET /review/queue`, `GET /pages/{sha}/{page}`, `POST /review/{sha}/{page}` | review screen |
| `POST /ask` | streaming NDJSON: `status`, `sources`, `thinking`, `token`, `reset`, `checks`, `done` |
| `GET /pages/{sha}/{page}/image`, `.../crop?bbox=x0,y0,x1,y1` | page image, cited region |
| `POST /eval/questions`, `GET /eval/latest`, `GET /eval/run/{id}` | test questions, evaluation results |

UI pages: **Ask** (chat; click `[n]` for the passage, the cropped page region and the full page; "add to test
set" under each answer), **Documents** (upload + quality report), **Review** (held/flagged values next to
the page image), **Evaluation** (latest run).

## Evaluation (`eval/`)

- Ask your own questions in the chat and press "add to test set": the question, the correct answer, the
  values the answer must contain and the gold pages are saved to `eval/questions_user.jsonl` (verified).
  Run them with `make eval-user` (server running).
- Or write question files by hand: `eval/questions_<split>.jsonl`, one JSON object per line with
  `id, question, language, category, expected_answer, must_contain, must_not_contain,
  gold_sources: [{file, page}], answerable, verified`. `eval/build_questions.py` can generate table-lookup
  questions from text-layer-verified rows and draft prose questions with qwen3:8b (marked unverified).
- `make eval` runs the verified held-out set through the HTTP API; `--judge` adds a qwen3:8b rubric for prose
  answers. Results go to `eval_runs` / `eval_results`, `eval/report.md` and the Evaluation page.
- `make ablation`: vector only vs hybrid, with/without re-ranker, bge-m3 vs qwen3-embedding:4b (run
  `python -m store.vectors qwen3-embedding:4b` first), with/without the second search round.
- `make cold-test`: drop the database and re-ingest every file (`NOCACHE=1` also discards the OCR cache, which
  means hours of OCR); reviewer decisions in `data/reviews/` are kept.

## Memory and GPU (16 GB M3)

Measured: llama-server (Surya) ~3.4 GB while ingesting; qwen3:8b 5.2 GB + context; bge-m3 1.2 GB; re-ranker
~1.1 GB in fp16. **Surya and the answering models must not run at the same time**: when they did, Surya
slowed from ~11 to ~1 token/s per slot and page requests timed out. The API's ingestion worker stops
llama-server when its queue is empty; a command-line `make ingest` should not run while people ask questions
(the chat shows a notice while an upload is being ingested).

## Known limitations

- Scanned-page path (Surya + Tesseract vote) is implemented and unit-tested but no document in the sample
  corpus is scanned, so it has not been measured.
- Bangla CER is measured on one hand-verified page (`bench/refs/nbr-02_p1.txt`); the brief's second page
  (`nbr_07_p1.txt`) was not available.
- Effective/expiry dates are detected per notice, not per table inside a notice.
- Notice links (amends/rescinds) are detected from wording; a reference without such wording is "refers".
