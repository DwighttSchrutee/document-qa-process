PY      := .venv/bin/python
UV      := $(HOME)/.local/bin/uv
PGBIN   := tools/pg/bin
PGDATA  := data/pg
PGOPTS  := -h 127.0.0.1 -p 5433 -U dqa
MAMBA   := MAMBA_ROOT_PREFIX=$(CURDIR)/tools/mamba tools/bin/micromamba
export PATH := $(CURDIR)/tools/tess/bin:$(PATH)

.PHONY: setup models db-init db-start db-stop ingest index regate serve eval eval-dev ablation cold-test bench test test-integration

## setup: Python env, Tesseract, PostgreSQL+pgvector (all inside this folder). Needs internet once.
setup:
	test -x $(UV) || curl -LsSf https://astral.sh/uv/install.sh | sh
	test -d .venv || $(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) "surya-ocr==0.22.1" pymupdf pillow numpy beautifulsoup4 python-docx openpyxl \
	  pandas pytest sentence-transformers fastapi "uvicorn[standard]" "psycopg[binary]" pgvector httpx python-multipart
	test -x tools/bin/micromamba || (mkdir -p tools && curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C tools bin/micromamba)
	test -x tools/tess/bin/tesseract || $(MAMBA) create -y -q -p $(CURDIR)/tools/tess -c conda-forge tesseract
	test -x $(PGBIN)/postgres || $(MAMBA) create -y -q -p $(CURDIR)/tools/pg -c conda-forge "postgresql=16" pgvector
	mkdir -p models/tessdata_best
	for l in ben eng osd; do test -f models/tessdata_best/$$l.traineddata || \
	  curl -sL -o models/tessdata_best/$$l.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/$$l.traineddata; done
	cp -r tools/tess/share/tessdata/configs tools/tess/share/tessdata/tessconfigs models/tessdata_best/
	$(PY) -m ingest.wordlist
	$(MAKE) models db-init

## models: one-time model downloads (Ollama + Hugging Face). Surya weights come from the bench setup.
models:
	ollama pull qwen3:8b
	ollama pull bge-m3
	ollama pull qwen3-embedding:4b
	DQA_ALLOW_DOWNLOADS=1 HF_HUB_OFFLINE=0 $(PY) -c "from huggingface_hub import snapshot_download as d; d('BAAI/bge-reranker-v2-m3')"

db-init:
	test -d $(PGDATA) || ($(PGBIN)/initdb -D $(PGDATA) -U dqa --auth=trust -E UTF8 --locale=C && \
	  printf "listen_addresses = '127.0.0.1'\nport = 5433\nunix_socket_directories = '/tmp'\nshared_buffers = 512MB\nmaintenance_work_mem = 1GB\n" >> $(PGDATA)/postgresql.conf)
	$(MAKE) db-start
	$(PGBIN)/createdb $(PGOPTS) dqa 2>/dev/null || true
	$(PY) -m store.db

db-start:
	$(PGBIN)/pg_ctl -D $(PGDATA) status >/dev/null 2>&1 || $(PGBIN)/pg_ctl -D $(PGDATA) -l data/pg.log -w start

db-stop:
	$(PGBIN)/pg_ctl -D $(PGDATA) stop

## ingest: OCR + merge + quality gate for every file in DQA_DOCS_DIR (default ~/Downloads/bench/docs)
ingest:
	SURYA_INFERENCE_TIMEOUT_SECONDS=1800 DQA_SURYA_BATCH=24 $(PY) -m ingest.pipeline

## index: (re)load every finished document into PostgreSQL (embeddings are cached)
index: db-start
	$(PY) -m store.indexer

## regate: rebuild pages from the OCR cache with the current merge code and re-run the gate (no OCR)
regate:
	$(PY) -m ingest.gate --remerge

serve: db-start
	.venv/bin/uvicorn web.api:app --host 127.0.0.1 --port 8000

bench:
	$(PY) -m bench.run_bench
	$(PY) -m bench.gate_calibration

test:
	$(PY) -m pytest -q

## test-integration: ingest two small PDFs into a scratch database and answer questions (needs GPU + Ollama)
test-integration: db-start
	DQA_INTEGRATION=1 $(PY) -m pytest -q tests/test_integration.py

## eval: run the held-out test set through the HTTP API (server must be running: make serve)
eval:
	$(PY) -m eval.run --split test --verified-only

## eval-user: run the questions you saved from the chat ("add to test set")
eval-user:
	$(PY) -m eval.run --split user

eval-dev:
	$(PY) -m eval.run --split dev

ablation:
	$(PY) -m eval.ablation --split dev

## cold-test: drop the database, re-ingest every file from docs/ (OCR cache kept unless NOCACHE=1), index
cold-test: db-start
	$(PY) -m store.db --reset
	test -z "$(NOCACHE)" || rm -rf data/cache/surya data/docs
	rm -rf data/docs
	SURYA_INFERENCE_TIMEOUT_SECONDS=1800 DQA_SURYA_BATCH=24 $(PY) -m ingest.pipeline
