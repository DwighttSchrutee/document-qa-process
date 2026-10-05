"""All settings in one place. Every value can be overridden with an environment variable DQA_<NAME>.

Importing this module also switches off telemetry and (after setup) network access for model hubs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Privacy: no telemetry anywhere; model hubs offline unless DQA_ALLOW_DOWNLOADS=1 (setup only).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
if os.environ.get("DQA_ALLOW_DOWNLOADS") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class Settings:
    data_dir: Path = ROOT / "data"
    docs_dir: Path = Path.home() / "Downloads" / "bench" / "docs"

    # --- OCR -------------------------------------------------------------------------------
    llama_server: Path = Path.home() / "Downloads" / "bench" / "llama" / "llama-b11376" / "llama-server"
    surya_dpi: int = 192
    surya_batch: int = 8                 # pages per request batch (llama-server runs 8 slots)
    # a page counts as scanned when it has (almost) no text layer but visible content
    scanned_min_chars: int = 20
    scanned_text_area: float = 0.05      # text boxes cover < 5% of the page ...
    scanned_image_area: float = 0.5      # ... and images cover > 50%
    legacy_font_pattern: str = r"MJ|Sutonny|Bijoy|SolaimanLipi_?MJ"   # ANSI-encoded Bangla fonts
    tesseract_cmd: str = str(ROOT / "tools" / "tess" / "bin" / "tesseract") \
        if (ROOT / "tools" / "tess" / "bin" / "tesseract").exists() else "tesseract"
    tessdata_dir: Path = ROOT / "models" / "tessdata_best"

    # --- quality gate (calibrated on bench pages; see README "Quality gate") ----------------
    # page HOLD: numeric corrections / non-Bangla tokens above this (good pages 0-1%, bad p101@150dpi 3.4%)
    gate_num_correction_rate: float = 0.02
    gate_min_corrections: int = 3            # ...and at least this many (small pages are noisy)
    # whole-page HOLD only when the held rows/lines are widespread; otherwise just those rows are held and the
    # rest of the page is indexed (a few garbled rows must not hide the correct ones next to them)
    gate_page_held_units_min: int = 5        # at least this many rows/lines held ...
    gate_page_held_share: float = 0.30       # ... and at least this share of the page's rows/lines
    gate_min_confidence: float = 0.90        # Surya mean token prob below this -> WARN
    gate_bangla_valid_warn: float = 0.80     # share of Bangla words in the word list
    gate_bangla_valid_hold: float = 0.60
    gate_bangla_min_words: int = 30
    gate_unassigned_warn: int = 5            # text-layer values outside every OCR block
    gate_min_text_chars: int = 20            # page with ink but less text than this -> HOLD
    gate_column_fill: float = 0.9            # column filled in >= 90% of rows: empty cell -> WARN
    domain_validators: str = ""              # optional plug-ins, e.g. "hs_codes"

    # --- models / services (used from checkpoint 3 on) -------------------------------------
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:8b"
    llm_num_ctx: int = 8192             # prompt is budgeted to ~3,000 tokens (retrieve.Options)
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"        # on a 16 GB Mac, sharing the GPU with Ollama stalls answers for minutes
    rerank_candidates: int = 30         # how many fused candidates the re-ranker scores
    # exact cosine search is ~20 ms at 10k-30k chunks and never misses; the HNSW index missed a best match
    # (thousands of near-identical tariff rows trap the graph search). Use "hnsw" only for very large corpora.
    vector_search: str = "exact"

    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for f in fields(self):
            env = os.environ.get(f"DQA_{f.name.upper()}")
            if env is None or f.name == "extra":
                continue
            cur = getattr(self, f.name)
            if isinstance(cur, bool):
                val = env.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                val = int(env)
            elif isinstance(cur, float):
                val = float(env)
            elif isinstance(cur, Path):
                val = Path(env).expanduser()
            else:
                val = env
            setattr(self, f.name, val)


settings = Settings()
