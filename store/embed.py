"""Embeddings through the local Ollama server, with a cache in PostgreSQL.

Models are swappable (settings.embed_model). Vectors are L2-normalised and cut to settings.embed_dim
(pgvector indexes at most 2,000 dims): bge-m3 is 1024 already; qwen3-embedding:4b (2560) is truncated
to 1024 and re-normalised (it is trained with Matryoshka representation learning, so this is supported).
Qwen3-Embedding expects an instruction on queries (not on documents).
"""
from __future__ import annotations

import hashlib
import math

import httpx

from core.config import settings

QUERY_INSTRUCTIONS = {
    "qwen3-embedding": "Instruct: Given a question, retrieve passages from regulations, notifications and tables "
                       "that answer it\nQuery: ",
}


def _norm(v: list[float], dim: int) -> list[float]:
    v = v[:dim]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _ollama(texts: list[str], model: str) -> list[list[float]]:
    r = httpx.post(f"{settings.ollama_url}/api/embed",
                   json={"model": model, "input": texts, "truncate": True, "keep_alive": "10m"}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]


def embed_documents(texts: list[str], model: str | None = None, conn=None, batch: int = 32) -> list[list[float]]:
    model = model or settings.embed_model
    dim = settings.embed_dim
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    cached: dict[str, list[float]] = {}
    if conn is not None and texts:
        rows = conn.execute("SELECT text_hash, embedding FROM embed_cache WHERE model = %s AND text_hash = ANY(%s)",
                            (model, list(set(hashes)))).fetchall()
        cached = {h: (e.to_list() if hasattr(e, "to_list") else list(e)) for h, e in rows}
    todo = [(h, t) for h, t in dict(zip(hashes, texts)).items() if h not in cached]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = _ollama([t for _, t in part], model)
        for (h, _), v in zip(part, vecs):
            cached[h] = _norm(v, dim)
            if conn is not None:
                conn.execute("INSERT INTO embed_cache (model, text_hash, embedding) VALUES (%s, %s, %s) "
                             "ON CONFLICT DO NOTHING", (model, h, cached[h]))
    return [cached[h] for h in hashes]


def embed_query(text: str, model: str | None = None) -> list[float]:
    model = model or settings.embed_model
    prefix = next((p for k, p in QUERY_INSTRUCTIONS.items() if model.startswith(k)), "")
    return _norm(_ollama([prefix + text], model)[0], settings.embed_dim)
