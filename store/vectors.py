"""Alternative embeddings for the ablation (stored in chunk_vectors, same 1024 dims).

    python -m store.vectors qwen3-embedding:4b      # embed every chunk with this model (cached)
Retrieval uses them with Options(embed_model="qwen3-embedding:4b").
"""
from __future__ import annotations

import sys

from store import db, embed


def fill(model: str, batch: int = 64):
    with db.connect() as conn:
        safe = model.replace(":", "_").replace("-", "_").replace(".", "_")
        conn.execute(f"CREATE INDEX IF NOT EXISTS chunk_vectors_{safe} ON chunk_vectors "
                     f"USING hnsw (embedding vector_cosine_ops) WHERE model = '{model}'")
        rows = conn.execute("""SELECT id, embed_text FROM chunks c WHERE NOT EXISTS
                               (SELECT 1 FROM chunk_vectors v WHERE v.chunk_id = c.id AND v.model = %s)
                               ORDER BY id""", (model,)).fetchall()
        for i in range(0, len(rows), batch):
            part = rows[i:i + batch]
            vecs = embed.embed_documents([t for _, t in part], model=model, conn=conn)
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO chunk_vectors (chunk_id, model, embedding) VALUES (%s, %s, %s) "
                                "ON CONFLICT DO NOTHING", [(cid, model, v) for (cid, _), v in zip(part, vecs)])
            conn.commit()
            print(f"{model}: {min(i + batch, len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    fill(sys.argv[1])
