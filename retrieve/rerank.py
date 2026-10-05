"""Cross-encoder re-ranker (default BAAI/bge-reranker-v2-m3 on Apple GPU via MPS, fp16). Swappable by name."""
from __future__ import annotations

from functools import lru_cache

from core.config import settings


@lru_cache(maxsize=2)
def _model(name: str):
    import torch
    from sentence_transformers import CrossEncoder
    device = settings.reranker_device if settings.reranker_device != "mps" or torch.backends.mps.is_available() else "cpu"
    m = CrossEncoder(name, device=device, max_length=512,
                     model_kwargs={"torch_dtype": torch.float16 if device == "mps" else torch.float32})
    return m


def score(question: str, passages: list[str], model: str | None = None, batch: int = 16) -> list[float]:
    if not passages:
        return []
    m = _model(model or settings.reranker_model)
    try:
        return [float(x) for x in m.predict([(question, p) for p in passages], batch_size=batch,
                                            show_progress_bar=False)]
    finally:
        _release()


def _release():
    """PyTorch keeps freed GPU memory cached; on a 16 GB Mac that starves the answering model (Ollama)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def unload():
    _model.cache_clear()
    _release()
