"""Surya OCR (full-page mode) through llama.cpp's llama-server, with a per-page JSON cache.

Cache: data/cache/surya/<sha256>/p<page>_<dpi>.json
  {image_size, image_bbox, mode: full_page|block_fallback, seconds, blocks: [{polygon, label, html, confidence, ...}]}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from core.config import settings

os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
# a dense table page is ~8,000 output tokens: ~12 min at ~11 tok/s on an M3, more than Surya's default 600 s
os.environ.setdefault("SURYA_INFERENCE_TIMEOUT_SECONDS", "3600")
os.environ.setdefault("LLAMA_CPP_BINARY", str(settings.llama_server))

_manager = None
_predictor = None


def cache_path(sha: str, page_no: int, dpi: int | None = None) -> Path:
    return settings.data_dir / "cache" / "surya" / sha / f"p{page_no:04d}_{dpi or settings.surya_dpi}.json"


def _get_predictor():
    global _manager, _predictor
    if _predictor is None:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings as ss
        ss.LLAMA_CPP_BINARY = str(settings.llama_server)
        ss.SURYA_INFERENCE_TIMEOUT_SECONDS = float(os.environ["SURYA_INFERENCE_TIMEOUT_SECONDS"])

        class Predictor(RecognitionPredictor):
            """Remembers which images went through block-mode fallback."""
            fallback_ids: set[int] = set()

            def __call__(self, images, layout_results=None, *, full_page=None):
                if full_page is False and layout_results is not None:
                    self.fallback_ids.update(id(i) for i in images)
                return super().__call__(images, layout_results, full_page=full_page)

        _manager = SuryaInferenceManager()
        _predictor = Predictor(_manager)
    return _predictor


def stop():
    """Shut down llama-server (frees GPU memory for the answering model)."""
    global _manager, _predictor
    if _manager is not None:
        _manager.stop()
    _manager = _predictor = None


def ocr_pages(sha: str, images: dict[int, "Image.Image"], dpi: int | None = None) -> dict[int, dict]:
    """OCR page images (page_no -> PIL image); cached pages are not re-run."""
    results, todo = {}, []
    for p, img in images.items():
        cp = cache_path(sha, p, dpi)
        if cp.exists():
            results[p] = json.loads(cp.read_text("utf-8"))
        else:
            todo.append(p)
    for i in range(0, len(todo), settings.surya_batch):
        batch = todo[i:i + settings.surya_batch]
        pred = _get_predictor()
        pred.fallback_ids.clear()
        imgs = [images[p] for p in batch]
        t0 = time.time()
        out = pred(imgs, full_page=True)
        secs = (time.time() - t0) / len(batch)
        for p, img, res in zip(batch, imgs, out):
            rec = {
                "image_size": list(img.size),
                "image_bbox": res.image_bbox,
                "mode": "block_fallback" if id(img) in pred.fallback_ids else "full_page",
                "seconds": round(secs, 2),
                "blocks": [b.model_dump() for b in res.blocks],
            }
            cp = cache_path(sha, p, dpi)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(rec, ensure_ascii=False), "utf-8")
            results[p] = rec
    return results
