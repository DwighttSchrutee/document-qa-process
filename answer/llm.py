"""Local LLM calls (Ollama, 127.0.0.1). qwen3:8b with thinking off unless asked."""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from core.config import settings


def chat(messages: list[dict], *, think: bool = False, fmt: dict | str | None = None, temperature: float = 0.0,
         num_ctx: int | None = None, num_predict: int | None = None, timeout: float = 180) -> str:
    body = {"model": settings.llm_model, "messages": messages, "stream": False, "think": think,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": num_ctx or settings.llm_num_ctx,
                        **({"num_predict": num_predict} if num_predict else {})}}
    if fmt is not None:
        body["format"] = fmt
    r = httpx.post(f"{settings.ollama_url}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_stream(messages: list[dict], *, think: bool = False, temperature: float = 0.0,
                num_ctx: int | None = None, timeout: float = 300) -> Iterator[dict]:
    """Yields {'type': 'thinking'|'token', 'text': ...} then {'type': 'stats', ...}."""
    body = {"model": settings.llm_model, "messages": messages, "stream": True, "think": think, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": num_ctx or settings.llm_num_ctx}}
    with httpx.stream("POST", f"{settings.ollama_url}/api/chat", json=body, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            msg = ev.get("message") or {}
            if msg.get("thinking"):
                yield {"type": "thinking", "text": msg["thinking"]}
            if msg.get("content"):
                yield {"type": "token", "text": msg["content"]}
            if ev.get("done"):
                yield {"type": "stats", "prompt_tokens": ev.get("prompt_eval_count"),
                       "output_tokens": ev.get("eval_count"),
                       "seconds": round((ev.get("total_duration") or 0) / 1e9, 2)}
