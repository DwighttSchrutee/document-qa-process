"""Question understanding: qwen3:8b turns a question into search queries, identifiers, intent, language.

Output (validated): {"queries": [3-4 rewrites, one in the other language], "identifiers": [...],
                     "intent": "...", "language": "bn"|"en", "source": "llm"|"fallback"}
On any failure: the raw question + identifiers found by regex.
"""
from __future__ import annotations

import json
import re

from core.textnorm import clean, digits_ascii, language
from ingest.links import find_refs

SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
        "identifiers": {"type": "array", "items": {"type": "string"}},
        "intent": {"type": "string", "enum": ["lookup_value", "conditions", "procedure", "definition", "dates",
                                              "comparison", "amendment", "list", "other"]},
        "language": {"type": "string", "enum": ["bn", "en"]},
    },
    "required": ["queries", "identifiers", "intent", "language"],
}

PROMPT = """You prepare search queries for a document search system. The documents are Bangladesh government
notifications, rules and tariff tables, in Bangla and English. Do NOT answer the question.

Return JSON with:
- "queries": 3 or 4 search queries that would find the passage answering the question:
  the question rephrased as a statement, a short keyword query, and AT LEAST ONE query in the other language
  (Bangla if the question is English, English if it is Bangla). Keep codes, numbers and names exactly as written.
- "identifiers": every code, reference number, SRO/notification number, HS code, date or number in the question,
  copied exactly as written (do not convert digits, do not invent any).
- "intent": one of lookup_value, conditions, procedure, definition, dates, comparison, amendment, list, other.
- "language": "bn" if the question is mainly Bangla, otherwise "en".

Question: {question}"""

ID_RE = re.compile(r"[0-9০-৯][0-9০-৯.\-/]{3,}[0-9০-৯]")


def regex_identifiers(q: str) -> list[str]:
    ids = [m.group(0) for m in ID_RE.finditer(q)]
    ids += [r["text"] for r in find_refs(q)]
    # "SRO 169/2024", "169/2024" without the এস.আর.ও. prefix
    ids += re.findall(r"(?<![0-9০-৯])[0-9০-৯]{1,4}\s*/\s*[0-9০-৯]{4}(?![0-9০-৯])", q)
    return list(dict.fromkeys(ids))


def fallback(q: str) -> dict:
    lang = "bn" if language(q) in ("bn", "mixed") else "en"
    return {"queries": [q], "identifiers": regex_identifiers(q), "intent": "other", "language": lang,
            "source": "fallback"}


def understand(question: str, use_llm: bool = True) -> dict:
    q = clean(question).strip()
    if not use_llm:
        return fallback(q)
    try:
        from answer import llm
        raw = llm.chat([{"role": "user", "content": PROMPT.replace("{question}", q)}], fmt=SCHEMA,
                       num_ctx=4096, num_predict=400, timeout=60)
        data = json.loads(raw)
        queries = [clean(x).strip() for x in data.get("queries", []) if isinstance(x, str) and x.strip()]
        ids = [clean(x).strip() for x in data.get("identifiers", []) if isinstance(x, str) and x.strip()]
        if not queries:
            return fallback(q)
        # identifiers must literally occur in the question (digit-normalised): the model may not add any
        qd = digits_ascii(q).replace(" ", "")
        ids = [x for x in ids if re.sub(r"\D", "", digits_ascii(x)) and
               re.sub(r"\D", "", digits_ascii(x)) in re.sub(r"\D", "", qd)]
        ids = list(dict.fromkeys(ids + regex_identifiers(q)))
        return {"queries": list(dict.fromkeys([q] + queries))[:5], "identifiers": ids,
                "intent": data.get("intent", "other"),
                "language": data.get("language") if data.get("language") in ("bn", "en") else fallback(q)["language"],
                "source": "llm"}
    except Exception as e:                                    # model down, bad JSON, timeout
        out = fallback(q)
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out
