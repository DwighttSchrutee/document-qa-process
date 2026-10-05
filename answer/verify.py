"""Checks on a generated answer (brief section 8).

number check   every number, code, date and percentage in the answer (Bangla or ASCII digits) must occur,
               digit-normalised, in at least one cited passage. Numbers from the question, citation markers
               and list numbering are exempt.
citation check every [n] must refer to a provided passage; a factual answer must cite something.
"""
from __future__ import annotations

import re

from core.textnorm import digits_ascii

CITE_RE = re.compile(r"\[(\d{1,2})\]")
NUM_RE = re.compile(r"(?<![\w.])[0-9০-৯]+(?:[.,/:\-][0-9০-৯]+)*%?")


def _variants(n: str) -> set[str]:
    a = digits_ascii(n).rstrip("%")
    out = {a, a.replace(",", ""), re.sub(r"[.,/\-:]", "", a)}
    if re.fullmatch(r"\d+\.0+", a):
        out.add(a.split(".")[0])                    # "3.00" written as "3"
    if re.fullmatch(r"0\d", a):
        out.add(a[1:])                              # "০৬ জুন" vs "৬ জুন"
    return {x for x in out if x}


def _passage_forms(text: str) -> tuple[str, str]:
    t = digits_ascii(text)
    return t.replace(",", ""), re.sub(r"[.,/\-:]", "", t)


def numbers_in(text: str) -> list[str]:
    t = CITE_RE.sub(" ", text)
    t = re.sub(r"(?m)^\s*\d{1,2}[.)]\s", " ", t)        # list numbering
    out = []
    for m in NUM_RE.finditer(t):
        v = m.group(0).strip(".,:-/")
        if v and re.search(r"[0-9০-৯]", v):
            out.append(v)
    return out


def cited(answer: str) -> list[int]:
    return sorted({int(x) for x in CITE_RE.findall(answer)})


def check(answer: str, passages: list[dict], question: str) -> dict:
    n_pass = len(passages)
    cites = cited(answer)
    bad_cites = [c for c in cites if not 1 <= c <= n_pass]
    good = [c for c in cites if 1 <= c <= n_pass]
    from answer.prompt import passage_header
    full = [passage_header(i, p) + "\n" + p["text"] if p.get("filename") else p["text"]
            for i, p in enumerate(passages, 1)]          # the model also sees each passage's header
    texts = [full[c - 1] for c in good] or full
    forms = [_passage_forms(t) for t in texts]
    all_forms = [_passage_forms(t) for t in full]
    q_vals = set().union(*(_variants(x) for x in numbers_in(question))) if numbers_in(question) else set()
    unsupported, uncited_only = [], []
    for n in dict.fromkeys(numbers_in(answer)):
        vs = _variants(n)
        if vs & q_vals:
            continue
        pats = [re.compile(rf"(?<![0-9]){re.escape(v)}(?![0-9])") for v in vs]
        hit = lambda fs: any(pt.search(f[0]) or pt.search(f[1]) for pt in pats for f in fs)
        if hit(forms):
            continue
        if hit(all_forms):
            uncited_only.append(n)                     # in a passage, just not in the one cited
            continue
        unsupported.append(n)
    unverified = [c for c in good if passages[c - 1].get("status") == "WARN"]
    placeholder = bool(re.search(r"<[^<>\n]{6,}>", answer))   # a copied template line, not an answer
    return {"citations": cites, "bad_citations": bad_cites, "unsupported_numbers": unsupported,
            "numbers_in_uncited_passages": uncited_only, "cites_unverified": unverified,
            "has_citation": bool(good), "placeholder": placeholder,
            "ok": not unsupported and not bad_cites and not placeholder}


def remove_unsupported(answer: str, values: list[str]) -> str:
    """Drop sentences / table rows that contain an unsupported value."""
    if not values:
        return answer
    vs = [set(_variants(v)) for v in values]
    parts = re.split(r"(?<=[।.!?])\s+|\n", answer)
    keep = []
    for p in parts:
        nums = {x for n in numbers_in(p) for x in _variants(n)}
        if any(v & nums for v in vs):
            continue
        keep.append(p)
    return "\n".join(x for x in keep if x.strip())


def strip_bad_citations(answer: str, bad: list[int]) -> str:
    for b in bad:
        answer = answer.replace(f"[{b}]", "")
    return answer
