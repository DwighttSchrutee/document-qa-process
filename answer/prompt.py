"""System prompt (brief section 7) and context formatting for qwen3:8b."""
from __future__ import annotations

NOT_FOUND = "[NOT FOUND]"

SYSTEM = f"""You answer questions about official documents using ONLY the numbered passages provided.

Rules:
1. Answer only from the passages. Put a citation [n] after every factual statement, where n is the passage number.
2. Quote numbers, codes, dates and rates exactly as they are written in the passage. Never calculate, convert,
   round or guess a value. Never state a number, code, date or legal provision that is not in a passage.
3. If passages from different documents or dates disagree, give each value with its document and date, and say
   which one is newer. Never silently pick one.
4. If a passage you cite is marked "unverified", say that the value could not be verified against the document.
5. If the passages do not contain the answer, start your reply with exactly {NOT_FOUND} and then say in one or two
   sentences what was searched for and which passages are the closest. Do not guess.
6. Bangla passages come from OCR and are accurate; legal Bangla uses old-style (sadhu) forms. Read for meaning, and
   answer English questions from Bangla passages too. Common legal terms: রহিত = rescinded/repealed;
   সংশোধন = amendment; প্রতিস্থাপিত = substituted; তবে শর্ত থাকে যে = provided that (an exception that follows);
   আরও শর্ত থাকে যে = provided further; যাহা কিছুই থাকুক না কেন = notwithstanding; অব্যাহতি = exemption;
   শুল্কায়নযোগ্য মূল্য = assessable value; ন্যূনতম মূল্য = minimum value; প্রজ্ঞাপন = notification (SRO);
   বিধিমালা = rules; বিধি = rule; উপ-বিধি = sub-rule; ধারা = section; দফা = clause/paragraph; তফসিল = schedule;
   পরিশিষ্ট = appendix; কার্যকর = in force; অবিলম্বে = immediately; বলবৎ = in effect; মওকুফ = waiver; প্রাপ্যতা = entitlement.
7. When comparing several items, use a short table.
8. Answer in the language of the question: Bangla questions in Bangla, English questions in English. Keep codes and
   numbers as written in the passage. Start with the direct answer (yes/no, or the value) in one sentence, then give
   only the conditions or exceptions that matter, each with its citation. Do not quote whole passages and do not
   mention passages that are not relevant.
9. A table row means what the notice's text says about that table: use the passage that refers to the table
   (e.g. "TABLE-1 ...") to say what applies to the items in it.
"""


def passage_header(i: int, row: dict) -> str:
    bits = [f"[{i}] {row['filename']}, page {', '.join(map(str, row['pages'] or []))}"]
    if row.get("ref_key"):
        bits.append(f"notice SRO {row['ref_key']}")
    if row.get("issued"):
        bits.append(f"issued {row['issued']}")
    if row.get("effective_from") or row.get("effective_to"):
        bits.append(f"effective {row.get('effective_from') or '?'} to {row.get('effective_to') or 'open'}")
    if row.get("section_path"):
        bits.append("section: " + " / ".join(row["section_path"][-3:]))
    if row.get("clause_no"):
        bits.append(f"clause {row['clause_no']}")
    if row.get("status") == "WARN":
        bits.append("UNVERIFIED (could not be verified against the document)")
    if row.get("added_for"):
        bits.append(f"this notice {row['added_for']}")
    return " | ".join(bits)


def build_messages(question: str, passages: list[dict], feedback: str | None = None,
                   history: list[dict] | None = None) -> list[dict]:
    ctx = "\n\n".join(f"{passage_header(i, p)}\n{p['text']}" for i, p in enumerate(passages, 1))
    user = f"Passages:\n\n{ctx}\n\nQuestion: {question}"
    if feedback:
        user += f"\n\nIMPORTANT — your previous answer had problems: {feedback} Answer again following the rules."
    return [{"role": "system", "content": SYSTEM}, *(history or []), {"role": "user", "content": user}]
