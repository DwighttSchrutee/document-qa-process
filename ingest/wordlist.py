"""Bangla word list for the quality gate, read from Tesseract's own dictionary (no external tools).

Reads the LSTM word DAWG (component 19) and the LSTM unicharset (component 21) straight out of
ben.traineddata (same result as `combine_tessdata -u` + `dawg2wordlist`), plus words from
hand-verified pages in bench/refs/. Cached as models/bangla_words.txt.

    python -m ingest.wordlist          # build the cache, print size and a sample
"""
from __future__ import annotations

import math
import re
import struct
import sys
from functools import lru_cache

from core.config import ROOT, settings
from core.textnorm import BN_LETTER, clean

LSTM_SYSTEM_DAWG, LSTM_UNICHARSET = 19, 21
WORDS_FILE = ROOT / "models" / "bangla_words.txt"
_BN_WORD = re.compile(f"[{BN_LETTER}]+")


def _components(path) -> dict[int, bytes]:
    data = path.read_bytes()
    n = struct.unpack("<i", data[:4])[0]
    offs = struct.unpack(f"<{n}q", data[4:4 + 8 * n])
    present = sorted((o, i) for i, o in enumerate(offs) if o >= 0)
    out = {}
    for k, (o, i) in enumerate(present):
        end = present[k + 1][0] if k + 1 < len(present) else len(data)
        out[i] = data[o:end]
    return out


def _unicharset(blob: bytes) -> list[str]:
    lines = blob.decode("utf-8").split("\n")
    count = int(lines[0])
    chars = []
    for ln in lines[1:count + 1]:
        ch = ln.split(" ", 1)[0]
        chars.append(" " if ch == "NULL" else ch)
    return chars


def _dawg_words(blob: bytes, chars: list[str]) -> list[str]:
    magic, size, n_edges = struct.unpack("<hii", blob[:10])
    assert magic == 42, "not a squished DAWG"
    edges = struct.unpack(f"<{n_edges}Q", blob[10:10 + 8 * n_edges])
    flag_start = math.ceil(math.log2(size)) if size > 1 else 0
    next_start = flag_start + 3
    letter_mask = (1 << flag_start) - 1
    MARKER, BACKWARD, WORD_END = 1, 2, 4
    words: list[str] = []
    stack = [(0, "")]
    seen = 0
    while stack:
        node, prefix = stack.pop()
        e = node
        while e < n_edges:
            rec = edges[e]
            flags = (rec >> flag_start) & 7
            if not flags & BACKWARD:
                letter = rec & letter_mask
                word = prefix + (chars[letter] if letter < len(chars) else "")
                if flags & WORD_END:
                    words.append(word)
                nxt = rec >> next_start
                if nxt:
                    stack.append((nxt, word))
            seen += 1
            if flags & MARKER or seen > 50_000_000:
                break
            e += 1
    return words


def build() -> set[str]:
    comps = _components(settings.tessdata_dir / "ben.traineddata")
    words = _dawg_words(comps[LSTM_SYSTEM_DAWG], _unicharset(comps[LSTM_UNICHARSET]))
    out = {clean(w) for w in words if _BN_WORD.fullmatch(clean(w))}
    for ref in (ROOT / "bench" / "refs").glob("*.txt"):
        out.update(_BN_WORD.findall(clean(ref.read_text("utf-8"))))
    WORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WORDS_FILE.write_text("\n".join(sorted(out)), "utf-8")
    return out


@lru_cache(maxsize=1)
def words() -> frozenset[str]:
    if not WORDS_FILE.exists():
        build()
    return frozenset(WORDS_FILE.read_text("utf-8").split("\n"))


def validity(text: str) -> tuple[float | None, int]:
    """Share of Bangla words (2+ letters) found in the word list, and how many were checked."""
    ws = [w for w in _BN_WORD.findall(clean(text)) if len(w) >= 2]
    if not ws:
        return None, 0
    vocab = words()
    return sum(1 for w in ws if w in vocab) / len(ws), len(ws)


if __name__ == "__main__":
    ws = build()
    print(len(ws), "Bangla words;", sorted(ws)[:5], sorted(ws)[-5:])
    sys.exit(0)
