"""Text normalisation shared by ingestion, search and answer checking.

- `clean`      : NFC, ZWJ/ZWNJ removed. Used for everything we store and display.
                 (NFC also turns the precomposed U+09DF "য়" into U+09AF U+09BC, as OCR emits it.)
- `search_form`: clean + Bangla digits -> ASCII + casefold. Used for keyword search.
- `norm_identifier`: digits -> ASCII, separators removed, upper-case. 8714.10.90 = 87141090 = ৮৭১৪.১০.৯০.
"""
from __future__ import annotations

import re
import unicodedata

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_DIGITS_TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")
_ZW = dict.fromkeys(map(ord, "‌‍﻿"), None)

# Bangla letters, signs and marks; Bangla digits are deliberately excluded (they are numbers).
BN_LETTER = r"ঀ-৥ৰ-৿"
_BN_LETTER_RE = re.compile(f"[{BN_LETTER}]")
_ID_SEP_RE = re.compile(r"[\s./\-_:,()]+")


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_ZW)


def digits_ascii(text: str) -> str:
    return text.translate(_DIGITS_TO_ASCII)


def search_form(text: str) -> str:
    return digits_ascii(clean(text)).casefold()


def norm_identifier(text: str) -> str:
    return _ID_SEP_RE.sub("", digits_ascii(clean(text))).upper()


def has_bangla(text: str) -> bool:
    return bool(_BN_LETTER_RE.search(text))


def bangla_share(text: str) -> float:
    letters = [c for c in text if c.isalpha() or _BN_LETTER_RE.match(c)]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _BN_LETTER_RE.match(c)) / len(letters)


def language(text: str) -> str:
    """'bn', 'en' or 'mixed' by share of Bangla letters."""
    s = bangla_share(text)
    return "bn" if s > 0.7 else "en" if s < 0.3 else "mixed"
