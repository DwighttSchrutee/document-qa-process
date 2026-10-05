"""Numbers written in words (Bangla and English), as legal text repeats them: "৬০ (ষাট)", "৯০ (নব্বই)",
"২,০০০/- (দুই হাজার)", "48 (forty-eight)". Used to confirm an OCR number by the words next to it.
"""
from __future__ import annotations

import re
import unicodedata

_BN_1_99 = """এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ বিশ
একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ উনত্রিশ ত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ
সাঁইত্রিশ আটত্রিশ উনচল্লিশ চল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ
উনপঞ্চাশ পঞ্চাশ একান্ন বাহান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন উনষাট ষাট একষট্টি বাষট্টি তেষট্টি
চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি উনসত্তর সত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর
আটাত্তর উনআশি আশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি আটাশি উননব্বই নব্বই একানব্বই বিরানব্বই
তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই""".split()
assert len(_BN_1_99) == 99

# spelling variants seen in official text
_BN_VARIANTS = {"ছয়": 6, "ছয়": 6, "নয়": 9, "নয়": 9, "এগার": 11, "বার": 12, "তের": 13, "পনের": 15, "ষোল": 16,
                "সতের": 17, "আঠার": 18, "শূন্য": 0, "একশ": 100, "একশো": 100, "একশত": 100}
_BN_SCALE = {"শত": 100, "শো": 100, "শ": 100, "হাজার": 1000, "লক্ষ": 100_000, "লাখ": 100_000, "কোটি": 10_000_000}

_EN_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen").split()
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_EN_SCALE = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "lac": 100_000, "million": 1_000_000,
             "crore": 10_000_000}


def _norm_bn(w: str) -> str:
    w = unicodedata.normalize("NFC", w).replace("‌", "").replace("‍", "")
    return w.replace("ঊ", "উ").replace("য়", "য়").replace("ড়", "ড়")


_BN_WORDS = {_norm_bn(w): i for i, w in enumerate(_BN_1_99, 1)}
_BN_WORDS.update({_norm_bn(k): v for k, v in _BN_VARIANTS.items()})


def words_to_number(text: str) -> int | None:
    """'নব্বই' -> 90, 'দুই হাজার' -> 2000, 'এক লক্ষ পঞ্চাশ হাজার' -> 150000, 'forty-eight' -> 48."""
    toks = [t for t in re.split(r"[\s\-,]+", text.strip().lower()) if t and t not in ("and", "only", "টাকা", "মাত্র")]
    if not toks:
        return None
    total, cur, seen = 0, 0, False
    for t in toks:
        tb = _norm_bn(t)
        if tb in _BN_WORDS:
            cur += _BN_WORDS[tb]
        elif tb in _BN_SCALE:
            scale = _BN_SCALE[tb]
            if scale == 100:
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * scale
                cur = 0
        elif t in _EN_UNITS:
            cur += _EN_UNITS.index(t)
        elif t in _EN_TENS:
            cur += _EN_TENS[t]
        elif t in _EN_SCALE:
            scale = _EN_SCALE[t]
            if scale == 100:
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * scale
                cur = 0
        else:
            return None                       # a word that is not a number word: not a number in words
        seen = True
    return total + cur if seen else None


_PAIR = re.compile(r"([0-9০-৯][0-9০-৯,]*)(?:\.0+)?\s*(?:/-)?\s*[\"“”‘’']?\s*\(\s*([^()]{2,60}?)\s*\)")


def confirmed_numbers(text: str) -> set[str]:
    """Numbers in `text` that are immediately followed by the same value in words (digits as ASCII)."""
    from core.textnorm import digits_ascii
    out = set()
    for m in _PAIR.finditer(text):
        n = digits_ascii(m.group(1)).replace(",", "")
        if n.isdigit() and words_to_number(m.group(2)) == int(n):
            out.add(n)
    return out
