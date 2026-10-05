"""Pluggable value validators for the quality gate.

A validator is `fn(text) -> list[(value, problem)]`. Generic validators are always on; domain
validators are enabled by name in settings.domain_validators (comma list). Validators only WARN:
they catch malformed values, not wrong-but-well-formed ones (that is the merge's job).
"""
from __future__ import annotations

import datetime as dt
import re

from core.textnorm import digits_ascii

D = r"[0-9০-৯]"
EN_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
EN_MONTHS.update({k[:3]: v for k, v in list(EN_MONTHS.items())})
BN_MONTHS_GREG = {"জানুয়ারি": 1, "ফেব্রুয়ারি": 2, "মার্চ": 3, "এপ্রিল": 4, "মে": 5, "জুন": 6, "জুলাই": 7,
                  "আগস্ট": 8, "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12}
# Bangla calendar (বঙ্গাব্দ) months: 31 days for the first six (revised calendar), 30 for the rest (ফাল্গুন 29/30)
BN_MONTHS_BANGLA = {"বৈশাখ": 31, "জ্যৈষ্ঠ": 31, "আষাঢ়": 31, "শ্রাবণ": 31, "ভাদ্র": 31, "আশ্বিন": 31,
                    "কার্তিক": 30, "অগ্রহায়ণ": 30, "পৌষ": 30, "মাঘ": 30, "ফাল্গুন": 30, "চৈত্র": 30}


def _real_date(y, m, d) -> bool:
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def dates(text: str):
    """dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, '08 June, 2026', '০৮ জুন, ২০২৬', '২৫ জ্যৈষ্ঠ, ১৪৩৩'."""
    out = []
    t = digits_ascii(text)
    for m in re.finditer(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?!\d)", t):
        d, mo, y = map(int, m.groups())
        if 1900 <= y <= 2100 and not _real_date(y, mo, d) and not _real_date(y, d, mo):
            out.append((m.group(0), "not a real date"))
    for m in re.finditer(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", t):
        d, name, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if name in EN_MONTHS and not _real_date(y, EN_MONTHS[name], d):
            out.append((m.group(0), "not a real date"))
    for name, mo in BN_MONTHS_GREG.items():
        for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s*{name},?\s*(\d{{4}})", t):
            if not _real_date(int(m.group(2)), mo, int(m.group(1))):
                out.append((m.group(0), "not a real date"))
    for name, days in BN_MONTHS_BANGLA.items():
        for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s*{name}", t):
            if not 1 <= int(m.group(1)) <= days:
                out.append((m.group(0), "day out of range for Bangla month"))
    return out


def percentages(text: str):
    out = []
    for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", digits_ascii(text)):
        if float(m.group(1)) > 100:
            # >100% occurs (e.g. supplementary duty); only flag when it is not near a duty/fixed marker
            ctx = text[max(0, m.start() - 40):m.end() + 10].lower()
            if not re.search(r"duty|শুল্ক|sd|fixed|tk|taka|টাকা", ctx):
                out.append((m.group(0), "percentage above 100"))
    return out


def references(text: str):
    """Notification references: '১৪৯-আইন/২০২৬/০৪/কাস্টমস' or '169-Law/2024/21/Customs' style."""
    out = []
    t = digits_ascii(text)
    for m in re.finditer(r"(?:এস\.?\s?আর\.?\s?ও\.?|S\.?\s?R\.?\s?O\.?)\s*(?:নং|No\.?)?\s*([^\s,;)]+)", t, re.I):
        ref = m.group(1).rstrip("।.-")
        if not re.search(r"\d", ref):
            continue
        if not re.fullmatch(r"\d{1,4}(?:-(?:আইন|Law|law|AIN))?/\d{4}(?:/\d{1,4})?(?:/[^\s/]+)?", ref):
            out.append((ref, "reference does not match number-আইন/year/number"))
    return out


def identifiers(text: str):
    """Code-like tokens with dot groups must have a consistent shape (e.g. 8714.10.90, not 8714.1090)."""
    out = []
    for m in re.finditer(r"(?<![\w.])(\d{4})\.(\d{1,6})(?:\.(\d{1,4}))?(?![\w.])", digits_ascii(text)):
        a, b, c = m.groups()
        if c is None and len(b) == 4:          # 8714.1090
            out.append((m.group(0), "malformed code"))
    return out


# ---- optional domain plug-ins ----------------------------------------------------------------

def hs_codes(text: str):
    """Bangladesh customs: HS codes are 8 digits (or 4.2.2 dotted)."""
    out = []
    for m in re.finditer(r"(?<![\d.])(\d{4}\.\d{2}\.\d{2}|\d{8})(?![\d.])", digits_ascii(text)):
        if int(m.group(0).replace(".", "")[:2]) > 97:
            out.append((m.group(0), "HS chapter above 97"))
    return out


GENERIC = {"dates": dates, "percentages": percentages, "references": references, "identifiers": identifiers}
DOMAIN = {"hs_codes": hs_codes}


def active(domain: str = "") -> dict:
    v = dict(GENERIC)
    for name in filter(None, (x.strip() for x in domain.split(","))):
        v[name] = DOMAIN[name]
    return v
