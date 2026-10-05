"""Generic document structure: notice segments, dates, and references between documents.

A file can hold several notices (a compilation of notifications). A *segment* starts at a
government/notification header or at an issuing reference ("এস.আর.ও. নং ১৪৯-আইন/২০২৬/০৪/কাস্টমস।-",
"S.R.O. No. 149-Law/2026/04/Customs"). For each segment we detect:
  issued         first full date near the top ("তারিখ: ... / ০৮ জুন, ২০২৬ খ্রিষ্টাব্দ", "Dated 8 June 2026")
  effective_from date in a sentence saying it takes effect ("... তারিখে কার্যকর হইবে", "shall come into force on ...");
                 "জারির তারিখ হইতে" / "from the date of issue" -> issued
  effective_to   date in a sentence saying it stays valid until ("... পর্যন্ত বলবৎ থাকিবে", "valid till/until ...")
  references     other notices it amends / rescinds / mentions

Nothing here is used to decide answers; it drives retrieval expansion ("always attach amendments")
and conflict display.
"""
from __future__ import annotations

import datetime as dt
import re

from core.textnorm import digits_ascii, norm_identifier
from ingest.validators import BN_MONTHS_GREG, EN_MONTHS

GOV_HEADER = re.compile(r"গণপ্রজাতন্ত্রী\s+বাংলাদেশ\s+সরকার|Government\s+of\s+the\s+People.?s\s+Republic", re.I)
REF_RE = re.compile(
    r"(?P<prefix>এস\.?\s?আর\.?\s?ও\.?|S\.?\s?R\.?\s?O\.?)\s*(?:নং|নম্বর|No\.?|Number)?[\s:\-–—]*"
    r"(?P<ref>[0-9০-৯]{1,4}\s*-?\s*(?:আইন|Law|law|AIN)?\s*/\s*[0-9০-৯]{4}(?:\s*/\s*[0-9০-৯]{1,4})?(?:\s*/\s*[^\s।,;)]+)?)",
    re.I)
AMEND = re.compile(r"সংশোধন|সংশোধিত|প্রতিস্থাপিত|amend|substitut|replace", re.I)
RESCIND = re.compile(r"রহিত|বাতিল|rescind|repeal|supersed|cancel", re.I)
EFFECT = re.compile(r"কার্যকর|বলবৎ\s+হইবে|come\s+into\s+force|take\s+effect|effective", re.I)
UNTIL = re.compile(r"পর্যন্ত\s+(?:বলবৎ|কার্যকর|বহাল)|মেয়াদ|valid\s+(?:till|until|up\s?to)|remain\s+in\s+force\s+(?:till|until)|expire", re.I)
FROM_ISSUE = re.compile(r"জারির\s+তারিখ\s+হইতে|জারীর\s+তারিখ\s+হইতে|অবিলম্বে|অনতিবিলম্বে|from\s+the\s+date\s+of\s+"
                        r"(?:issue|publication)|immediately|forthwith|with\s+immediate\s+effect", re.I)
# a notice's own name: "এই বিধিমালা <title> নামে অভিহিত হইবে", "These rules may be called the <title>."
OWN_TITLE = re.compile(r"এই\s+(?:বিধিমালা|প্রবিধানমালা|প্রজ্ঞাপন|আদেশ|নীতিমালা)\s+(.{5,300}?)\s+নামে\s+অভিহিত"
                       r"|(?:These|This)\s+(?:rules|regulations|order|notification)\s+may\s+be\s+called\s+(?:the\s+)?"
                       r"(.{5,300}?)\s*[.;]", re.I)
SENT_SPLIT = re.compile(r"(?<=[।;])\s+|(?<=[a-z0-9)]\.)\s+(?=[A-Z])")


def ref_key(ref: str) -> str | None:
    """'১৬৯-আইন/২০২৪/২১/কাস্টমস' -> '169/2024' (number/year identifies a notice)."""
    t = digits_ascii(ref)
    m = re.match(r"\s*(\d{1,4})\s*-?\s*(?:আইন|Law|law|AIN)?\s*/\s*(\d{4})", t)
    return f"{int(m.group(1))}/{m.group(2)}" if m else None


def find_refs(text: str) -> list[dict]:
    out = []
    for m in REF_RE.finditer(text):
        k = ref_key(m.group("ref"))
        if k:
            out.append({"key": k, "text": m.group(0).strip(), "start": m.start(), "end": m.end()})
    return out


def find_dates(text: str) -> list[tuple[int, dt.date]]:
    """(position, date) for Gregorian dates written in Bangla or English, or numerically."""
    t = digits_ascii(text)
    out = []
    for name, mo in BN_MONTHS_GREG.items():
        for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s*{name},?\s*(\d{{4}})", t):
            out.append((m.start(), int(m.group(2)), mo, int(m.group(1))))
    for m in re.finditer(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(\d{4})", t):
        mo = EN_MONTHS.get(m.group(2).lower())
        if mo:
            out.append((m.start(), int(m.group(3)), mo, int(m.group(1))))
    for m in re.finditer(r"([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})", t):
        mo = EN_MONTHS.get(m.group(1).lower())
        if mo:
            out.append((m.start(), int(m.group(3)), mo, int(m.group(2))))
    for m in re.finditer(r"(?<![\d/.])(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?![\d/.])", t):
        out.append((m.start(), int(m.group(3)), int(m.group(2)), int(m.group(1))))
    res = []
    for pos, y, mo, d in out:
        try:
            res.append((pos, dt.date(y, mo, d)))
        except ValueError:
            pass
    return sorted(res)


def iso(d: dt.date | None) -> str | None:
    return d.isoformat() if d else None


class SegmentTracker:
    """Feed lines in reading order; tells which segment each line belongs to and collects metadata."""

    def __init__(self):
        self.segments: list[dict] = []
        self._new(None, None)

    def _new(self, page, why):
        self.segments.append({"index": len(self.segments), "ref_key": None, "ref_text": None, "title": None,
                              "issued": None,
                              "effective_from": None, "effective_to": None, "first_page": page, "last_page": page,
                              "start": why, "lines": 0, "references": [], "_from_issue": False,
                              "effective_text": None, "expiry_text": None})

    @property
    def cur(self) -> dict:
        return self.segments[-1]

    def feed(self, text: str, page: int | None) -> int:
        s = self.cur
        stripped = text.strip()
        m = REF_RE.match(stripped)
        # issuing reference: the line starts with it and it is followed by "।-", ".", ":" (not "এর", "of", "dated")
        issuing = bool(m) and re.match(r"\s*[।.|:\-]", stripped[m.end():] or "।") is not None
        if GOV_HEADER.search(stripped) and (s["ref_key"] or s["lines"] > 8):
            self._new(page, "header")
        elif issuing and s["ref_key"]:
            self._new(page, "reference")
        s = self.cur
        if s["first_page"] is None:
            s["first_page"] = page
        s["last_page"] = page
        s["lines"] += 1
        refs = find_refs(stripped)
        if issuing and refs and s["ref_key"] is None and refs[0]["start"] == 0:
            s["ref_key"], s["ref_text"] = refs[0]["key"], refs[0]["text"]
            refs = refs[1:]
        dates = find_dates(stripped)
        if s["issued"] is None and dates and s["lines"] <= 12:
            s["issued"] = dates[-1][1]          # "১৫ জ্যৈষ্ঠ, ১৪৩১ বঙ্গাব্দ/ ২৯ মে, ২০২৪": take the Gregorian one
        if s["title"] is None:
            mt = OWN_TITLE.search(stripped)
            if mt:
                s["title"] = (mt.group(1) or mt.group(2)).strip(" ,।")
        for sent in SENT_SPLIT.split(stripped):
            sd = find_dates(sent)
            if s["issued"]:                       # a notice cannot take effect / expire before it is issued
                sd = [x for x in sd if x[1] >= s["issued"] - dt.timedelta(days=31)]
            if EFFECT.search(sent):
                if FROM_ISSUE.search(sent):
                    s["_from_issue"] = True
                    s["effective_text"] = s["effective_text"] or sent[:300]
                elif sd and s["effective_from"] is None and not UNTIL.search(sent):
                    s["effective_from"] = sd[0][1]
                    s["effective_text"] = sent[:300]
            if UNTIL.search(sent) and sd and s["effective_to"] is None:
                s["effective_to"] = sd[-1][1]
                s["expiry_text"] = sent[:300]
            found = False
            for r in find_refs(sent):
                if r["key"] == s["ref_key"]:
                    continue
                found = True
                kind = "rescinds" if RESCIND.search(sent) else "amends" if AMEND.search(sent) else "refers"
                s["references"].append({"key": r["key"], "text": r["text"], "kind": kind, "page": page,
                                        "evidence": sent[:300]})
            m_act = RESCIND.search(sent) or AMEND.search(sent)
            if not found and m_act and not OWN_TITLE.search(sent):
                # amends / rescinds something named by title: keep the words before the verb; the indexer
                # matches them against the titles of indexed notices
                kind = "rescinds" if RESCIND.search(sent) else "amends"
                s["references"].append({"key": None, "text": None, "kind": kind, "page": page,
                                        "evidence": sent[max(0, m_act.start() - 400):m_act.end() + 40]})
        return s["index"]

    def finish(self) -> list[dict]:
        """Segments that never got their own issuing reference are folded into the one before
        (e.g. a form in an appendix that repeats the government header). self.remap: old -> new index."""
        self.remap = {}
        kept: list[dict] = []
        for s in self.segments:
            if kept and s["ref_key"] is None:
                k = kept[-1]
                self.remap[s["index"]] = k["index"]
                k["lines"] += s["lines"]
                k["last_page"] = s["last_page"] or k["last_page"]
                k["references"] += s["references"]
                k["_from_issue"] = k["_from_issue"] or s["_from_issue"]
                for f in ("effective_from", "effective_to", "effective_text", "expiry_text", "title"):
                    k[f] = k[f] or s[f]
                continue
            self.remap[s["index"]] = s["index"]
            kept.append(s)
        out = []
        for s in kept:
            if s["lines"] == 0:
                continue
            if s["effective_from"] is None and s["_from_issue"]:
                s["effective_from"] = s["issued"]
            # strongest link per referenced notice: rescinds > amends > refers
            best: dict[str, dict] = {}
            rank = {"rescinds": 3, "amends": 2, "refers": 1}
            for r in s["references"]:
                k = r["key"] or "evidence:" + r["evidence"]
                if k not in best or rank[r["kind"]] > rank[best[k]["kind"]]:
                    best[k] = r
            out.append({**{k: v for k, v in s.items() if not k.startswith("_") and k != "references"},
                        "issued": iso(s["issued"]), "effective_from": iso(s["effective_from"]),
                        "effective_to": iso(s["effective_to"]), "references": list(best.values())})
        return out


def identifier_tokens(text: str) -> list[tuple[str, str, str]]:
    """(normalised, original, kind) for identifier-like tokens: codes, notice references, dates."""
    t = digits_ascii(text)
    out = []
    # dd.dd is left out on purpose: "68.05" (a heading) and "62.00" (a price) look the same; headings are
    # found through the 4-digit prefix of the full codes below
    for m in re.finditer(r"(?<![\w.])(\d{4}\.\d{2}(?:\.\d{2,4})?|\d{6,10})(?![\w.])", t):
        orig = text[m.start():m.end()]
        n = norm_identifier(m.group(1))
        out.append((n, orig, "code"))
        if len(n) >= 6:                                   # 87033211 also answers 8703 and 870332
            out.append((n[:4], orig, "prefix"))
            out.append((n[:6], orig, "prefix"))
    for r in find_refs(text):
        out.append(("REF:" + r["key"], r["text"], "ref"))
    for _, d in find_dates(text):
        out.append(("DATE:" + d.isoformat(), d.isoformat(), "date"))
    seen, uniq = set(), []
    for x in out:
        if (x[0], x[2]) not in seen:
            seen.add((x[0], x[2]))
            uniq.append(x)
    return uniq
