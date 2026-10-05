"""Retrieval pieces that need no database: identifier keys, fusion order, context selection."""
from answer.understand import regex_identifiers
from retrieve.search import Hit, Options, _rrf, identifier_keys, select_context


def test_identifier_forms_hit_the_same_key():
    for form in ["8714.10.90", "87141090", "৮৭১৪.১০.৯০", "HS 8714.10.90"]:
        assert identifier_keys([form]) == {"87141090": "code"}, form
    assert identifier_keys(["8714"]) == {"8714": "prefix"}


def test_notice_references_and_dates():
    assert identifier_keys(["SRO 169/2024"]) == {"REF:169/2024": "ref"}
    assert identifier_keys(["এস.আর.ও. নং ১৬৯-আইন/২০২৪/২১/কাস্টমস"]) == {"REF:169/2024": "ref"}
    assert "DATE:2024-05-29" in identifier_keys(["29 May, 2024"])


def test_regex_identifiers_from_question():
    ids = regex_identifiers("What is the CD for ৮৭১৪.১০.৯০ under SRO 169/2024?")
    assert "৮৭১৪.১০.৯০" in ids and any("169/2024" in x for x in ids)


def test_rrf_rewards_agreement():
    hits = {}
    _rrf({"a": [1, 2, 3], "b": [2, 4]}, hits)
    assert max(hits.values(), key=lambda h: h.score).id == 2


def row(i, tokens=100, ref=None):
    return {"id": i, "n_tokens": tokens, "ref_key": ref, "section_path": [], "text": f"t{i}"}


def test_context_keeps_exact_hits_and_budget():
    cands = [Hit(i, row=row(i, 3000 if i == 5 else 100)) for i in range(1, 12)]
    cands[9].exact = "code"                     # id 10: exact code hit, ranked low by fusion
    for h in cands:
        h.rerank = 1.0 / h.id
    picked = select_context("q", cands, Options(rerank=False, top_k=4, context_tokens=1000))
    ids = [h.id for h in picked]
    assert ids[0] == 10 and len(ids) == 4 and 5 not in ids


def test_amendments_follow_their_notice():
    a = Hit(1, row=row(1, ref="169/2024"))
    b = Hit(2, row=row(2), added_for="amends 169/2024")
    c = Hit(3, row=row(3))
    picked = select_context("q", [a, c, b], Options(rerank=False, top_k=2))
    assert [h.id for h in picked] == [1, 3, 2]


def test_clause_numbers_from_question():
    from retrieve.search import clause_numbers
    assert clause_numbers("পোশাক শিল্পের বিধিমালার বিধি ১২ এর উপ-বিধি ১ কী বলে?") == ["12"]
    assert clause_numbers("rule 12, sub-rule 3") == ["12"]
    assert clause_numbers("What does rule 7 of the General Bond Rules say?") == ["7"]
    assert clause_numbers("Minimum value for HS 3305.10.00?") == []


def test_keyword_split_keeps_bangla_words_whole():
    import re
    from core.textnorm import search_form
    q = search_form("ইউডি যাচাই কমিটি কতদিন পরপর সভা করবে?")
    words = [w for w in re.split(r"[\s,;:!?()\[\]{}\"'“”‘’।|]+", q) if len(w) > 1]
    assert words[:3] == ["ইউডি", "যাচাই", "কমিটি"]


def test_notice_ranges_and_lists():
    from retrieve.search import notice_keys
    q = "When did SRO 213/2024 take effect, and how does this differ from SRO 210-212?"
    assert set(notice_keys(q)) == {"REF:213/2024", "REF:210/2024", "REF:211/2024", "REF:212/2024"}
    assert set(notice_keys("SROs 169, 202 and 210 of 2024")) >= {"REFNUM:169", "REFNUM:202", "REFNUM:210"}
    assert set(notice_keys("এস.আর.ও. নং ১৬৯/২০২৪")) == {"REF:169/2024"}
    assert identifier_keys(["210-212"]) == {}
