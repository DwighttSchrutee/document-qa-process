from answer.verify import check, numbers_in, remove_unsupported

P = [{"text": "HSCODE: 87033211 | CD: 25 | SD: 45 | TTI: 1,036.00", "status": "PASS"},
     {"text": "এই বিধিমালা ৬ জুন, ২০২৪ তারিখে কার্যকর হইবে।", "status": "WARN"}]


def test_numbers_extracted_without_citations_and_list_markers():
    assert numbers_in("1. The CD is 25 [1]\n2. SD ৪৫ [2]") == ["25", "৪৫"]


def test_supported_numbers_pass_in_any_digit_form():
    r = check("For 8703.32.11 the CD is 25% and TTI 1036.00 [1]. Effective ০৬ জুন ২০২৪ [2].", P, "q")
    assert r["ok"] and r["unsupported_numbers"] == []
    assert r["cites_unverified"] == [2]


def test_invented_number_is_caught():
    r = check("The SD is 40 [1].", P, "What is the SD for 87033211?")
    assert r["unsupported_numbers"] == ["40"] and not r["ok"]


def test_question_numbers_are_exempt_and_bad_citations_found():
    r = check("HS 99999999 is not in the passages [3].", P, "Is 99999999 listed?")
    assert r["unsupported_numbers"] == [] and r["bad_citations"] == [3]


def test_value_from_an_uncited_passage_is_reported_separately():
    r = check("The rule applies from 6 June 2024 [1].", P, "q")
    assert r["ok"] and "6" in r["numbers_in_uncited_passages"]


def test_remove_unsupported_sentence():
    a = "The CD is 25 [1]. The SD is 40 [1]."
    assert remove_unsupported(a, ["40"]) == "The CD is 25 [1]."


def test_dates_from_the_passage_header_are_supported():
    p = [{"text": "ইহা অবিলম্বে কার্যকর হইবে।", "filename": "nbr_10.pdf", "pages": [1], "ref_key": "213/2024",
          "issued": "2024-05-29", "status": "PASS"}]
    r = check("SRO 213/2024 was issued on 29 May 2024 and took effect immediately [1].", p, "q")
    assert r["ok"], r
