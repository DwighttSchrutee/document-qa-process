from core.textnorm import clean, digits_ascii, language, norm_identifier, search_form


def test_ya_precomposed_matches_ocr_form():
    assert clean("য়") == "য়"
    assert clean("ব্যয়") == clean("ব্যয়")


def test_zero_width_removed():
    assert clean("ক্‍ষ") == "ক্ষ"
    assert clean("র‌্য") == "র্য"


def test_bangla_digits():
    assert digits_ascii("৮৭১৪.১০.৯০") == "8714.10.90"
    assert search_form("SRO ১৬৯-আইন") == "sro 169-আইন"


def test_identifier_forms_are_equal():
    forms = ["8714.10.90", "87141090", "৮৭১৪.১০.৯০", "8714 10 90", "8714-10-90"]
    assert {norm_identifier(f) for f in forms} == {"87141090"}
    assert norm_identifier("১৬৯-আইন/২০২৪/২১/কাস্টমস") == "169আইন202421কাস্টমস"


def test_language():
    assert language("গণপ্রজাতন্ত্রী বাংলাদেশ সরকার") == "bn"
    assert language("Parts and accessories of motorcycles") == "en"
