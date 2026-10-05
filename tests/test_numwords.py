from core.numwords import confirmed_numbers, words_to_number


def test_bangla_words():
    assert words_to_number("ষাট") == 60 and words_to_number("নব্বই") == 90
    assert words_to_number("চব্বিশ") == 24 and words_to_number("আটচল্লিশ") == 48
    assert words_to_number("দুই হাজার") == 2000 and words_to_number("এক লক্ষ পঞ্চাশ হাজার") == 150000
    assert words_to_number("ছয়") == 6 and words_to_number("ঊনত্রিশ") == 29
    assert words_to_number("ষাট দিন") is None


def test_english_words():
    assert words_to_number("forty-eight") == 48 and words_to_number("two thousand") == 2000
    assert words_to_number("one hundred and twenty") == 120


def test_confirmed_pairs():
    t = "“৬০ (ষাট)” সংখ্যার পরিবর্তে “৯০ (নব্বই)” এবং ২,০০০/- (দুই হাজার) টাকা; ৫ (ছয়) ভুল"
    assert confirmed_numbers(t) == {"60", "90", "2000"}
