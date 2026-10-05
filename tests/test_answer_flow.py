"""Answer pipeline control flow with a fake model and fake retrieval (no GPU, no database)."""
import pytest

import answer.pipeline as P
from answer.prompt import NOT_FOUND
from retrieve.search import Hit

ROW = {"id": 1, "filename": "x.pdf", "sha256": "s", "pages": [5], "bbox": {}, "section_path": ["SRO 202/2024"],
       "clause_no": None, "kind": "table_row", "status": "PASS", "ref_key": "202/2024", "issued": "2024-05-29",
       "effective_from": None, "effective_to": None, "n_tokens": 30, "segment_id": 1,
       "text": "HS Code: 3305.10.00 | Description: Shampoos | Minimum Value USD/Unit: 4.00"}


@pytest.fixture
def fake(monkeypatch):
    calls = {"wide": 0}

    def retrieve(q, u, o):
        return {"hits": [Hit(1, row=dict(ROW), rerank=0.9)], "candidates": [], "queries": [q], "timings": {}}

    def retrieve_wide(q, u, o):
        calls["wide"] += 1
        return retrieve(q, u, o)

    monkeypatch.setattr(P, "retrieve", retrieve)
    monkeypatch.setattr(P, "retrieve_wide", retrieve_wide)
    monkeypatch.setattr(P, "understand", lambda q, use_llm=False: {"queries": [q], "identifiers": [], "language": "en"})

    def script(*answers):
        it = iter(answers)

        def stream(msgs, **kw):
            calls.setdefault("prompts", []).append(msgs[-1]["content"])
            yield {"type": "token", "text": next(it)}
            yield {"type": "stats", "seconds": 0.0}
        monkeypatch.setattr(P.llm, "chat_stream", stream)
    calls["script"] = script
    return calls


def run(q="What is the minimum value for 3305.10.00?"):
    evs = list(P.ask(q, log=False))
    return evs, evs[-1]


def test_correct_answer_passes(fake):
    fake["script"]("The minimum value for shampoos is 4.00 USD per unit [1].")
    evs, done = run()
    assert done["answer"].startswith("The minimum value") and not done["warnings"]
    assert "reset" not in [e["type"] for e in evs]


def test_invented_value_regenerated_with_feedback(fake):
    fake["script"]("It is 4.50 [1].", "It is 4.00 [1].")
    evs, done = run()
    assert done["answer"] == "It is 4.00 [1]."
    assert "4.50" in fake["prompts"][1] and "do not appear" in fake["prompts"][1]


def test_value_still_invented_is_removed_with_warning(fake):
    fake["script"]("It is 4.50 [1].", "Shampoos are listed [1]. It is 4.75 [1].")
    _, done = run()
    assert "4.75" not in done["answer"] and "Shampoos are listed [1]." in done["answer"]
    assert any("4.75" in w for w in done["warnings"])


def test_not_found_triggers_second_round_then_reports_closest(fake):
    fake["script"](f"{NOT_FOUND} Nothing about helmets.", f"{NOT_FOUND} Still nothing about helmets.")
    evs, done = run("Minimum value for helmets?")
    assert fake["wide"] == 1 and done["not_found"]
    assert done["closest"] and done["closest"][0]["filename"] == "x.pdf"
    assert [e["round"] for e in evs if e["type"] == "sources"] == [1, 2]


def test_missing_citation_regenerates(fake):
    fake["script"]("The value is 4.00.", "The value is 4.00 [1].")
    _, done = run()
    assert done["answer"].endswith("[1].")
