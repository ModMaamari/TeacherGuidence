"""Tests for teacher_guidance.metrics."""

from agentsim.teacher_guidance.metrics import (
    normalize_answer,
    exact_match,
    cover_match,
    f1_score,
    supporting_doc_recall,
    supporting_fact_recall,
    binary_from_continuous,
    compute_final_metrics,
)


def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Delhi, India!") == "delhi india"


def test_exact_match_is_normalized():
    assert exact_match("the Delhi", "Delhi")
    assert not exact_match("Mumbai", "Delhi")


def test_f1_partial_overlap():
    assert round(f1_score("Delhi India", "Delhi"), 4) == 0.6667
    assert f1_score("", "Delhi") == 0.0
    assert f1_score("Delhi", "Delhi") == 1.0


def test_f1_value():
    # precision 1/2, recall 1/1 -> F1 = 2*0.5*1/1.5 = 0.6667
    assert round(f1_score("Delhi city", "Delhi"), 4) == 0.6667


def test_cover_match_answer_plus_explanation():
    # The exact case from the UI: gold "no", student answered "No. <explanation>"
    pred = "No. Roger Donaldson is an Australian-born New Zealand filmmaker, while Andre Cayatte was a French filmmaker."
    assert exact_match(pred, "no") is False
    assert cover_match(pred, "no") is True


def test_cover_match_entity_contained():
    assert cover_match("The answer is The Oberoi Group.", "The Oberoi Group") is True
    assert cover_match("It is located in Delhi today", "Delhi") is True


def test_cover_match_short_gold_must_lead():
    # incidental "no" deep in a wrong answer must NOT count
    assert cover_match("There is no clear winner", "no") is False
    assert cover_match("No, definitely", "no") is True


def test_cover_match_negative():
    assert cover_match("Mumbai", "Delhi") is False
    assert cover_match("", "Delhi") is False


def test_supporting_doc_recall():
    assert supporting_doc_recall({"d1", "d2"}, {"d1", "d3"}) == 0.5
    assert supporting_doc_recall(set(), {"d1"}) == 0.0
    assert supporting_doc_recall({"d1"}, set()) == 0.0


def test_supporting_fact_recall_verbatim():
    corpus = {
        "d0": {"title": "T", "sentences": ["gold sentence here", "other"]},
    }
    gold_facts = [{"title": "T", "sent_id": 0}]
    extracted_ok = [{"doc_id": "d0", "span": "gold sentence here", "fact": "x"}]
    extracted_bad = [{"doc_id": "d0", "span": "paraphrased text", "fact": "x"}]
    assert supporting_fact_recall(extracted_ok, gold_facts, corpus) == 1.0
    assert supporting_fact_recall(extracted_bad, gold_facts, corpus) == 0.0


def test_binary_from_continuous():
    assert binary_from_continuous(0.75) == 1
    assert binary_from_continuous(0.74) == 0


def test_compute_final_metrics():
    corpus = {"d0": {"title": "T", "sentences": ["gold sentence"]}}
    m = compute_final_metrics(
        final_answer="the Delhi",
        gold_answer="Delhi",
        retrieved_doc_ids={"d0"},
        gold_doc_ids={"d0"},
        extracted_spans=[{"doc_id": "d0", "span": "gold sentence", "fact": "x"}],
        gold_facts=[{"title": "T", "sent_id": 0}],
        corpus=corpus,
    )
    assert m["exact_match"] is True
    assert m["f1"] == 1.0
    assert m["supporting_doc_recall"] == 1.0
    assert m["supporting_fact_recall"] == 1.0
