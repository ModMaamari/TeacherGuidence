"""Tests for the leakage checker."""

from agentsim.teacher_guidance.schemas import GuidanceConfig
from agentsim.teacher_guidance.leakage import detect_leakage, sanitize_rendered_guidance


def _cfg(**kw):
    return GuidanceConfig(level=3, leak_policy="strict", **kw)


def test_detect_gold_answer():
    report = detect_leakage(
        {"feedback": "It is Delhi"}, "Delhi", [], [], []
    )
    assert report["gold_answer_leaked"] is True
    assert "Delhi" in report["matched_strings"]


def test_sanitize_removes_gold_answer():
    rendered = {"score": 1.0, "feedback": "The answer is Delhi."}
    visibility = {
        "gold_answer": "Delhi",
        "gold_titles": [],
        "gold_doc_ids": [],
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(rendered, visibility, _cfg())
    assert "Delhi" not in clean["feedback"]
    assert "[answer hidden]" in clean["feedback"]
    assert "[answer hidden]" in report["sanitizations_applied"]


def test_hidden_title_detected_but_not_redacted():
    # Gold titles are fair game (routinely the question's entities): a mention is still
    # *detected* for telemetry, but the text is left intact -- only the gold answer is
    # ever removed.
    rendered = {"feedback": "Look at Oberoi family and The Oberoi Group."}
    visibility = {
        "gold_answer": "Delhi",
        "gold_titles": ["Oberoi family", "The Oberoi Group"],
        "gold_doc_ids": [],
        "retrieved_titles": ["The Oberoi Group"],  # already seen by student
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(rendered, visibility, _cfg())
    # Unretrieved gold title is still flagged for telemetry ...
    assert report["hidden_title_leaked"] is True
    # ... but nothing is redacted: both titles remain visible to the student.
    assert "Oberoi family" in clean["feedback"]
    assert "The Oberoi Group" in clean["feedback"]
    assert "[title hidden]" not in clean["feedback"]
    assert report["sanitizations_applied"] == []


def test_hidden_doc_id_detected_but_not_redacted():
    rendered = {"feedback": "Use q1::doc1 for evidence."}
    visibility = {
        "gold_answer": "X",
        "gold_titles": [],
        "gold_doc_ids": ["q1::doc1"],
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(rendered, visibility, _cfg())
    assert report["hidden_doc_id_leaked"] is True
    assert "q1::doc1" in clean["feedback"]  # not redacted anymore
    assert "[doc hidden]" not in clean["feedback"]


def test_noted_does_not_leak_no():
    # The UI false positive: gold answer "no" must not match the word "noted".
    report = detect_leakage(
        {"feedback": "Your uncertainties are reasonable. Proceed with the plan."},
        "no", [], [], [],
    )
    assert report["gold_answer_leaked"] is False


def test_standalone_no_still_detected_and_sanitized():
    rendered = {"feedback": "No, they are not the same."}
    visibility = {
        "gold_answer": "no",
        "gold_titles": [],
        "gold_doc_ids": [],
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(rendered, visibility, _cfg())
    assert report["gold_answer_leaked"] is True
    assert "[answer hidden]" in clean["feedback"]
    # the standalone "No" is replaced, nothing else mangled
    assert "they are not the same" in clean["feedback"]


def test_word_boundary_does_not_mangle_substrings():
    rendered = {"feedback": "I noted the cannot-do nothing knowledge."}
    visibility = {
        "gold_answer": "no",
        "gold_titles": [],
        "gold_doc_ids": [],
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(rendered, visibility, _cfg())
    assert report["gold_answer_leaked"] is False
    assert clean["feedback"] == "I noted the cannot-do nothing knowledge."


def test_permissive_policy_does_not_sanitize():
    rendered = {"feedback": "The answer is Delhi."}
    visibility = {
        "gold_answer": "Delhi",
        "gold_titles": [],
        "gold_doc_ids": [],
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }
    clean, report = sanitize_rendered_guidance(
        rendered, visibility, GuidanceConfig(level=3, leak_policy="permissive")
    )
    assert clean["feedback"] == "The answer is Delhi."
    assert report["gold_answer_leaked"] is True
