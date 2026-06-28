"""Tests for the guidance renderer."""

from agentsim.teacher_guidance.schemas import GuidanceConfig
from agentsim.teacher_guidance.guidance_policy import (
    render_student_guidance,
    truncate,
    derive_plan_review_guidance_config,
)


TEACHER_FULL = {
    "guidance_level": 4,
    "student_visible": {
        "score_binary": 0,
        "score_continuous": 0.35,
        "feedback": "The step retrieves one relevant document but does not support the second hop.",
        "hint": {"suggested_tool": "search", "suggested_focus": "missing relation"},
    },
    "private_diagnosis": {"main_error": "missing_second_hop"},
}

VISIBILITY = {
    "gold_answer": "Delhi",
    "gold_titles": ["Oberoi family"],
    "gold_doc_ids": ["q1::doc1"],
    "retrieved_titles": [],
    "retrieved_doc_ids": [],
    "hidden_spans": [],
}


def _cfg(level, **kw):
    return GuidanceConfig(level=level, leak_policy="strict", **kw)


def test_level0_binary_only():
    rendered, _ = render_student_guidance(TEACHER_FULL, _cfg(0), VISIBILITY)
    assert rendered == {"score": 0}


def test_level1_continuous_only():
    rendered, _ = render_student_guidance(TEACHER_FULL, _cfg(1), VISIBILITY)
    assert set(rendered.keys()) == {"score"}
    assert rendered["score"] == 0.35


def test_level2_has_feedback_no_hint():
    rendered, _ = render_student_guidance(TEACHER_FULL, _cfg(2), VISIBILITY)
    assert "feedback" in rendered and "hint" not in rendered


def test_level3_diagnostic_no_hint():
    rendered, _ = render_student_guidance(TEACHER_FULL, _cfg(3), VISIBILITY)
    assert "feedback" in rendered and "hint" not in rendered


def test_level4_hint_only_when_allowed():
    # tool hint disabled -> no hint content
    rendered_off, _ = render_student_guidance(TEACHER_FULL, _cfg(4), VISIBILITY)
    assert rendered_off["hint"] is None

    rendered_on, _ = render_student_guidance(
        TEACHER_FULL,
        _cfg(4, expose_tool_hint=True, expose_next_action_hint=True),
        VISIBILITY,
    )
    assert rendered_on["hint"]["suggested_tool"] == "search"
    assert rendered_on["hint"]["suggested_focus"] == "missing relation"


def test_feedback_truncated_to_max_words():
    assert truncate("a b c d e", 3) == "a b c…"
    cfg = _cfg(3, max_feedback_words=5)
    rendered, _ = render_student_guidance(TEACHER_FULL, cfg, VISIBILITY)
    assert len(rendered["feedback"].split()) <= 6  # 5 words + ellipsis token


def test_gold_answer_never_leaks_in_feedback():
    leaky = {
        "student_visible": {
            "score_continuous": 0.9,
            "feedback": "Great, the answer is Delhi.",
        },
        "private_diagnosis": {},
    }
    rendered, report = render_student_guidance(leaky, _cfg(3), VISIBILITY)
    assert "Delhi" not in rendered["feedback"]
    assert "[answer hidden]" in rendered["feedback"]
    assert report["gold_answer_leaked"] is True


def test_derive_plan_review_guidance_overrides_level():
    base = _cfg(0)
    derived = derive_plan_review_guidance_config(base, 3)
    assert derived.level == 3
    same = derive_plan_review_guidance_config(base, None)
    assert same.level == 0
