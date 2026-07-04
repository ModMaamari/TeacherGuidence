"""Tests for teacher-guidance internalization."""

from agentsim.teacher_guidance.guidance_policy import FALLBACK_FEEDBACK
from agentsim.teacher_guidance.sft_internalize import (
    to_first_person,
    strip_teacher_guidance_block,
    guidance_to_reflection,
    internalize_step,
)


def test_to_first_person_handles_verb_agreement():
    assert to_first_person("You are on the right track") == "I am on the right track"
    assert to_first_person("you were close") == "I was close"
    assert to_first_person("You've retrieved the doc") == "I've retrieved the doc"
    assert to_first_person("You correctly found your source") == "I correctly found my source"
    assert to_first_person("Your plan is solid") == "My plan is solid"


def test_strip_teacher_guidance_block_removes_only_that_block():
    prompt = (
        "You are an agent.\n\n"
        "Question: who?\n\n"
        "Previous teacher guidance: {\"score\": 0.9, \"feedback\": \"good\"}\n\n"
        "Return ONLY a JSON object."
    )
    stripped = strip_teacher_guidance_block(prompt)
    assert "Previous teacher guidance" not in stripped
    assert "Question: who?" in stripped
    assert "Return ONLY a JSON object." in stripped


def test_guidance_to_reflection_builds_first_person_note():
    g = {"score": 0.8, "feedback": "You retrieved the right doc; now extract the fact."}
    r = guidance_to_reflection(g)
    assert r.startswith("Reflecting on my progress so far:")
    assert "I retrieved the right doc" in r


def test_guidance_to_reflection_includes_hint():
    g = {"score": 0.5, "feedback": "Keep going.", "hint": {"suggested_tool": "search",
         "suggested_focus": "the birth year"}}
    r = guidance_to_reflection(g)
    assert "focus on the birth year" in r
    assert "use the search tool" in r


def test_guidance_to_reflection_skips_fallback_and_empty():
    assert guidance_to_reflection({"feedback": FALLBACK_FEEDBACK}) == ""
    assert guidance_to_reflection({"score": 0.9}) == ""       # score-only level
    assert guidance_to_reflection({}) == ""
    assert guidance_to_reflection(None) == ""


def test_internalize_step_prefixes_thought_and_strips_prompt():
    prompt = "Question: q\n\nPrevious teacher guidance: {\"feedback\": \"You did well.\"}\n\nSchema."
    tf_prompt, thought = internalize_step(
        prompt, "I will now extract the fact.",
        {"feedback": "You did well.", "score": 1.0},
    )
    assert "Previous teacher guidance" not in tf_prompt
    assert thought.startswith("Reflecting on my progress so far: I did well.")
    assert thought.endswith("I will now extract the fact.")


def test_internalize_step_without_guidance_keeps_thought():
    tf_prompt, thought = internalize_step("Question: q\n\nSchema.", "My original thought.", None)
    assert thought == "My original thought."
