"""
Internalize teacher guidance into the student's own reasoning.

The fine-tuned student runs with NO teacher at inference time, yet we want it to reason as
if it had internalized a good teacher's nudges. So for each accepted step we build a
training pair whose:

* input  = the student-visible state with the teacher-guidance block removed (so it
           matches the teacher-free conditions the student will actually see), and
* target = the student's action, with its ``thought`` prefixed by a first-person
           reflection derived from the guidance the student received going into that step.

The reflection is produced deterministically from the *already student-visible* (leakage-
sanitized) guidance -- feedback plus any next-step hint -- with a light second-person ->
first-person rewrite, so the student learns to generate that self-correction itself. It
never introduces gold/hidden information (the guidance was sanitized at generation, and
accepted traces are gold-leak-free), and it references only generic reasoning moves, so it
teaches transferable behaviour rather than question-specific answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agentsim.teacher_guidance.guidance_policy import FALLBACK_FEEDBACK

_GUIDANCE_BLOCK_PREFIX = "Previous teacher guidance:"

# Ordered second-person -> first-person rewrites. Multi-word verb forms come first so
# "you are" -> "I am" (not the ungrammatical "I are") before the bare "you" -> "I" rule.
_PRONOUN_RULES = [
    (re.compile(r"\byou've\b", re.IGNORECASE), "I've"),
    (re.compile(r"\byou're\b", re.IGNORECASE), "I'm"),
    (re.compile(r"\byou are\b", re.IGNORECASE), "I am"),
    (re.compile(r"\byou were\b", re.IGNORECASE), "I was"),
    (re.compile(r"\byourself\b", re.IGNORECASE), "myself"),
    (re.compile(r"\byou\b", re.IGNORECASE), "I"),
]


def _fix_your(text: str) -> str:
    # Preserve capitalization for the possessive: "Your" -> "My", "your" -> "my".
    text = re.sub(r"\bYour\b", "My", text)
    return re.sub(r"\byour\b", "my", text)


def to_first_person(text: str) -> str:
    """Best-effort second-person -> first-person rewrite for internalizing feedback."""
    text = _fix_your(text)
    for pattern, repl in _PRONOUN_RULES:
        text = pattern.sub(repl, text)
    return text


def strip_teacher_guidance_block(student_prompt: str) -> str:
    """Remove the ``Previous teacher guidance: ...`` paragraph so the training input
    matches the teacher-free state the student sees at inference."""
    blocks = student_prompt.split("\n\n")
    kept = [b for b in blocks if not b.lstrip().startswith(_GUIDANCE_BLOCK_PREFIX)]
    return "\n\n".join(kept)


def _hint_sentence(hint: Any) -> str:
    if not isinstance(hint, dict):
        return ""
    bits: List[str] = []
    if hint.get("suggested_focus"):
        bits.append(f"focus on {hint['suggested_focus']}")
    if hint.get("suggested_tool"):
        bits.append(f"use the {hint['suggested_tool']} tool")
    if hint.get("suggested_query"):
        bits.append(f'try searching "{hint["suggested_query"]}"')
    return ("I should " + ", ".join(bits) + ".") if bits else ""


def guidance_to_reflection(guidance: Optional[Dict[str, Any]]) -> str:
    """Turn a rendered student-visible guidance dict into a first-person reflection, or
    "" when there's nothing substantive to internalize (score-only levels, empty or
    fallback feedback)."""
    if not isinstance(guidance, dict):
        return ""
    feedback = (guidance.get("feedback") or "").strip()
    if feedback == FALLBACK_FEEDBACK:
        feedback = ""
    pieces: List[str] = []
    if feedback:
        pieces.append(to_first_person(feedback))
    hint = _hint_sentence(guidance.get("hint"))
    if hint:
        pieces.append(hint)
    if not pieces:
        return ""
    return "Reflecting on my progress so far: " + " ".join(pieces)


def internalize_step(
    student_prompt: str,
    student_thought: str,
    guidance: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """Return ``(teacher_free_prompt, internalized_thought)`` for one step."""
    teacher_free_prompt = strip_teacher_guidance_block(student_prompt)
    reflection = guidance_to_reflection(guidance)
    thought = (student_thought or "").strip()
    internalized = f"{reflection} {thought}".strip() if reflection else thought
    return teacher_free_prompt, internalized
