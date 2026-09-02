"""
Turn accepted episodes into chat-format SFT training examples.

Each accepted trajectory becomes a sequence of teacher-free training examples:

* the student's initial plan (already teacher-free -- it precedes any teacher input), and
* one example per tool-use step, whose user turn is the student-visible state with the
  teacher-guidance block removed and whose assistant turn is the student's action with the
  received guidance internalized into its ``thought`` (see sft_internalize).

The result is what the student should emit at inference with no teacher present. Examples
carry only trajectory content and generic reasoning, never gold/hidden info, so training
on them improves agentic search behaviour without overfitting to specific questions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentsim.teacher_guidance.sft_internalize import internalize_step, strip_teacher_guidance_block

DEFAULT_SYSTEM = (
    "You are an information-seeking retrieval agent. You solve the question using tools, "
    "grounding every fact in retrieved documents and never answering from memory. Reason "
    "internally, then output exactly one JSON action."
)


def _example(system: str, user: str, assistant: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": meta,
    }


def _guidance_into_step(steps: List[Dict[str, Any]], i: int) -> Optional[Dict[str, Any]]:
    """The rendered guidance the student saw going *into* step ``i`` -- i.e. the previous
    step's student-visible guidance (the first step has none)."""
    if i <= 0:
        return None
    return steps[i - 1].get("student_visible_guidance")


def build_examples_from_episode(
    episode: Dict[str, Any], system_message: str = DEFAULT_SYSTEM
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    qid = episode.get("qid")
    student_model = episode.get("student_model", "")
    steps = episode.get("steps", []) or []

    pr = episode.get("plan_review") or {}
    if pr.get("enabled") and pr.get("initial_student_plan_prompt") and pr.get("initial_student_plan_raw"):
        # The initial plan is authored before any teacher input, so it's already
        # teacher-free; we still strip defensively in case a guidance block appears.
        examples.append(_example(
            system_message,
            strip_teacher_guidance_block(pr["initial_student_plan_prompt"]),
            pr["initial_student_plan_raw"],
            {"qid": qid, "kind": "plan", "step": 0, "student_model": student_model},
        ))

    for i, s in enumerate(steps):
        action = s.get("student_action") or {}
        tf_prompt, internal_thought = internalize_step(
            s.get("student_prompt", ""), action.get("thought", ""), _guidance_into_step(steps, i)
        )
        target = dict(action)
        target["thought"] = internal_thought
        examples.append(_example(
            system_message,
            tf_prompt,
            json.dumps(target, ensure_ascii=False),
            {"qid": qid, "kind": "action", "step": s.get("t"),
             "tool": (action.get("action") or {}).get("tool"), "student_model": student_model},
        ))

    return examples


def build_dataset(episodes: List[Dict[str, Any]], system_message: str = DEFAULT_SYSTEM) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep in episodes:
        out.extend(build_examples_from_episode(ep, system_message))
    return out
