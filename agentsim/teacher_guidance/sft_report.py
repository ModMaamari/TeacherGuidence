"""
SFT dataset statistics and a hard leakage healthcheck.

Summarizes the training set (tool/step distributions, question-type balance, diversity)
and -- most importantly -- verifies that no gold answer reached the student through the
teacher. Two checks:

* no accepted step is flagged ``gold_answer_leaked`` (the gate should guarantee this), and
* the reflection text we internalize from each step's guidance never contains the gold
  answer (unless that answer is already in the question, which is not a leak).

The second check targets exactly the teacher-derived internalized text, so it does not
false-positive on the student legitimately stating its answer in a finish step. A clean
healthcheck is the guarantee that fine-tuning on this data teaches reasoning, not answers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from agentsim.teacher_guidance.leakage import _contains
from agentsim.teacher_guidance.sft_internalize import guidance_to_reflection
from agentsim.teacher_guidance.dataset_split import question_stratum
from agentsim.teacher_guidance.sft_diversity import diversity_report


def _answer_leaks(text: str, gold: str, question: str) -> bool:
    if not gold or not gold.strip():
        return False
    if _contains(question or "", gold):  # already known from the question -> not a leak
        return False
    return _contains(text or "", gold)


def leakage_healthcheck(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify no gold answer leaked into the teacher-derived internalized text."""
    reflection_leaks: List[Dict[str, Any]] = []
    flagged_steps = 0
    for ep in episodes:
        gold = ep.get("gold_answer", "") or ""
        question = ep.get("query", "") or ""
        steps = ep.get("steps") or []
        for i, s in enumerate(steps):
            if (s.get("leakage_check") or {}).get("gold_answer_leaked"):
                flagged_steps += 1
            guidance_in = steps[i - 1].get("student_visible_guidance") if i > 0 else None
            reflection = guidance_to_reflection(guidance_in)
            if reflection and _answer_leaks(reflection, gold, question):
                reflection_leaks.append(
                    {"qid": ep.get("qid"), "step": s.get("t"), "reflection": reflection[:160]}
                )
    return {
        "reflection_leaks": reflection_leaks,
        "flagged_gold_answer_leaked_steps": flagged_steps,
        "clean": not reflection_leaks and flagged_steps == 0,
    }


def tool_distribution(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(
        (ex.get("metadata") or {}).get("tool")
        for ex in examples
        if (ex.get("metadata") or {}).get("kind") == "action" and (ex.get("metadata") or {}).get("tool")
    )
    return dict(c.most_common())


def stratum_balance(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    unique = {ep.get("qid"): ep for ep in episodes}
    return dict(Counter(question_stratum(ep) for ep in unique.values()).most_common())


def step_count_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = [len(ep.get("steps") or []) for ep in episodes]
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(counts), "max": max(counts), "mean": round(sum(counts) / len(counts), 3)}


def build_report(episodes: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(episodes),
        "unique_questions": len({ep.get("qid") for ep in episodes}),
        "examples": len(examples),
        "example_kinds": dict(Counter((ex.get("metadata") or {}).get("kind") for ex in examples)),
        "tool_distribution": tool_distribution(examples),
        "question_type_balance": stratum_balance(episodes),
        "step_count_distribution": step_count_distribution(episodes),
        "diversity": diversity_report(episodes),
        "leakage_healthcheck": leakage_healthcheck(episodes),
    }
