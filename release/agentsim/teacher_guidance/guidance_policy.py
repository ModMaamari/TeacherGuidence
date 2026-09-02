"""
Guidance renderer: converts the full teacher evaluation into the student-visible
object permitted by the configured ``guidance_level``.

The renderer enforces the level in code (the teacher prompt is not trusted), then the
leakage checker sanitizes the result. Levels:

* 0 — binary score only
* 1 — continuous score only
* 2 — score + short outcome feedback
* 3 — score + diagnostic feedback
* 4 — score + diagnostic feedback + next-step hint
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from agentsim.teacher_guidance.schemas import GuidanceConfig
from agentsim.teacher_guidance.leakage import sanitize_rendered_guidance

# Shown in place of a blank feedback string at levels 2-4 — covers both a total
# teacher-response parse failure and a structurally-valid-but-empty response (neither
# of which is distinguishable from "the model legitimately had nothing to say", so we
# treat any blank feedback the same way rather than threading a parse-failure flag
# through this function's signature).
FALLBACK_FEEDBACK = (
    "No detailed feedback is available for this round. Rely on the score above and "
    "continue using your own judgment."
)


def truncate(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "…"


def _sanitize_hint(hint: Any, config: GuidanceConfig) -> Any:
    if not isinstance(hint, dict):
        return None
    clean: Dict[str, Any] = {}
    if config.expose_tool_hint and hint.get("suggested_tool"):
        clean["suggested_tool"] = hint["suggested_tool"]
    # focus is a generic textual nudge; allowed at level 4 next-action exposure
    if config.expose_next_action_hint and hint.get("suggested_focus"):
        clean["suggested_focus"] = hint["suggested_focus"]
    if config.expose_query_hint and hint.get("suggested_query"):
        clean["suggested_query"] = hint["suggested_query"]
    return clean or None


def render_student_guidance(
    teacher_full: Dict[str, Any],
    guidance_config: GuidanceConfig,
    visibility: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Render the student-visible guidance and run leakage sanitization.

    Returns ``(rendered, leakage_report)``.
    """
    teacher_full = teacher_full or {}
    visible = teacher_full.get("student_visible", {}) or {}
    private = teacher_full.get("private_diagnosis", {}) or {}

    score_binary = int(visible.get("score_binary", private.get("score_binary", 0)) or 0)
    score_continuous = float(
        visible.get("score_continuous", private.get("score_continuous", 0.0)) or 0.0
    )
    feedback = visible.get("feedback", "") or ""
    feedback_missing = not feedback.strip()
    max_words = guidance_config.max_feedback_words
    shown_feedback = FALLBACK_FEEDBACK if feedback_missing else truncate(feedback, max_words)

    level = guidance_config.level
    if level == 0:
        rendered: Dict[str, Any] = {"score": score_binary}
    elif level == 1:
        rendered = {"score": round(score_continuous, 3)}
    elif level == 2:
        rendered = {
            "score": round(score_continuous, 3),
            "feedback": shown_feedback,
        }
    elif level == 3:
        rendered = {
            "score": round(score_continuous, 3),
            "feedback": shown_feedback,
        }
    elif level == 4:
        rendered = {
            "score": round(score_continuous, 3),
            "feedback": shown_feedback,
            "hint": _sanitize_hint(visible.get("hint"), guidance_config),
        }
    else:
        raise ValueError(f"Unknown guidance level: {level}")

    rendered, leakage = sanitize_rendered_guidance(rendered, visibility, guidance_config)
    leakage["feedback_fallback_used"] = feedback_missing and level in (2, 3, 4)
    return rendered, leakage


def derive_plan_review_guidance_config(
    step_guidance: GuidanceConfig, review_guidance_level: Any
) -> GuidanceConfig:
    """Return a GuidanceConfig for plan review.

    Reuses the step guidance config but overrides the level with
    ``review_guidance_level`` when it is not None.
    """
    if review_guidance_level is None:
        return step_guidance
    return GuidanceConfig(
        level=int(review_guidance_level),
        name=step_guidance.name,
        score_mode=step_guidance.score_mode,
        max_feedback_words=step_guidance.max_feedback_words,
        expose_next_action_hint=step_guidance.expose_next_action_hint,
        expose_tool_hint=step_guidance.expose_tool_hint,
        expose_query_hint=step_guidance.expose_query_hint,
        expose_doc_title_hint=step_guidance.expose_doc_title_hint,
        expose_gold_answer_hint=step_guidance.expose_gold_answer_hint,
        leak_policy=step_guidance.leak_policy,
    )
