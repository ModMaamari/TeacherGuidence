"""
teacher_guided_plan_review: optional preflight plan generation, teacher review, and
plan revision performed before the tool-use trajectory begins.

When disabled it short-circuits with verdict PROCEED. When enabled it does not consume
the step budget; it stores a full plan-review record and the revised plan on the
context so each later student step can include the revised plan.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.control.base import ControlComponent
from agentsim.workflow.context import WorkflowContext

from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig
from agentsim.teacher_guidance.json_utils import (
    parse_student_plan,
    parse_teacher_plan_review,
    parse_revised_plan,
)
from agentsim.teacher_guidance.prompts import (
    build_student_visible_state,
    build_initial_plan_prompt,
    build_plan_review_prompt,
    build_revised_plan_prompt,
)
from agentsim.teacher_guidance.guidance_policy import (
    render_student_guidance,
    derive_plan_review_guidance_config,
)
from agentsim.teacher_guidance.plan_review import compute_plan_review_metrics


def _guidance_config(context: WorkflowContext) -> GuidanceConfig:
    return GuidanceConfig.from_mode_config({"guidance": context.metadata.get("guidance", {})})


def _plan_review_config(context: WorkflowContext) -> PlanReviewConfig:
    return PlanReviewConfig.from_mode_config(
        {"plan_review": context.metadata.get("plan_review_config", {})}
    )


def _preflight_visibility(context: WorkflowContext) -> Dict[str, Any]:
    gold = context.metadata.get("gold", {}) or {}
    # Nothing retrieved yet, so all gold values are hidden.
    return {
        "gold_answer": gold.get("answer", ""),
        "gold_titles": gold.get("supporting_titles", []),
        "gold_doc_ids": gold.get("gold_doc_ids", []),
        "retrieved_titles": [],
        "retrieved_doc_ids": [],
        "hidden_spans": [],
    }


@ComponentRegistry.register("teacher_guided_plan_review")
class TeacherGuidedPlanReview(ControlComponent):
    """Optional preflight plan generation, teacher review, and plan revision."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="teacher_guided_plan_review",
            category=self.category,
            description="Optional preflight plan generation, teacher review, and plan revision",
            input_keys=["query", "metadata.gold", "metadata.retrieval_scope"],
            output_keys=["metadata.plan_review", "metadata.revised_plan"],
            config_schema={"enabled": {"type": "boolean", "default": False}},
            requires_llm=True,
        )

    async def execute(self, context: WorkflowContext) -> ComponentResult:
        start = time.time()
        config = _plan_review_config(context)
        # The component config flag can also enable it directly in the workflow YAML.
        enabled = config.enabled or bool(self.config.get("enabled", False))

        if not enabled:
            context.metadata["plan_review"] = {"enabled": False}
            return ComponentResult(success=True, data={"skipped": True, "verdict": "PROCEED"})

        if not self.llm_client:
            return ComponentResult(success=False, error="LLM client not provided")

        gold = context.metadata.get("gold", {}) or {}
        step_guidance = _guidance_config(context)
        review_guidance = derive_plan_review_guidance_config(
            step_guidance, config.review_guidance_level
        )

        student_model = context.metadata.get("student_model")
        teacher_model = context.metadata.get("teacher_model")
        student_temp = context.metadata.get("student_temperature", 0.2)
        teacher_temp = context.metadata.get("teacher_temperature", 0.1)
        state = build_student_visible_state(context, step_index=0, budget=0)

        # 1. Initial plan (student).
        initial_prompt = build_initial_plan_prompt(state, config)
        initial_raw = await self.llm_client.get_completion(
            prompt=initial_prompt, model=student_model, temperature=student_temp,
            max_tokens=context.metadata.get("student_plan_max_tokens", 900),
        )
        initial_plan, _ = parse_student_plan(initial_raw)

        # 2. Planning loop: teacher review -> student revision, up to planning_steps
        #    rounds or until the teacher accepts the plan (whichever comes first).
        current_plan = dict(initial_plan) if isinstance(initial_plan, dict) else initial_plan
        rounds = []
        review_full = {}
        rendered_feedback = {}
        leakage = {}
        review_prompt = review_raw = revision_prompt = revision_raw = None
        revisions_done = 0

        for round_idx in range(1, config.planning_steps + 1):
            review_prompt = build_plan_review_prompt(
                state, gold, current_plan, review_guidance, config
            )
            review_raw = await self.llm_client.get_completion(
                prompt=review_prompt, model=teacher_model, temperature=teacher_temp,
                max_tokens=context.metadata.get("teacher_plan_review_max_tokens", 1000),
            )
            review_full, _ = parse_teacher_plan_review(review_raw)
            rendered_feedback, leakage = render_student_guidance(
                review_full, review_guidance, _preflight_visibility(context)
            )
            accepted = review_full.get("teacher_decision") == "accept_plan"
            round_rec = {
                "round": round_idx,
                "teacher_plan_review_prompt": review_prompt,
                "teacher_plan_review_raw": review_raw,
                "teacher_plan_review_full": review_full,
                "student_visible_plan_feedback": rendered_feedback,
                "leakage_check": leakage,
                "accepted": accepted,
            }
            if accepted:
                round_rec["revised_student_plan"] = current_plan
                rounds.append(round_rec)
                break

            revision_prompt = build_revised_plan_prompt(state, current_plan, rendered_feedback, config)
            revision_raw = await self.llm_client.get_completion(
                prompt=revision_prompt, model=student_model, temperature=student_temp,
                max_tokens=context.metadata.get("student_plan_max_tokens", 900),
            )
            revised_plan, _ = parse_revised_plan(revision_raw)
            round_rec["revised_student_plan_prompt"] = revision_prompt
            round_rec["revised_student_plan_raw"] = revision_raw
            round_rec["revised_student_plan"] = revised_plan
            rounds.append(round_rec)
            current_plan = revised_plan
            revisions_done += 1

        revised_plan = current_plan
        revision_skipped = revisions_done == 0

        metrics = compute_plan_review_metrics(initial_plan, revised_plan, review_full)
        metrics["num_planning_rounds"] = len(rounds)
        metrics["revisions_done"] = revisions_done
        metrics["revision_skipped"] = revision_skipped

        record = {
            "enabled": True,
            "planner": "student",
            "planning_steps": config.planning_steps,
            "num_planning_rounds": len(rounds),
            "revision_skipped": revision_skipped,
            "initial_student_plan_prompt": initial_prompt,
            "initial_student_plan_raw": initial_raw,
            "initial_student_plan": initial_plan,
            "rounds": rounds,
            # Top-level fields reflect the final round (kept for backward compatibility).
            "teacher_plan_review_prompt": review_prompt,
            "teacher_plan_review_raw": review_raw,
            "teacher_plan_review_full": review_full,
            "student_visible_plan_feedback": rendered_feedback,
            "revised_student_plan_prompt": revision_prompt,
            "revised_student_plan_raw": revision_raw,
            "revised_student_plan": revised_plan,
            "leakage_check": leakage,
            "metrics": metrics,
        }
        context.metadata["plan_review"] = record
        context.metadata["revised_plan"] = revised_plan

        logger.info(
            f"[TG plan_review] planner=student rounds={len(rounds)} "
            f"final_decision={review_full.get('teacher_decision')} "
            f"revision_skipped={revision_skipped}"
        )

        return ComponentResult(
            success=True,
            data={"plan_review": record, "verdict": "PROCEED"},
            metadata={
                "llm_input": initial_prompt,
                "llm_output": initial_raw,
                "rationale_tag": "TEACHER_GUIDED_PLAN_REVIEW",
                "private_reasoning": "Preflight plan review",
            },
            execution_time_ms=(time.time() - start) * 1000,
        )
