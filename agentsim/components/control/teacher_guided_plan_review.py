"""
teacher_guided_plan_review: optional preflight plan generation, teacher review, and
plan revision performed before the tool-use trajectory begins.

When disabled it short-circuits with verdict PROCEED. When enabled it does not consume
the step budget; it stores a full plan-review record and the revised plan on the
context so each later student step can include the revised plan.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
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
    build_teacher_plan_prompt,
)
from agentsim.teacher_guidance.guidance_policy import (
    render_student_guidance,
    derive_plan_review_guidance_config,
)
from agentsim.teacher_guidance.leakage import sanitize_rendered_guidance
from agentsim.teacher_guidance.plan_review import compute_plan_review_metrics
from agentsim.teacher_guidance.llm_call_log import timed_completion
from agentsim.teacher_guidance.pydantic_schemas import (
    StudentPlanGenerationModel,
    RevisedStudentPlanGenerationModel,
    TeacherPlanReviewModel,
)

# The *GenerationModel variants (not the lenient base models used for post-hoc
# validation) require substantive plan_summary/goal/rationale/stop_condition content
# and at least one step -- an all-optional schema let Ollama's grammar-constrained
# decoding skip straight to a near-empty plan. See pydantic_schemas.py for details.
STUDENT_PLAN_SCHEMA = StudentPlanGenerationModel.model_json_schema()
REVISED_STUDENT_PLAN_SCHEMA = RevisedStudentPlanGenerationModel.model_json_schema()
TEACHER_PLAN_REVIEW_SCHEMA = TeacherPlanReviewModel.model_json_schema()


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
        "question": context.query or "",
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
        plan_review_started_at = datetime.now(timezone.utc).isoformat()
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
        budget = int(context.metadata.get("budget") or 0)
        state = build_student_visible_state(context, step_index=0, budget=budget)
        skip_teacher = bool(context.metadata.get("skip_teacher", False))

        # Teacher-planner mode: the teacher authors the whole plan; the student just
        # follows it. No student drafting/revision.
        if config.planner == "teacher":
            return await self._teacher_planner(
                context, state, gold, review_guidance, config,
                teacher_model, teacher_temp, start, plan_review_started_at,
            )

        # student_use_response_schema=False disables grammar-constrained decoding for
        # student calls (some models collapse under llama.cpp grammar constraints --
        # see teacher_guided_agent_step.execute); the parse/repair path covers them.
        use_student_schema = context.metadata.get("student_use_response_schema", True)

        # 1. Initial plan (student).
        initial_prompt = build_initial_plan_prompt(state, config)
        initial_call, initial_raw = await timed_completion(
            self.llm_client, prompt=initial_prompt, model=student_model, temperature=student_temp,
            max_tokens=context.metadata.get("student_plan_max_tokens", 900),
            response_schema=STUDENT_PLAN_SCHEMA if use_student_schema else None,
        )
        initial_plan_calls = [initial_call]
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

        # No-teacher-guidance ablation: the student's initial plan is used as-is, with
        # zero teacher review/revision calls.
        planning_rounds = 0 if skip_teacher else config.planning_steps
        for round_idx in range(1, planning_rounds + 1):
            review_prompt = build_plan_review_prompt(
                state, gold, current_plan, review_guidance, config
            )
            review_full, review_raw, _, review_repair_attempts, review_calls = await self._teacher_review_with_repair(
                context, review_prompt, teacher_model, teacher_temp
            )
            rendered_feedback, leakage = render_student_guidance(
                review_full, review_guidance, _preflight_visibility(context)
            )
            accepted = review_full.get("teacher_decision") == "accept_plan"
            round_rec = {
                "round": round_idx,
                "teacher_plan_review_prompt": review_prompt,
                "teacher_plan_review_raw": review_raw,
                "teacher_plan_review_full": review_full,
                "teacher_plan_review_repair_attempts": review_repair_attempts,
                "review_calls": review_calls,
                "review_call_ms": sum(c["elapsed_ms"] for c in review_calls),
                "student_visible_plan_feedback": rendered_feedback,
                "leakage_check": leakage,
                "accepted": accepted,
            }
            if accepted:
                round_rec["revised_student_plan"] = current_plan
                rounds.append(round_rec)
                break

            revision_prompt = build_revised_plan_prompt(state, current_plan, rendered_feedback, config)
            revision_call, revision_raw = await timed_completion(
                self.llm_client, prompt=revision_prompt, model=student_model, temperature=student_temp,
                max_tokens=context.metadata.get("student_plan_max_tokens", 900),
                response_schema=REVISED_STUDENT_PLAN_SCHEMA if use_student_schema else None,
            )
            round_rec["revision_calls"] = [revision_call]
            round_rec["revision_call_ms"] = revision_call["elapsed_ms"]
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
            "skip_teacher_review": skip_teacher,
            "planning_steps": config.planning_steps,
            "num_planning_rounds": len(rounds),
            "revision_skipped": revision_skipped,
            "initial_student_plan_prompt": initial_prompt,
            "initial_student_plan_raw": initial_raw,
            "initial_plan_calls": initial_plan_calls,
            "initial_plan_call_ms": sum(c["elapsed_ms"] for c in initial_plan_calls),
            "initial_student_plan": initial_plan,
            "rounds": rounds,
            # Top-level fields reflect the final round (kept for backward compatibility).
            "teacher_plan_review_prompt": review_prompt,
            "teacher_plan_review_raw": review_raw,
            "teacher_plan_review_full": review_full,
            "teacher_plan_review_repair_attempts": rounds[-1]["teacher_plan_review_repair_attempts"] if rounds else 0,
            "student_visible_plan_feedback": rendered_feedback,
            "revised_student_plan_prompt": revision_prompt,
            "revised_student_plan_raw": revision_raw,
            "revised_student_plan": revised_plan,
            "leakage_check": leakage,
            "metrics": metrics,
            "plan_review_started_at": plan_review_started_at,
            "plan_review_ended_at": datetime.now(timezone.utc).isoformat(),
            "plan_review_elapsed_ms": (time.time() - start) * 1000,
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

    async def _teacher_review_with_repair(self, context, review_prompt, teacher_model, teacher_temp):
        """Call the teacher for plan review; if the response fails to parse into a valid
        review object (often a reasoning model burning its token budget on hidden
        reasoning before emitting JSON, leaving the response truncated), retry up to
        teacher_max_repair_attempts with a corrective note and a larger token budget.

        Returns (review_full, final_raw, parse_info, repair_attempts, calls), where
        ``calls`` is a list with one call-log entry per HTTP request made (including
        failed attempts).
        """
        max_repairs = int(context.metadata.get("teacher_max_repair_attempts", 1))
        base_tokens = context.metadata.get("teacher_plan_review_max_tokens", 1000)
        retry_tokens = context.metadata.get("teacher_plan_review_max_tokens_retry", 2000)
        teacher_router = context.metadata.get("teacher_router")

        prompt = review_prompt
        attempts = 0
        calls = []
        call_entry, review_raw = await timed_completion(
            self.llm_client, prompt=prompt, model=teacher_model, temperature=teacher_temp,
            max_tokens=base_tokens, attempt=1, response_schema=TEACHER_PLAN_REVIEW_SCHEMA,
            router_models=teacher_router,
        )
        calls.append(call_entry)
        review_full, parse_info = parse_teacher_plan_review(review_raw)

        while (not parse_info.get("json_valid") or not parse_info.get("review_valid")) and attempts < max_repairs:
            attempts += 1
            problems = ", ".join(parse_info.get("errors", [])) or "the output was not one complete, valid JSON object"
            correction = (
                f"\n\nYour previous response was not a valid plan review ({problems}); it may "
                "have been cut off before the JSON object was complete. Return ONLY one "
                "complete, corrected JSON object matching the schema exactly, with no text "
                "outside the JSON. Do not escape single quotes/apostrophes (') -- only \\\", "
                "\\\\, and control characters need escaping in JSON strings."
            )
            call_entry, review_raw = await timed_completion(
                self.llm_client, prompt=prompt + correction, model=teacher_model, temperature=teacher_temp,
                max_tokens=retry_tokens, attempt=attempts + 1, response_schema=TEACHER_PLAN_REVIEW_SCHEMA,
                router_models=teacher_router,
            )
            calls.append(call_entry)
            review_full, parse_info = parse_teacher_plan_review(review_raw)

        if attempts:
            logger.info(
                f"[TG plan_review] teacher review repaired after {attempts} retry(s); "
                f"valid={parse_info.get('review_valid')}"
            )
        return review_full, review_raw, parse_info, attempts, calls

    async def _teacher_planner(
        self, context, state, gold, review_guidance, config, teacher_model, teacher_temp, start,
        plan_review_started_at,
    ) -> ComponentResult:
        plan_prompt = build_teacher_plan_prompt(state, gold, review_guidance, config)
        plan_call, plan_raw = await timed_completion(
            self.llm_client, prompt=plan_prompt, model=teacher_model, temperature=teacher_temp,
            max_tokens=context.metadata.get("teacher_plan_review_max_tokens", 1000),
            response_schema=STUDENT_PLAN_SCHEMA, router_models=context.metadata.get("teacher_router"),
        )
        plan_calls = [plan_call]
        teacher_plan, _ = parse_student_plan(plan_raw)

        # The teacher-authored plan is student-visible, so sanitize any leaked gold.
        clean_plan, leakage = sanitize_rendered_guidance(
            teacher_plan, _preflight_visibility(context), review_guidance
        )

        steps = clean_plan.get("steps", []) if isinstance(clean_plan, dict) else []
        record = {
            "enabled": True,
            "planner": "teacher",
            "planning_steps": config.planning_steps,
            "num_planning_rounds": 0,
            "revision_skipped": True,
            "initial_student_plan": None,
            "teacher_plan_prompt": plan_prompt,
            "teacher_plan_raw": plan_raw,
            "plan_calls": plan_calls,
            "plan_call_ms": sum(c["elapsed_ms"] for c in plan_calls),
            "teacher_authored_plan": teacher_plan,
            "revised_student_plan": clean_plan,
            "student_visible_plan_feedback": None,
            "leakage_check": leakage,
            "metrics": {
                "planner": "teacher",
                "plan_step_count": len(steps),
                "num_planning_rounds": 0,
                "revisions_done": 0,
                "revision_skipped": True,
            },
            "plan_review_started_at": plan_review_started_at,
            "plan_review_ended_at": datetime.now(timezone.utc).isoformat(),
            "plan_review_elapsed_ms": (time.time() - start) * 1000,
        }
        context.metadata["plan_review"] = record
        context.metadata["revised_plan"] = clean_plan

        logger.info(f"[TG plan_review] planner=teacher steps={len(steps)}")

        return ComponentResult(
            success=True,
            data={"plan_review": record, "verdict": "PROCEED"},
            metadata={
                "llm_input": plan_prompt,
                "llm_output": plan_raw,
                "rationale_tag": "TEACHER_GUIDED_PLAN_REVIEW",
                "private_reasoning": "Teacher-authored plan",
            },
            execution_time_ms=(time.time() - start) * 1000,
        )
