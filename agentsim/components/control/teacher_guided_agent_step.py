"""
teacher_guided_agent_step: one teacher-guided student tool-use step.

Pipeline per execution:
    build student prompt (student-visible state only)
    -> student LLM -> parse action -> (force finish on final step)
    -> execute tool -> build teacher prompt (gold metadata)
    -> teacher LLM -> parse evaluation -> render student guidance + leakage check
    -> log full step record -> update done/stop_reason.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from agentsim.components.base import ComponentSpec, ComponentResult, ComponentRegistry
from agentsim.components.control.base import ControlComponent
from agentsim.workflow.context import WorkflowContext

from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig, StudentAction
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever
from agentsim.teacher_guidance.plan_execution import PlanTracker
from agentsim.teacher_guidance.json_utils import parse_student_action, parse_teacher_evaluation
from agentsim.teacher_guidance.prompts import (
    build_student_visible_state,
    build_student_prompt,
    build_teacher_prompt,
)
from agentsim.teacher_guidance.tool_executor import execute_student_tool
from agentsim.teacher_guidance.guidance_policy import render_student_guidance
from agentsim.teacher_guidance.metrics import compute_step_metrics
from agentsim.teacher_guidance.llm_call_log import timed_completion


def _guidance_config(context: WorkflowContext) -> GuidanceConfig:
    return GuidanceConfig.from_mode_config({"guidance": context.metadata.get("guidance", {})})


def get_plan_tracker(context: WorkflowContext) -> Optional[PlanTracker]:
    """Return a per-episode PlanTracker when formal-plan tracking is enabled and a plan
    exists, creating it on first use."""
    pr_config = PlanReviewConfig.from_mode_config(
        {"plan_review": context.metadata.get("plan_review_config", {})}
    )
    if not pr_config.formal_plan:
        return None
    plan = context.metadata.get("revised_plan")
    if not plan:
        return None
    tracker = getattr(context, "_tg_plan_tracker", None)
    if tracker is None:
        tracker = PlanTracker(plan)
        context._tg_plan_tracker = tracker
    return tracker


def get_retriever(context: WorkflowContext) -> HotpotLocalRetriever:
    """Resolve and cache the per-run retriever on the context."""
    retriever = getattr(context, "_tg_retriever", None)
    if retriever is None:
        corpus_path = context.metadata.get("corpus_path")
        if not corpus_path:
            raise ValueError("teacher_guided_agent_step requires mode_config.corpus_path")
        retriever = HotpotLocalRetriever(corpus_path)
        context._tg_retriever = retriever
    return retriever


def _visibility(context: WorkflowContext, retriever: HotpotLocalRetriever) -> Dict[str, Any]:
    gold = context.metadata.get("gold", {}) or {}
    retrieved_docs = context.metadata.get("retrieved_docs", []) or []
    retrieved_titles = [d.get("title", "") for d in retrieved_docs]
    retrieved_doc_ids = context.metadata.get("retrieved_doc_ids", []) or []

    # Hidden spans: gold supporting-fact sentences not yet retrieved.
    hidden_spans: List[str] = []
    for fact in gold.get("supporting_facts", []) or []:
        # best effort: resolve sentence text from gold docs via the retriever
        for doc_id in gold.get("gold_doc_ids", []) or []:
            doc = retriever.get_doc(doc_id)
            if doc and doc.get("title") == fact.get("title"):
                sentences = doc.get("sentences", [])
                sid = fact.get("sent_id", 0)
                if 0 <= sid < len(sentences):
                    hidden_spans.append(sentences[sid])
    return {
        "gold_answer": gold.get("answer", ""),
        "gold_titles": gold.get("supporting_titles", []),
        "gold_doc_ids": gold.get("gold_doc_ids", []),
        "retrieved_titles": retrieved_titles,
        "retrieved_doc_ids": retrieved_doc_ids,
        "hidden_spans": hidden_spans,
    }


def _force_finish_action(context: WorkflowContext) -> StudentAction:
    from agentsim.teacher_guidance.tool_executor import derive_final_answer

    answer = derive_final_answer(context)
    return StudentAction.from_dict(
        {
            "thought": "Budget exhausted; committing best available answer.",
            "decision": {"category": "finish", "parametric_knowledge_used": False},
            "action": {"tool": "finish", "params": {"answer": answer, "citations": []}},
            "new_facts_extracted": [],
        }
    )


@ComponentRegistry.register("teacher_guided_agent_step")
class TeacherGuidedAgentStep(ControlComponent):
    """One teacher-guided student tool-use step."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    @property
    def spec(self) -> ComponentSpec:
        return ComponentSpec(
            name="teacher_guided_agent_step",
            category=self.category,
            description="One teacher-guided student tool-use step",
            input_keys=["query", "metadata.gold", "metadata.retrieval_scope"],
            output_keys=["metadata.teacher_guided_steps", "metadata.last_teacher_guidance_for_student"],
            config_schema={
                "step_index": {"type": "integer", "default": 1},
                "budget": {"type": "integer", "default": 5},
                "force_finish": {"type": "boolean", "default": False},
            },
            requires_llm=True,
        )

    async def execute(self, context: WorkflowContext) -> ComponentResult:
        start = time.time()
        step_started_at = datetime.now(timezone.utc).isoformat()

        if context.metadata.get("done"):
            return ComponentResult(
                success=True,
                data={"skipped": True, "verdict": "FINISH"},
                execution_time_ms=0,
            )

        if not self.llm_client:
            return ComponentResult(success=False, error="LLM client not provided")

        step_index = int(self.config.get("step_index", 1))
        budget = int(self.config.get("budget", 5))
        force_finish = bool(self.config.get("force_finish", False))

        guidance_config = _guidance_config(context)
        retriever = get_retriever(context)

        student_temp = context.metadata.get("student_temperature", 0.2)
        teacher_temp = context.metadata.get("teacher_temperature", 0.1)
        student_model = context.metadata.get("student_model")
        teacher_model = context.metadata.get("teacher_model")

        # --- Student turn ---
        state = build_student_visible_state(context, step_index, budget)
        plan_tracker = get_plan_tracker(context)
        if plan_tracker is not None:
            state["expected_plan_step"] = plan_tracker.expected_step()
        student_prompt = build_student_prompt(state, guidance_config, force_finish)
        student_action, student_raw, parse_info, repair_attempts, student_calls = await self._student_action_with_repair(
            context, student_prompt, student_model, student_temp
        )

        if force_finish and student_action.action.tool != "finish":
            student_action = _force_finish_action(context)

        # Programmatically verify this action against the formal plan (if enabled).
        plan_adherence_info = None
        if plan_tracker is not None:
            plan_adherence_info = plan_tracker.record(student_action.action.tool)
            context.metadata["plan_adherence"] = plan_tracker.adherence()

        tool_observation = execute_student_tool(context, student_action, retriever)

        # --- Teacher turn ---
        gold = context.metadata.get("gold", {}) or {}
        teacher_prompt = build_teacher_prompt(
            state, gold, student_action.to_dict(), tool_observation, guidance_config
        )
        teacher_eval, teacher_raw, teacher_parse, teacher_repair_attempts, teacher_calls = await self._teacher_eval_with_repair(
            context, teacher_prompt, teacher_model, teacher_temp
        )
        teacher_full = {
            "guidance_level": teacher_eval.guidance_level,
            "student_visible": teacher_eval.student_visible,
            "private_diagnosis": teacher_eval.private_diagnosis,
            "teacher_decision": teacher_eval.teacher_decision,
        }

        rendered_guidance, leakage = render_student_guidance(
            teacher_full, guidance_config, _visibility(context, retriever)
        )

        gold_doc_ids = set(gold.get("gold_doc_ids", []) or [])
        step_metrics = compute_step_metrics(
            student_action.to_dict(),
            tool_observation,
            parse_info=parse_info,
            retrieved_doc_ids=set(context.metadata.get("retrieved_doc_ids", []) or []),
            gold_doc_ids=gold_doc_ids,
        )
        if plan_adherence_info is not None:
            step_metrics.update(plan_adherence_info)

        done = self._update_done(context, student_action, teacher_eval.teacher_decision, force_finish)

        step_record = {
            "t": step_index,
            "student_prompt": student_prompt,
            "student_raw": student_raw,
            "student_repair_attempts": repair_attempts,
            "student_calls": student_calls,
            "student_call_ms": sum(c["elapsed_ms"] for c in student_calls),
            "student_action": student_action.to_dict(),
            "tool_observation": tool_observation,
            "teacher_prompt": teacher_prompt,
            "teacher_raw": teacher_raw,
            "teacher_repair_attempts": teacher_repair_attempts,
            "teacher_calls": teacher_calls,
            "teacher_call_ms": sum(c["elapsed_ms"] for c in teacher_calls),
            "teacher_full": teacher_full,
            "student_visible_guidance": rendered_guidance,
            "leakage_check": leakage,
            "metrics": step_metrics,
            "stop_condition": "FINISH" if done else "CONTINUE",
            "step_started_at": step_started_at,
            "step_ended_at": datetime.now(timezone.utc).isoformat(),
            "step_elapsed_ms": (time.time() - start) * 1000,
        }
        context.metadata.setdefault("teacher_guided_steps", []).append(step_record)
        context.metadata["last_teacher_guidance_for_student"] = rendered_guidance

        logger.info(
            f"[TG step {step_index}/{budget}] tool={student_action.action.tool} "
            f"decision={teacher_eval.teacher_decision} done={done}"
        )

        return ComponentResult(
            success=True,
            data={
                "step": step_index,
                "student_action": student_action.to_dict(),
                "tool_observation": tool_observation,
                "student_visible_guidance": rendered_guidance,
                "teacher_decision": teacher_eval.teacher_decision,
                "metrics": step_metrics,
                "verdict": "FINISH" if done else "PROCEED",
            },
            metadata={
                "llm_input": student_prompt,
                "llm_output": student_raw,
                "teacher_llm_input": teacher_prompt,
                "teacher_llm_output": teacher_raw,
                "parameters": {
                    "step_index": step_index,
                    "budget": budget,
                    "force_finish": force_finish,
                    "guidance_level": guidance_config.level,
                },
                "rationale_tag": "TEACHER_GUIDED_STEP",
                "private_reasoning": f"Teacher-guided student step {step_index}",
            },
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _student_action_with_repair(self, context, student_prompt, student_model, student_temp):
        """Call the student; if the action is unparseable or has an invalid tool, re-ask
        it (up to student_max_repair_attempts) with a generic, gold-free correction note.

        Returns ``(student_action, final_raw, parse_info, repair_attempts, calls)``, where
        ``calls`` is a list with one call-log entry per HTTP request made (see
        ``llm_call_log.timed_completion``) -- including failed attempts, so a truncated
        first attempt's raw text/response isn't lost.
        """
        max_repairs = int(context.metadata.get("student_max_repair_attempts", 1))
        max_tokens = context.metadata.get("student_max_tokens", 1200)
        prompt = student_prompt
        attempts = 0
        calls = []
        call_entry, student_raw = await timed_completion(
            self.llm_client, prompt=prompt, model=student_model, temperature=student_temp,
            max_tokens=max_tokens, attempt=1,
        )
        calls.append(call_entry)
        student_action, parse_info = parse_student_action(student_raw)

        while (not parse_info.get("json_valid") or not parse_info.get("action_valid")) and attempts < max_repairs:
            attempts += 1
            problems = ", ".join(parse_info.get("errors", [])) or "the output was not a single valid JSON action"
            correction = (
                f"\n\nYour previous response was not a valid action ({problems}). "
                "Return ONLY one corrected JSON object that matches the action schema exactly: "
                "a valid action.tool from the allowed list, with no text outside the JSON."
            )
            call_entry, student_raw = await timed_completion(
                self.llm_client, prompt=prompt + correction, model=student_model, temperature=student_temp,
                max_tokens=max_tokens, attempt=attempts + 1,
            )
            calls.append(call_entry)
            student_action, parse_info = parse_student_action(student_raw)

        if attempts:
            logger.info(f"[TG step] student action repaired after {attempts} retry(s); valid={parse_info.get('action_valid')}")
        return student_action, student_raw, parse_info, attempts, calls

    async def _teacher_eval_with_repair(self, context, teacher_prompt, teacher_model, teacher_temp):
        """Call the teacher for a per-step evaluation; if the response fails to parse
        into a valid evaluation (often a reasoning model burning its token budget on
        hidden reasoning before emitting JSON, leaving the response truncated), retry up
        to teacher_max_repair_attempts with a corrective note and a larger token budget.

        Returns ``(teacher_eval, final_raw, parse_info, repair_attempts, calls)``, where
        ``calls`` is a list with one call-log entry per HTTP request made (including
        failed attempts).
        """
        max_repairs = int(context.metadata.get("teacher_max_repair_attempts", 1))
        base_tokens = context.metadata.get("teacher_max_tokens", 1000)
        retry_tokens = context.metadata.get("teacher_max_tokens_retry", 2000)

        prompt = teacher_prompt
        attempts = 0
        calls = []
        call_entry, teacher_raw = await timed_completion(
            self.llm_client, prompt=prompt, model=teacher_model, temperature=teacher_temp,
            max_tokens=base_tokens, attempt=1,
        )
        calls.append(call_entry)
        teacher_eval, parse_info = parse_teacher_evaluation(teacher_raw)

        while (not parse_info.get("json_valid") or not parse_info.get("eval_valid")) and attempts < max_repairs:
            attempts += 1
            problems = ", ".join(parse_info.get("errors", [])) or "the output was not one complete, valid JSON object"
            correction = (
                f"\n\nYour previous response was not a valid evaluation ({problems}); it may "
                "have been cut off before the JSON object was complete. Return ONLY one "
                "complete, corrected JSON object matching the schema exactly, with no text "
                "outside the JSON."
            )
            call_entry, teacher_raw = await timed_completion(
                self.llm_client, prompt=prompt + correction, model=teacher_model, temperature=teacher_temp,
                max_tokens=retry_tokens, attempt=attempts + 1,
            )
            calls.append(call_entry)
            teacher_eval, parse_info = parse_teacher_evaluation(teacher_raw)

        if attempts:
            logger.info(
                f"[TG step] teacher evaluation repaired after {attempts} retry(s); "
                f"valid={parse_info.get('eval_valid')}"
            )
        return teacher_eval, teacher_raw, parse_info, attempts, calls

    def _update_done(self, context, student_action, teacher_decision, force_finish) -> bool:
        if force_finish:
            context.metadata["done"] = True
            context.metadata["stop_reason"] = "budget_forced_finish"
            return True
        if student_action.action.tool == "finish" and teacher_decision != "reject_finish":
            context.metadata["done"] = True
            context.metadata["stop_reason"] = "teacher_accept"
            return True
        return False
