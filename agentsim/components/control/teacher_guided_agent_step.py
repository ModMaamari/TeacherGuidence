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

from agentsim.teacher_guidance.schemas import GuidanceConfig, PlanReviewConfig, StudentAction, TeacherEvaluation
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever
from agentsim.teacher_guidance.plan_execution import PlanTracker
from agentsim.teacher_guidance.json_utils import parse_student_action, parse_teacher_evaluation
from agentsim.teacher_guidance.prompts import (
    build_student_visible_state,
    build_student_prompt,
    build_teacher_prompt,
    build_forced_answer_prompt,
    build_wiki_update_prompt,
)
from agentsim.teacher_guidance.tool_executor import execute_student_tool
from agentsim.teacher_guidance.guidance_policy import render_student_guidance
from agentsim.teacher_guidance.metrics import compute_step_metrics
from agentsim.teacher_guidance.llm_call_log import timed_completion
from agentsim.teacher_guidance.pydantic_schemas import (
    StudentActionGenerationModel,
    StudentActionWikiGenerationModel,
    StudentFinishActionGenerationModel,
    TeacherEvaluationModel,
)

# StudentActionGenerationModel (not the lenient StudentActionModel used for post-hoc
# validation) requires substantive 'thought' content -- see pydantic_schemas.py for why
# an all-optional schema backfires under Ollama's grammar-constrained decoding.
STUDENT_ACTION_SCHEMA = StudentActionGenerationModel.model_json_schema()
# Same grammar extended with wiki_read/wiki_write, used when the run has wiki_enabled.
STUDENT_ACTION_WIKI_SCHEMA = StudentActionWikiGenerationModel.model_json_schema()
# Finish-only schema used on the final force-finish step so the model commits a real
# answer from its context instead of searching again (which yielded answer "unknown").
STUDENT_FINISH_ACTION_SCHEMA = StudentFinishActionGenerationModel.model_json_schema()
TEACHER_EVALUATION_SCHEMA = TeacherEvaluationModel.model_json_schema()


def _guidance_config(context: WorkflowContext) -> GuidanceConfig:
    return GuidanceConfig.from_mode_config({"guidance": context.metadata.get("guidance", {})})


def _extract_teacher_final_judgment(private_diagnosis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize the teacher's final-answer verdict from its private_diagnosis into
    ``{"correct": 0|1, "score": float}`` (or ``None`` when the teacher didn't return it).

    Tolerant of the shapes a model may emit: bool/int/float/str for the binary flag and
    any 0-1-ish value for the continuous score. If only one is present the other is
    derived (score>=0.5 -> correct; correct -> score 1.0/0.0)."""
    pd = private_diagnosis or {}
    if "final_answer_correct" not in pd and "final_answer_score" not in pd:
        return None

    def _to_binary(v: Any) -> Optional[int]:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return 1 if v >= 0.5 else 0
        if isinstance(v, str):
            return 1 if v.strip().lower() in {"1", "true", "yes", "correct"} else 0
        return None

    binary = _to_binary(pd.get("final_answer_correct"))
    score: Optional[float] = None
    raw_score = pd.get("final_answer_score")
    if raw_score is not None:
        try:
            score = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            score = None

    if binary is None and score is not None:
        binary = 1 if score >= 0.5 else 0
    if score is None and binary is not None:
        score = float(binary)
    if binary is None and score is None:
        return None
    return {"correct": binary, "score": score}


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
        "question": context.query or "",
        "gold_titles": gold.get("supporting_titles", []),
        "gold_doc_ids": gold.get("gold_doc_ids", []),
        "retrieved_titles": retrieved_titles,
        "retrieved_doc_ids": retrieved_doc_ids,
        "hidden_spans": hidden_spans,
    }


def _build_finish_action(answer: str, thought: str, citations: List[Any]) -> StudentAction:
    return StudentAction.from_dict(
        {
            "thought": thought,
            "decision": {"category": "finish", "parametric_knowledge_used": False},
            "action": {"tool": "finish", "params": {"answer": answer, "citations": citations}},
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
        # On the final step, constrain generation to a finish-only schema so the model
        # commits an actual answer from its context rather than searching again.
        # student_use_response_schema=False disables the all-tools grammar for normal
        # steps: some models (observed: MiniCPM5-1B) collapse to degenerate shortest-path
        # actions under the big discriminated-union grammar while producing valid,
        # sensible JSON unconstrained -- the prompt + parse/repair path handles the rest.
        # The force-finish step KEEPS the small finish-only schema either way: without it
        # the final step can come back malformed/non-finish and degrade to answer
        # "unknown", and the tiny finish grammar doesn't trigger the collapse.
        if force_finish:
            action_schema = STUDENT_FINISH_ACTION_SCHEMA
        elif context.metadata.get("student_use_response_schema", True):
            action_schema = (
                STUDENT_ACTION_WIKI_SCHEMA
                if context.metadata.get("wiki_enabled")
                else STUDENT_ACTION_SCHEMA
            )
        else:
            action_schema = None
        student_action, student_raw, parse_info, repair_attempts, student_calls = await self._student_action_with_repair(
            context, student_prompt, student_model, student_temp, response_schema=action_schema
        )

        if force_finish:
            student_action, forced_call = await self._resolve_forced_finish(
                context, state, student_action, student_model, student_temp
            )
            if forced_call is not None:
                student_calls.append(forced_call)

        # Programmatically verify this action against the formal plan (if enabled).
        plan_adherence_info = None
        if plan_tracker is not None:
            plan_adherence_info = plan_tracker.record(student_action.action.tool)
            context.metadata["plan_adherence"] = plan_tracker.adherence()

        tool_observation = execute_student_tool(context, student_action, retriever)

        # --- Teacher turn ---
        skip_teacher = bool(context.metadata.get("skip_teacher", False))
        gold = context.metadata.get("gold", {}) or {}
        if skip_teacher:
            # No-teacher-guidance ablation: the student's own 'finish' choice is the only
            # stop signal (never "reject_finish"), and there's nothing to show or leak.
            teacher_prompt = ""
            teacher_raw = ""
            teacher_repair_attempts = 0
            teacher_calls: List[Dict[str, Any]] = []
            teacher_eval = TeacherEvaluation(
                guidance_level=0, student_visible={}, private_diagnosis={}, teacher_decision="continue"
            )
            rendered_guidance, leakage = {}, {}
        else:
            is_final_answer = student_action.action.tool == "finish"
            teacher_prompt = build_teacher_prompt(
                state, gold, student_action.to_dict(), tool_observation, guidance_config,
                student_raw=student_raw, student_action_valid=bool(parse_info.get("action_valid")),
                is_final_answer=is_final_answer,
            )
            teacher_eval, teacher_raw, teacher_parse, teacher_repair_attempts, teacher_calls = await self._teacher_eval_with_repair(
                context, teacher_prompt, teacher_model, teacher_temp
            )
            if is_final_answer:
                judgment = _extract_teacher_final_judgment(teacher_eval.private_diagnosis)
                if judgment is not None:
                    context.metadata["teacher_final_judgment"] = judgment
            rendered_guidance, leakage = render_student_guidance(
                {
                    "guidance_level": teacher_eval.guidance_level,
                    "student_visible": teacher_eval.student_visible,
                    "private_diagnosis": teacher_eval.private_diagnosis,
                    "teacher_decision": teacher_eval.teacher_decision,
                },
                guidance_config, _visibility(context, retriever)
            )

        teacher_full = {
            "guidance_level": teacher_eval.guidance_level,
            "student_visible": teacher_eval.student_visible,
            "private_diagnosis": teacher_eval.private_diagnosis,
            "teacher_decision": teacher_eval.teacher_decision,
        }

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

        # Auto wiki mode: after every (non-final) step, one dedicated call asks the
        # student to rewrite wiki.md from what it just did/observed. The read half is
        # implicit -- build_student_visible_state puts the wiki into every step's prompt.
        wiki_update_call = None
        wiki_auto = bool(context.metadata.get("wiki_enabled")) and \
            context.metadata.get("wiki_mode", "tools") == "auto"
        if wiki_auto and not done:
            wiki_update_call = await self._update_wiki(
                context, state, student_action, tool_observation, student_model, student_temp
            )

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
            "teacher_skipped": skip_teacher,
            "teacher_full": teacher_full,
            "student_visible_guidance": rendered_guidance,
            "leakage_check": leakage,
            "metrics": step_metrics,
            "stop_condition": "FINISH" if done else "CONTINUE",
            "step_started_at": step_started_at,
            "step_ended_at": datetime.now(timezone.utc).isoformat(),
            "step_elapsed_ms": (time.time() - start) * 1000,
        }
        if wiki_auto:
            step_record["wiki_update_call"] = wiki_update_call
            step_record["wiki_after"] = context.metadata.get("wiki", "")
            step_record["wiki_edit_ops"] = context.metadata.pop("wiki_edit_ops", None)
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

    async def _resolve_forced_finish(self, context, state, student_action, student_model, student_temp):
        """Guarantee the forced-finish step commits a real answer, never "unknown".

        Priority: (1) the student's own ``finish`` answer if it produced one this step,
        (2) a previously-committed answer / synthesized draft / extracted facts (via
        ``derive_final_answer``), (3) a last-ditch free-text answer-extraction call to the
        student model over its retrieved context. The last tier is what fixes the
        "unknown" regression: when the student burned the final step on another search and
        nothing was committed earlier, we still turn its evidence into a best-effort
        answer instead of fabricating "unknown".

        Returns ``(finish_action, forced_call_entry_or_None)``; the call entry (if a
        fallback LLM call was made) is appended to the step's ``student_calls`` so the
        extra request is logged.
        """
        from agentsim.teacher_guidance.tool_executor import derive_final_answer, clean_forced_answer

        gave_finish = student_action.action.tool == "finish"
        params = student_action.action.params if gave_finish else None
        citations = (params or {}).get("citations", []) or []
        answer = derive_final_answer(context, params)

        # The student produced a usable finish this step -- keep its action untouched
        # (thought, decision, any new_facts_extracted) so nothing downstream regresses.
        if gave_finish and answer != "unknown":
            return student_action, None

        forced_call = None
        if answer == "unknown":
            try:
                forced_call, raw = await timed_completion(
                    self.llm_client,
                    prompt=build_forced_answer_prompt(state),
                    model=student_model,
                    temperature=student_temp,
                    max_tokens=context.metadata.get("student_max_tokens", 1200),
                    attempt=1,
                )
                cleaned = clean_forced_answer(raw)
                if cleaned:
                    answer = cleaned
            except Exception as exc:  # never let the fallback crash the episode
                logger.warning(f"[TG force-finish] answer-extraction fallback failed: {exc}")

        thought = (
            student_action.thought
            if gave_finish and student_action.thought
            else "Budget exhausted; committing best available answer from retrieved evidence."
        )
        return _build_finish_action(answer, thought, citations), forced_call

    async def _update_wiki(
        self, context, state, student_action, tool_observation, student_model, student_temp
    ):
        """Auto-wiki write half: one short call emitting surgical edit commands
        (ADD/EDIT/DEL/ANSWER/NEXT/KEEP) that ``wiki.apply_wiki_edits`` applies
        deterministically -- the model only generates the changed lines, never the
        whole file. Applied/ignored ops land in ``metadata['wiki_edit_ops']`` for the
        step record.

        Never crashes the episode -- on any failure the wiki simply keeps its previous
        content. Returns the call-log entry (or None) so the step record shows the call."""
        from agentsim.teacher_guidance.wiki import apply_wiki_edits

        try:
            call_entry, raw = await timed_completion(
                self.llm_client,
                prompt=build_wiki_update_prompt(state, student_action.to_dict(), tool_observation),
                model=student_model,
                temperature=student_temp,
                max_tokens=context.metadata.get("wiki_max_tokens", 400),
                attempt=1,
            )
            new_wiki, applied, ignored = apply_wiki_edits(context.metadata.get("wiki", ""), raw)
            context.metadata["wiki"] = new_wiki
            context.metadata["wiki_edit_ops"] = {"applied": applied, "ignored": ignored}
            return call_entry
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[TG wiki] auto wiki update failed (keeping previous wiki): {exc}")
            return None

    async def _student_action_with_repair(
        self, context, student_prompt, student_model, student_temp, response_schema=None
    ):
        """Call the student; if the action is unparseable or has an invalid tool, re-ask
        it (up to student_max_repair_attempts) with a generic, gold-free correction note.

        ``response_schema`` is the grammar-constraining JSON schema handed to Ollama
        (all-tools on normal steps, finish-only on the force-finish step), or ``None`` to
        disable constrained decoding entirely (student_use_response_schema=False -- see
        ``execute``); the caller always passes it explicitly.

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
            max_tokens=max_tokens, attempt=1, response_schema=response_schema,
        )
        calls.append(call_entry)
        student_action, parse_info = parse_student_action(student_raw)

        while (not parse_info.get("json_valid") or not parse_info.get("action_valid")) and attempts < max_repairs:
            attempts += 1
            problems = ", ".join(parse_info.get("errors", [])) or "the output was not a single valid JSON action"
            correction = (
                f"\n\nYour previous response was not a valid action ({problems}). "
                "Return ONLY one corrected JSON object that matches the action schema exactly: "
                "a valid action.tool from the allowed list, with no text outside the JSON. "
                "Do not escape single quotes/apostrophes (') -- only \\\", \\\\, and control "
                "characters need escaping in JSON strings; \\' is not valid JSON and will fail."
            )
            call_entry, student_raw = await timed_completion(
                self.llm_client, prompt=prompt + correction, model=student_model, temperature=student_temp,
                max_tokens=max_tokens, attempt=attempts + 1, response_schema=response_schema,
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
        teacher_router = context.metadata.get("teacher_router")

        prompt = teacher_prompt
        attempts = 0
        calls = []
        call_entry, teacher_raw = await timed_completion(
            self.llm_client, prompt=prompt, model=teacher_model, temperature=teacher_temp,
            max_tokens=base_tokens, attempt=1, response_schema=TEACHER_EVALUATION_SCHEMA,
            router_models=teacher_router,
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
                "outside the JSON. Do not escape single quotes/apostrophes (') -- only \\\", "
                "\\\\, and control characters need escaping in JSON strings."
            )
            call_entry, teacher_raw = await timed_completion(
                self.llm_client, prompt=prompt + correction, model=teacher_model, temperature=teacher_temp,
                max_tokens=retry_tokens, attempt=attempts + 1, response_schema=TEACHER_EVALUATION_SCHEMA,
                router_models=teacher_router,
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
