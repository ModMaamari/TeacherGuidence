"""Episodes driven by the harness's teacher-guided components.

Two of the four evaluation arms run here:

* **guided**  -- the student (local policy) acts, a teacher (API) reviews its plan and
                 every step; the student sees the teacher's feedback.
* **teacher** -- the teacher model itself is the agent; ``skip_teacher`` disables the
                 critic. Same prompts, tools, budget and metrics as every other arm.

The components are the real harness ones (``TeacherGuidedPlanReview``,
``TeacherGuidedAgentStep``), so prompts, guidance rendering, leakage checks, repair
loops and stop semantics are identical to the collection runs that produced the
training data. A ``HybridLLMClient`` routes calls for the sentinel model name
``local-student`` to the local policy and every other model id to the provider router.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from agentsim.clients.llm_client import LLMClient
from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep
from agentsim.components.control.teacher_guided_plan_review import TeacherGuidedPlanReview
from agentsim.teacher_guidance.metrics import cover_match, exact_match, f1_score, supporting_doc_recall
from agentsim.teacher_guidance.tool_executor import derive_final_answer
from agentsim.workflow.context import WorkflowContext

from tgd.hf_agent_loop import _messages

LOCAL_STUDENT = "local-student"


class HybridLLMClient:
    """Route ``LOCAL_STUDENT`` calls to the local policy, everything else to ``LLMClient``."""

    def __init__(self, policy, teacher_client: LLMClient, serialize_gpu: bool):
        self.policy = policy
        self.teacher = teacher_client
        # in-process HF generation must be serialized across episodes; a vLLM server
        # continuous-batches, so concurrent calls are fine there
        self._gpu_lock = asyncio.Lock() if serialize_gpu else None

    async def get_completion(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7,
                             max_tokens: Optional[int] = None, return_raw: bool = False,
                             response_schema: Optional[Dict[str, Any]] = None, **kwargs: Any):
        if model == LOCAL_STUDENT:
            if self.policy is None:
                raise RuntimeError("no local policy configured for the student")
            if self._gpu_lock is not None:
                async with self._gpu_lock:
                    text = await asyncio.to_thread(self.policy.generate, _messages(prompt), max_tokens or 1200, temperature)
                    stats = list(getattr(self.policy, "last_stats", []) or [])
            else:
                text, stat = await asyncio.to_thread(self.policy.generate_with_stats, _messages(prompt),
                                                     max_tokens or 1200, temperature)
                stats = [stat]
            usage = None
            if stats and stats[0]:
                usage = {"prompt_tokens": stats[0]["prompt_tokens"], "completion_tokens": stats[0]["completion_tokens"]}
            return {"text": text, "raw_response": None, "usage": usage} if return_raw else text
        return await self.teacher.get_completion(prompt=prompt, model=model, temperature=temperature,
                                                 max_tokens=max_tokens, return_raw=return_raw,
                                                 response_schema=response_schema, **kwargs)

    async def get_completion_with_fallback(self, router_models: List[str], **kwargs: Any):
        return await self.teacher.get_completion_with_fallback(router_models, **kwargs)


def build_metadata(row: Dict[str, Any], *, budget: int, corpus_path: str, disclose_budget: bool,
                   student_model: str, teacher_router: List[str], skip_teacher: bool,
                   student_temperature: float, teacher_temperature: float, with_plan: bool,
                   teacher_max_tokens: int, student_max_tokens: int) -> Dict[str, Any]:
    """Per-episode context, mirroring the collection template's mode_config."""
    gold = row.get("gold") or {"answer": row.get("answer", "")}
    return {
        "sample_id": row["id"],
        "dataset_sample": row,
        "gold": gold,
        "gold_answer": row.get("answer") or gold.get("answer", ""),
        "retrieval_scope": row.get("retrieval_scope", {}),
        "budget": budget,
        "student_model": student_model,
        "teacher_model": teacher_router[0] if teacher_router else None,
        "teacher_router": list(teacher_router),
        "student_use_response_schema": False,
        "disclose_budget": disclose_budget,
        "corpus_path": corpus_path,
        "retrieval_backend": "hotpot_local",
        "skip_teacher": skip_teacher,
        "teacher_max_tokens": teacher_max_tokens,
        "teacher_max_tokens_retry": teacher_max_tokens * 2,
        "student_max_tokens": student_max_tokens,
        "student_temperature": student_temperature,
        "teacher_temperature": teacher_temperature,
        "guidance": {
            "level": 3, "name": "diagnostic_feedback", "score_mode": "continuous",
            "max_feedback_words": 60, "expose_next_action_hint": False, "expose_tool_hint": False,
            "expose_query_hint": False, "expose_doc_title_hint": False,
            "expose_gold_answer_hint": False, "leak_policy": "strict",
        },
        "plan_review_config": {
            "enabled": with_plan, "planner": "student", "planning_steps": 1, "formal_plan": False,
            "review_guidance_level": 3, "max_initial_plan_steps": budget,
            "max_revised_plan_steps": budget, "consume_budget": False,
            "include_revised_plan_in_student_context": True,
            "allow_teacher_to_suggest_tools": True, "allow_teacher_to_suggest_queries": False,
            "allow_teacher_to_reveal_gold_titles": False, "allow_teacher_to_reveal_gold_answer": False,
        },
    }


def _trim_plan_review(pr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Keep the analysis-relevant parts of the plan-review record, including the call
    records (token usage) of every round."""
    if not isinstance(pr, dict):
        return pr
    keep = {k: pr.get(k) for k in (
        "enabled", "initial_student_plan", "revised_student_plan", "student_visible_plan_feedback",
        "final_teacher_decision", "rounds_used", "initial_plan_calls",
    ) if k in pr}
    keep["rounds"] = [
        {k: r.get(k) for k in ("round", "accepted", "student_visible_plan_feedback",
                               "review_calls", "revision_calls") if k in r}
        for r in (pr.get("rounds") or []) if isinstance(r, dict)
    ]
    return keep


async def run_guided_episode(client: HybridLLMClient, row: Dict[str, Any], *, budget: int, log,
                             **meta_kwargs) -> Dict[str, Any]:
    qid = row["id"]
    gold = row.get("answer", "") or (row.get("gold") or {}).get("answer", "")
    gold_doc_ids = set((row.get("gold") or {}).get("gold_doc_ids", []) or [])
    started = time.time()
    metadata = build_metadata(row, budget=budget, **meta_kwargs)
    ctx = WorkflowContext(task_id=qid, query=row["query"], metadata=metadata)

    plan_res = await TeacherGuidedPlanReview(config={}, llm_client=client).execute(ctx)
    if not plan_res.success:
        log.warning(f"qid={qid} plan review failed: {plan_res.error}")
    for t in range(1, budget + 1):
        if ctx.metadata.get("done"):
            break
        step_res = await TeacherGuidedAgentStep(
            config={"step_index": t, "budget": budget, "force_finish": t == budget}, llm_client=client,
        ).execute(ctx)
        if not step_res.success:
            log.warning(f"qid={qid} step {t} failed: {step_res.error}")
            break

    steps: List[Dict[str, Any]] = ctx.metadata.get("teacher_guided_steps", []) or []
    final_answer = ctx.metadata.get("final_answer") or derive_final_answer(ctx)
    retrieved_ids = set(ctx.metadata.get("retrieved_doc_ids", []) or [])
    metrics = {
        "exact_match": bool(exact_match(final_answer, gold)),
        "f1": round(f1_score(final_answer, gold), 4),
        "cover_match": bool(cover_match(final_answer, gold)),
        "doc_recall": round(supporting_doc_recall(retrieved_ids, gold_doc_ids), 4) if gold_doc_ids else None,
    }
    return {
        "qid": qid, "query": row["query"], "gold_answer": gold, "final_answer": final_answer,
        "budget": budget, "used_steps": len(steps),
        "stop_reason": ctx.metadata.get("stop_reason", "budget_forced_finish"),
        "plan_review": _trim_plan_review(ctx.metadata.get("plan_review")),
        "steps": steps,
        "teacher_final_judgment": ctx.metadata.get("teacher_final_judgment"),
        "final_metrics": metrics,
        "elapsed_s": round(time.time() - started, 2),
    }
