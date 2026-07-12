"""Teacher-IN-THE-LOOP evaluation for a HF checkpoint (base or base+LoRA).

Third eval arm: the student is the SAME local HF bf16 policy used by the
teacherless ``eval_agent.py`` (so student inference is directly comparable), but
the REAL teacher (gpt-oss-120b via the FAU -> free OpenRouter -> paid OpenRouter
router) reviews the plan and evaluates every step, exactly as in the b-run trace
collections. This measures whether external guidance still adds anything on top
of guidance the model has internalized.

Implementation: drives the REAL harness components (``TeacherGuidedPlanReview``,
``TeacherGuidedAgentStep``) -- prompts, guidance rendering, leakage checks,
repair loops and stop semantics are byte-identical to the trace runs -- through a
hybrid LLM client that routes student calls to the local HF model and everything
else to the normal ``LLMClient`` router.

Artifacts (timestamped, mirrors eval_agent.py):
    <out>/<ts>_<tag>/episodes.jsonl   episodes incl. full teacher records
    <out>/<ts>_<tag>/metrics.json     EM/F1/cover/doc-recall + teacher stats
    <out>/<ts>_<tag>/eval.log

Example:
    .venv_train/bin/python training_methods/common/teacher_eval_agent.py \
        --adapter training_methods/m1_sft/runs/<ts>_train4gpu/adapter \
        --questions training_methods/m1_sft/data/dev_questions.jsonl \
        --corpus data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl \
        --out training_methods/m1_sft/runs/eval_dev_teacher --tag m1_teacher --budget 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402
from training_methods.common.hf_agent_loop import PolicyModel, _messages, load_questions  # noqa: E402

from agentsim.workflow.context import WorkflowContext  # noqa: E402
from agentsim.clients.llm_client import LLMClient  # noqa: E402
from agentsim.components.control.teacher_guided_plan_review import TeacherGuidedPlanReview  # noqa: E402
from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep  # noqa: E402
from agentsim.teacher_guidance.tool_executor import derive_final_answer  # noqa: E402
from agentsim.teacher_guidance.metrics import (  # noqa: E402
    cover_match,
    exact_match,
    f1_score,
    supporting_doc_recall,
)
from scripts.gen_fau_smoke_template import TEACHER_ROUTER  # noqa: E402

HF_STUDENT_SENTINEL = "hf-local"


class HybridLLMClient:
    """Routes student calls (model == "hf-local") to the local HF policy and every
    other call (the teacher) to the shared ``LLMClient`` fallback router."""

    def __init__(self, policy: PolicyModel, teacher_client: LLMClient):
        self.policy = policy
        self.teacher = teacher_client
        # serialize GPU generation across concurrent episodes; teacher API calls
        # still overlap freely (that overlap is where the concurrency speedup lives)
        self._gpu_lock = asyncio.Lock()

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        return_raw: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if model == HF_STUDENT_SENTINEL:
            async with self._gpu_lock:
                text = await asyncio.to_thread(
                    self.policy.generate,
                    _messages(prompt),
                    max_tokens or 1200,
                    temperature,
                )
                usage = None
                if getattr(self.policy, "last_stats", None):
                    st = self.policy.last_stats[0]
                    usage = {"prompt_tokens": st["prompt_tokens"],
                             "completion_tokens": st["completion_tokens"]}
            return {"text": text, "raw_response": None, "usage": usage} if return_raw else text
        return await self.teacher.get_completion(
            prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens,
            return_raw=return_raw, response_schema=response_schema, **kwargs,
        )

    async def get_completion_with_fallback(self, router_models: List[str], **kwargs: Any):
        return await self.teacher.get_completion_with_fallback(router_models, **kwargs)


def build_metadata(row: Dict[str, Any], budget: int, corpus_path: str, disclose_budget: bool,
                   student_temperature: float = 0.0) -> Dict[str, Any]:
    """Per-episode context metadata, mirroring standard.py's mode_config injection
    with the b-run (g3, plan-review) settings."""
    gold = row.get("gold") or {"answer": row.get("answer", "")}
    return {
        "sample_id": row["id"],
        "dataset_sample": row,
        "gold": gold,
        "gold_answer": row.get("answer") or gold.get("answer", ""),
        "retrieval_scope": row.get("retrieval_scope", {}),
        "budget": budget,
        "student_model": HF_STUDENT_SENTINEL,
        "teacher_model": TEACHER_ROUTER[0],
        "teacher_router": list(TEACHER_ROUTER),
        # HF generation is not grammar-constrained; the parse/repair path covers it
        # (same semantics as student_use_response_schema=False Ollama runs).
        "student_use_response_schema": False,
        "disclose_budget": disclose_budget,
        "corpus_path": corpus_path,
        "retrieval_backend": "hotpot_local",
        "skip_teacher": False,
        "teacher_max_tokens": 2500,
        "teacher_max_tokens_retry": 4000,
        "student_temperature": student_temperature,
        "teacher_temperature": 0.1,
        "guidance": {
            "level": 3,
            "name": "diagnostic_feedback",
            "score_mode": "continuous",
            "max_feedback_words": 60,
            "expose_next_action_hint": False,
            "expose_tool_hint": False,
            "expose_query_hint": False,
            "expose_doc_title_hint": False,
            "expose_gold_answer_hint": False,
            "leak_policy": "strict",
        },
        "plan_review_config": {
            "enabled": True,
            "planner": "student",
            "planning_steps": 1,
            "formal_plan": False,
            "review_guidance_level": 3,
            "max_initial_plan_steps": budget,
            "max_revised_plan_steps": budget,
            "consume_budget": False,
            "include_revised_plan_in_student_context": True,
            "allow_teacher_to_suggest_tools": True,
            "allow_teacher_to_suggest_queries": False,
            "allow_teacher_to_reveal_gold_titles": False,
            "allow_teacher_to_reveal_gold_answer": False,
        },
    }


def _trim_plan_review(pr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Keep the analysis-relevant parts of the plan-review record."""
    if not isinstance(pr, dict):
        return pr
    keep = {k: pr.get(k) for k in (
        "enabled", "initial_student_plan", "revised_student_plan",
        "student_visible_plan_feedback", "final_teacher_decision", "rounds_used",
    ) if k in pr}
    rounds = pr.get("rounds") or []
    keep["rounds"] = [
        {k: r.get(k) for k in ("round", "accepted", "student_visible_plan_feedback") if k in r}
        for r in rounds if isinstance(r, dict)
    ]
    return keep


async def run_teacher_episode(
    client: HybridLLMClient,
    row: Dict[str, Any],
    budget: int,
    corpus_path: str,
    disclose_budget: bool,
    with_plan: bool,
    log,
    student_temperature: float = 0.0,
) -> Dict[str, Any]:
    qid = row["id"]
    gold = row.get("answer", "") or (row.get("gold") or {}).get("answer", "")
    gold_doc_ids = set((row.get("gold") or {}).get("gold_doc_ids", []) or [])

    started = time.time()
    metadata = build_metadata(row, budget, corpus_path, disclose_budget, student_temperature)
    if not with_plan:
        metadata["plan_review_config"]["enabled"] = False
    ctx = WorkflowContext(task_id=qid, query=row["query"], metadata=metadata)

    plan_res = await TeacherGuidedPlanReview(config={}, llm_client=client).execute(ctx)
    if not plan_res.success:
        log.warning(f"qid={qid} plan review failed: {plan_res.error}")

    for t in range(1, budget + 1):
        if ctx.metadata.get("done"):
            break
        step_res = await TeacherGuidedAgentStep(
            config={"step_index": t, "budget": budget, "force_finish": t == budget},
            llm_client=client,
        ).execute(ctx)
        if not step_res.success:
            log.warning(f"qid={qid} step {t} failed: {step_res.error}")
            break
        d = step_res.data or {}
        log.info(
            f"qid={qid} step={t}/{budget} tool={(d.get('student_action') or {}).get('action', {}).get('tool')} "
            f"teacher={d.get('teacher_decision')} verdict={d.get('verdict')}"
        )

    steps: List[Dict[str, Any]] = ctx.metadata.get("teacher_guided_steps", []) or []
    final_answer = ctx.metadata.get("final_answer") or derive_final_answer(ctx)
    stop_reason = ctx.metadata.get("stop_reason", "budget_forced_finish")

    retrieved_ids = set(ctx.metadata.get("retrieved_doc_ids", []) or [])
    metrics = {
        "exact_match": bool(exact_match(final_answer, gold)),
        "f1": round(f1_score(final_answer, gold), 4),
        "cover_match": bool(cover_match(final_answer, gold)),
        "doc_recall": round(supporting_doc_recall(retrieved_ids, gold_doc_ids), 4) if gold_doc_ids else None,
    }
    return {
        "qid": qid,
        "query": row["query"],
        "gold_answer": gold,
        "final_answer": final_answer,
        "budget": budget,
        "used_steps": len(steps),
        "stop_reason": stop_reason,
        "plan_review": _trim_plan_review(ctx.metadata.get("plan_review")),
        "steps": steps,
        "teacher_final_judgment": ctx.metadata.get("teacher_final_judgment"),
        "final_metrics": metrics,
        "elapsed_s": round(time.time() - started, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="base dir; a timestamped subdir is created")
    ap.add_argument("--tag", default="teacher_eval")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--hidden-budget", action="store_true")
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None, help="i/n to run only shard i of n (0-based)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help=">1 runs episodes concurrently: teacher API waits overlap "
                         "with (GPU-serialized) student generation of other episodes")
    ap.add_argument("--student-temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch/random/numpy (meaningful with --student-temperature > 0)")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out, args.tag)
    log = setup_logger("teacher_eval", run_dir / "eval.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")
    log.info(f"teacher router: {TEACHER_ROUTER}")

    questions = load_questions(args.questions, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for j, q in enumerate(questions) if j % n == i]
        log.info(f"shard {i}/{n}: {len(questions)} questions")
    log.info(f"loaded {len(questions)} questions | corpus={args.corpus}")

    if args.seed is not None:
        import random as _random

        import numpy as _np
        import torch as _torch
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        _torch.manual_seed(args.seed)
        log.info(f"seeded everything with {args.seed}")

    policy = PolicyModel(args.model, args.adapter, device=args.device)
    log.info(f"policy loaded: {args.model} adapter={args.adapter}")
    client = HybridLLMClient(policy, LLMClient())
    t_run0 = time.time()

    async def run_all() -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(max(1, args.concurrency))
        with open(run_dir / "episodes.jsonl", "w") as w:

            async def run_one(q):
                async with sem:
                    try:
                        ep = await run_teacher_episode(
                            client, q, args.budget, args.corpus,
                            disclose_budget=not args.hidden_budget,
                            with_plan=not args.no_plan, log=log,
                            student_temperature=args.student_temperature,
                        )
                    except Exception as exc:  # noqa: BLE001 -- keep the eval going
                        log.error(f"qid={q.get('id')} episode crashed: {type(exc).__name__}: {exc}")
                        return
                    episodes.append(ep)
                    w.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
                    w.flush()
                    m = ep["final_metrics"]
                    log.info(
                        f"[{len(episodes)}/{len(questions)}] qid={ep['qid']} em={m['exact_match']} "
                        f"f1={m['f1']} cover={m['cover_match']} steps={ep['used_steps']} "
                        f"stop={ep['stop_reason']} ans={ep['final_answer'][:60]!r}"
                    )

            if args.concurrency > 1:
                await asyncio.gather(*(run_one(q) for q in questions))
            else:
                for q in questions:
                    await run_one(q)
        return episodes

    episodes = asyncio.run(run_all())
    n = len(episodes)
    if not n:
        log.error("no episodes completed")
        sys.exit(1)

    doc_recalls = [e["final_metrics"]["doc_recall"] for e in episodes
                   if e["final_metrics"]["doc_recall"] is not None]
    guid_scores = [
        (s.get("student_visible_guidance") or {}).get("score")
        for e in episodes for s in e["steps"]
    ]
    guid_scores = [s for s in guid_scores if isinstance(s, (int, float))]
    teacher_decisions: Dict[str, int] = {}
    for e in episodes:
        for s in e["steps"]:
            d = (s.get("teacher_full") or {}).get("teacher_decision") or "unknown"
            teacher_decisions[d] = teacher_decisions.get(d, 0) + 1
    # token/time accounting from the per-call logs the harness components keep
    import torch as _torch
    stu_pt = stu_ct = tea_pt = tea_ct = 0
    stu_ms = tea_ms = 0.0
    for e in episodes:
        for s in e["steps"]:
            for c in (s.get("student_calls") or []):
                u = c.get("usage") or {}
                stu_pt += u.get("prompt_tokens") or 0
                stu_ct += u.get("completion_tokens") or 0
                stu_ms += c.get("elapsed_ms") or 0
            for c in (s.get("teacher_calls") or []):
                u = c.get("usage") or {}
                tea_pt += u.get("prompt_tokens") or 0
                tea_ct += u.get("completion_tokens") or 0
                tea_ms += c.get("elapsed_ms") or 0
    agg = {
        "arm": "teacher_in_loop",
        "model": args.model,
        "adapter": args.adapter,
        "teacher_router": list(TEACHER_ROUTER),
        "seed": args.seed,
        "student_temperature": args.student_temperature,
        "concurrency": args.concurrency,
        "wall_time_s": round(time.time() - t_run0, 1),
        "gpu_peak_mem_gb": round(_torch.cuda.max_memory_allocated() / 1e9, 3)
        if _torch.cuda.is_available() else None,
        "student_prompt_tokens": stu_pt,
        "student_completion_tokens": stu_ct,
        "student_call_time_s": round(stu_ms / 1000, 1),
        "teacher_prompt_tokens": tea_pt,
        "teacher_completion_tokens": tea_ct,
        "teacher_call_time_s": round(tea_ms / 1000, 1),
        "n": n,
        "budget": args.budget,
        "em": round(sum(e["final_metrics"]["exact_match"] for e in episodes) / n, 4),
        "f1": round(sum(e["final_metrics"]["f1"] for e in episodes) / n, 4),
        "cover_match": round(sum(e["final_metrics"]["cover_match"] for e in episodes) / n, 4),
        "doc_recall": round(statistics.mean(doc_recalls), 4) if doc_recalls else None,
        "mean_steps": round(statistics.mean(e["used_steps"] for e in episodes), 2),
        "stop_reasons": {r: sum(1 for e in episodes if e["stop_reason"] == r)
                         for r in {e["stop_reason"] for e in episodes}},
        "invalid_action_steps": sum(
            1 for e in episodes for s in e["steps"]
            if not (s.get("metrics") or {}).get("action_schema_valid", True)),
        "total_steps": sum(e["used_steps"] for e in episodes),
        "teacher_decisions": teacher_decisions,
        "mean_teacher_step_score": round(statistics.mean(guid_scores), 4) if guid_scores else None,
        "teacher_correct_final": sum(
            1 for e in episodes if (e.get("teacher_final_judgment") or {}).get("correct") == 1),
    }
    write_json(run_dir / "metrics.json", agg)
    log.info(f"AGGREGATE: {json.dumps(agg)}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
