"""Async all-API collector for teacher-only expert trajectories (PART A of exp_teacher_only).

gpt-oss-120b plays the STUDENT role (it plans and acts) with NO teacher in the loop
(``skip_teacher=True``). It drives the REAL harness components -- ``TeacherGuidedPlanReview``
and ``TeacherGuidedAgentStep`` -- so prompt rendering, deterministic tool execution and the
EM/F1/cover/doc-recall metrics are byte-identical to the eval and to the guidance-recipe
trace collection. The only difference from the guidance recipe is the trace source: here
the 120b is the expert agent instead of a weak student being critiqued by a teacher.

This is Variant 1 of PLAN.md (solo expert, one plan): ``skip_teacher=True`` means zero
teacher review/revision rounds, so the 120b produces one plan (<= 3 planned steps) and
executes <= 3 tool steps. The 120b-as-student never sees gold, so the traces are leak-free
by construction (a belt-and-suspenders leak gate still runs at dataset-build time).

Robustness (see PLAN.md G-9):
  * Fresh ``LLMClient`` per episode. The router's circuit breaker trips a provider on a
    hard wall-clock timeout and never un-trips within a process; scoping the client to one
    episode means a burst of timeouts can never cascade into "every remaining episode fails
    instantly" -- at worst one episode retries a dead endpoint once and falls through.
  * Every call is routed FAU -> free OpenRouter -> paid OpenRouter, so a rate-limited FAU
    falls through instead of failing. ``student_use_response_schema=False`` so only the
    force-finish step constrains output (the 120b already accepts a response schema on the
    guidance teacher path, so that one constrained call is safe).
  * Generous completion budgets: the 120b is a reasoning model that burns tokens on hidden
    reasoning before the JSON, so a tight cap truncates the action (G-8).
  * Manifest + resumable: expected qids are written up front, finished qids are skipped on
    restart, and ``missing_ids.txt`` is refreshed so gaps can be re-run.

Usage:
    .venv_train/bin/python scripts/run_teacher_only_traces.py \
        --questions data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl \
        --corpus    data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl \
        --out       data/simulation_output/traces_oss120b_teacheronly_3000 \
        --limit 3000 --concurrency 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentsim.config import config  # noqa: E402  (loads .env)
from agentsim.clients.llm_client import LLMClient  # noqa: E402
from agentsim.workflow.context import WorkflowContext  # noqa: E402
from agentsim.components.control.teacher_guided_plan_review import TeacherGuidedPlanReview  # noqa: E402
from agentsim.components.control.teacher_guided_agent_step import TeacherGuidedAgentStep  # noqa: E402
from agentsim.teacher_guidance.tool_executor import derive_final_answer  # noqa: E402
from agentsim.teacher_guidance.json_utils import parse_student_plan  # noqa: E402
from agentsim.teacher_guidance.metrics import (  # noqa: E402
    cover_match,
    exact_match,
    f1_score,
    supporting_doc_recall,
)
from scripts.gen_fau_smoke_template import TEACHER_ROUTER  # noqa: E402


class AllApiClient:
    """LLM client for the teacher-only run: routes EVERY call through the FAU -> free ->
    paid fallback router. A fresh underlying ``LLMClient`` per instance keeps the router's
    per-process circuit breaker scoped to a single episode (see module docstring / G-9)."""

    def __init__(self, router: List[str]):
        self._client = LLMClient()
        self._router = list(router)

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
        # student_model is nominally fau/gpt-oss-120b; route it through the whole chain so a
        # throttled FAU falls through to OpenRouter instead of failing the step.
        result, _used = await self._client.get_completion_with_fallback(
            self._router, prompt=prompt, temperature=temperature, max_tokens=max_tokens,
            return_raw=return_raw, response_schema=response_schema,
        )
        return result

    async def get_completion_with_fallback(self, models: List[str], **kwargs: Any):
        return await self._client.get_completion_with_fallback(models, **kwargs)


def build_metadata(row: Dict[str, Any], budget: int, corpus_path: str,
                   student_temperature: float) -> Dict[str, Any]:
    """Per-episode context metadata for the teacher-only (solo-expert) run.

    Mirrors the guidance-collection mode_config, but with ``skip_teacher=True`` (no
    reviewer, no per-step grade), the 120b as both student_model and teacher_model, a
    hidden budget, and generous student token budgets for the reasoning model."""
    gold = row.get("gold") or {"answer": row.get("answer", "")}
    return {
        "sample_id": row["id"],
        "dataset_sample": row,
        "gold": gold,
        "gold_answer": row.get("answer") or gold.get("answer", ""),
        "retrieval_scope": row.get("retrieval_scope", {}),
        "budget": budget,
        "student_model": TEACHER_ROUTER[0],
        "teacher_model": TEACHER_ROUTER[0],
        "teacher_router": list(TEACHER_ROUTER),
        "skip_teacher": True,
        # The 120b is served via API (no local grammar): disable constrained decoding for
        # normal student calls; the parse/repair path covers them. The force-finish step
        # keeps its tiny finish-only schema regardless (handled inside the component).
        "student_use_response_schema": False,
        "disclose_budget": False,  # hidden budget -- matches the guidance collection
        "corpus_path": corpus_path,
        "retrieval_backend": "hotpot_local",
        # Reasoning-model budgets (G-8): give the 120b room to finish the JSON after its
        # hidden chain-of-thought. Student repair does not bump tokens, so set it high once.
        "student_max_tokens": 3000,
        "student_plan_max_tokens": 2000,
        "student_max_repair_attempts": 2,
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
            # 3 review->revise rounds are configured to honour "3 planning steps", but
            # skip_teacher forces 0 actual rounds -> the 120b writes exactly one plan.
            "planning_steps": 3,
            "formal_plan": False,
            "review_guidance_level": 3,
            "max_initial_plan_steps": 3,
            "max_revised_plan_steps": 3,
            "consume_budget": False,
            "include_revised_plan_in_student_context": True,
            "allow_teacher_to_suggest_tools": True,
            "allow_teacher_to_suggest_queries": False,
            "allow_teacher_to_reveal_gold_titles": False,
            "allow_teacher_to_reveal_gold_answer": False,
        },
    }


def _export_steps(raw_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the fields the dataset builder + judge need, plus token usage for cost."""
    out = []
    for s in raw_steps:
        usage = [c.get("usage") for c in (s.get("student_calls") or []) if c.get("usage")]
        out.append({
            "t": s.get("t"),
            "student_prompt": s.get("student_prompt"),
            "student_raw": s.get("student_raw"),
            "student_action": s.get("student_action"),
            "action_valid": bool((s.get("metrics") or {}).get("action_schema_valid")),
            "tool_observation": s.get("tool_observation"),
            "student_usage": usage,
        })
    return out


async def run_episode(row: Dict[str, Any], budget: int, corpus_path: str,
                      student_temperature: float) -> Dict[str, Any]:
    """One solo-expert episode: 120b plans (1 plan) then executes <= budget tool steps."""
    qid = row["id"]
    gold_answer = row.get("answer", "") or (row.get("gold") or {}).get("answer", "")
    gold_doc_ids = set((row.get("gold") or {}).get("gold_doc_ids", []) or [])

    started = time.time()
    client = AllApiClient(TEACHER_ROUTER)  # fresh per episode -> isolated circuit breaker
    metadata = build_metadata(row, budget, corpus_path, student_temperature)
    ctx = WorkflowContext(task_id=qid, query=row["query"], metadata=metadata)

    await TeacherGuidedPlanReview(config={}, llm_client=client).execute(ctx)
    for t in range(1, budget + 1):
        if ctx.metadata.get("done"):
            break
        await TeacherGuidedAgentStep(
            config={"step_index": t, "budget": budget, "force_finish": t == budget},
            llm_client=client,
        ).execute(ctx)

    steps = ctx.metadata.get("teacher_guided_steps", []) or []
    final_answer = ctx.metadata.get("final_answer") or derive_final_answer(ctx)
    stop_reason = ctx.metadata.get("stop_reason", "budget_forced_finish")
    retrieved_ids = set(ctx.metadata.get("retrieved_doc_ids", []) or [])

    pr = ctx.metadata.get("plan_review") or {}
    plan_raw = pr.get("initial_student_plan_raw")
    plan_obj, plan_info = parse_student_plan(plan_raw or "")
    plan_export = {
        "prompt": pr.get("initial_student_plan_prompt"),
        "raw": plan_raw,
        "parsed": plan_obj,
        # episode_to_examples re-parses ``raw`` and requires steps; mark valid accordingly.
        "valid": bool(plan_info.get("json_valid") and isinstance(plan_obj, dict)
                      and plan_obj.get("steps")),
    }

    em = bool(exact_match(final_answer, gold_answer))
    return {
        "qid": qid,
        "query": row["query"],
        "gold_answer": gold_answer,
        "final_answer": final_answer,
        "budget": budget,
        "used_steps": len(steps),
        "stop_reason": stop_reason,
        "plan": plan_export,
        "steps": _export_steps(steps),
        "final_metrics": {
            "exact_match": em,
            "em": em,
            "f1": round(f1_score(final_answer, gold_answer), 4),
            "cover_match": bool(cover_match(final_answer, gold_answer)),
            "doc_recall": (round(supporting_doc_recall(retrieved_ids, gold_doc_ids), 4)
                           if gold_doc_ids else None),
        },
        "elapsed_s": round(time.time() - started, 2),
    }


def load_questions(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _done_qids(traces_path: Path) -> set:
    done = set()
    if traces_path.exists():
        with open(traces_path) as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(json.loads(line)["qid"])
                    except Exception:  # noqa: BLE001 -- tolerate a torn final line
                        pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="output directory (traces.jsonl lands here)")
    ap.add_argument("--limit", type=int, default=3000, help="first N questions in file order")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--student-temperature", type=float, default=0.2)
    ap.add_argument("--shard", default=None, help="i/n: run only shard i of n (round-robin)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    manifest_path = out_dir / "manifest_ids.txt"
    missing_path = out_dir / "missing_ids.txt"
    stats_path = out_dir / "stats_live.json"

    for m in TEACHER_ROUTER:
        if not config.provider_available(m):
            print(f"WARNING: provider for {m} is not configured", file=sys.stderr)

    questions = load_questions(args.questions, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for j, q in enumerate(questions) if j % n == i]
    expected_ids = [q["id"] for q in questions]
    manifest_path.write_text("\n".join(expected_ids) + "\n")

    done = _done_qids(traces_path)
    todo = [q for q in questions if q["id"] not in done]
    print(f"[teacher-only] expected={len(expected_ids)} done={len(done)} "
          f"todo={len(todo)} concurrency={args.concurrency} budget={args.budget}",
          flush=True)

    write_lock = asyncio.Lock()
    t0 = time.time()
    counters = {"done": len(done), "correct_cover": 0, "unknown": 0, "crashed": 0}
    traces_w = open(traces_path, "a")

    async def refresh_progress():
        seen = _done_qids(traces_path)
        missing = [q for q in expected_ids if q not in seen]
        missing_path.write_text("\n".join(missing) + "\n")
        elapsed = time.time() - t0
        new_done = len(seen) - len(done)
        rate = new_done / elapsed if elapsed > 0 else 0.0
        eta_h = (len(missing) / rate / 3600) if rate > 0 else None
        stats = {
            "expected": len(expected_ids),
            "done": len(seen),
            "missing": len(missing),
            "this_run_completed": new_done,
            "cover_correct_this_run": counters["correct_cover"],
            "unknown_this_run": counters["unknown"],
            "crashed_this_run": counters["crashed"],
            "elapsed_min": round(elapsed / 60, 1),
            "throughput_per_hour": round(rate * 3600, 1),
            "eta_hours": round(eta_h, 2) if eta_h is not None else None,
        }
        stats_path.write_text(json.dumps(stats, indent=2))

    async def run_all():
        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def run_one(q, idx):
            async with sem:
                try:
                    ep = await run_episode(q, args.budget, args.corpus,
                                           args.student_temperature)
                except Exception as exc:  # noqa: BLE001 -- one bad episode never stops the run
                    counters["crashed"] += 1
                    print(f"  qid={q.get('id')} CRASHED: {type(exc).__name__}: {str(exc)[:200]}",
                          flush=True)
                    return
                async with write_lock:
                    traces_w.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
                    traces_w.flush()
                    counters["done"] += 1
                    fm = ep["final_metrics"]
                    if fm["cover_match"]:
                        counters["correct_cover"] += 1
                    if str(ep["final_answer"]).strip().lower() == "unknown":
                        counters["unknown"] += 1
                    n = counters["done"]
                    if n % 25 == 0 or n <= 5:
                        await refresh_progress()
                    print(f"[{n}/{len(expected_ids)}] qid={ep['qid']} "
                          f"em={fm['exact_match']} cover={fm['cover_match']} f1={fm['f1']} "
                          f"steps={ep['used_steps']} stop={ep['stop_reason']} "
                          f"ans={str(ep['final_answer'])[:50]!r}", flush=True)

        await asyncio.gather(*(run_one(q, j) for j, q in enumerate(todo)))
        await refresh_progress()

    try:
        asyncio.run(run_all())
    finally:
        traces_w.close()

    elapsed = time.time() - t0
    final_done = len(_done_qids(traces_path))
    print(f"[teacher-only] finished: {final_done}/{len(expected_ids)} traces "
          f"in {elapsed/60:.1f} min | cover-correct this run={counters['correct_cover']} "
          f"| crashed={counters['crashed']}", flush=True)


if __name__ == "__main__":
    main()
