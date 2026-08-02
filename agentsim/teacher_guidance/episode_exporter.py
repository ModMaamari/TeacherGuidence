"""
Clean Teacher Guidance episode exporter.

Writes downstream-friendly views from a finished workflow context:

    teacher_guidance_episodes.jsonl      one row per question trajectory
    student_sft.jsonl                    one row per student step
    teacher_sft.jsonl                    one row per teacher step
    student_visible_guidance.jsonl       one row per rendered guidance object
    plan_review_rows.jsonl               one row per plan review (when enabled)
    teacher_guidance_metrics.json        final metrics for the episode

The student SFT input is the student prompt, which is built only from
student-visible state and therefore never contains the gold answer or the teacher's
private diagnosis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agentsim.teacher_guidance.metrics import compute_final_metrics
from agentsim.teacher_guidance.optimality import compute_path_optimality
from agentsim.teacher_guidance.provenance import (
    EPISODE_SCHEMA_VERSION,
    framework_commit,
)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TeacherGuidanceEpisodeExporter:
    """Export a single episode (one workflow context) to clean files."""

    def export_episode(self, context: Any, output_dir: str) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        episode = self._build_episode_record(context)
        _append_jsonl(out / "teacher_guidance_episodes.jsonl", episode)

        steps = context.metadata.get("teacher_guided_steps", []) or []
        self._export_student_sft(steps, episode, out)
        self._export_teacher_sft(steps, episode, out)
        self._export_guidance_rows(steps, episode, out)
        self._export_plan_review(context, episode, out)

        with open(out / "teacher_guidance_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "qid": episode["qid"],
                    "guidance_level": episode["guidance_level"],
                    "stop_reason": episode["stop_reason"],
                    "final_metrics": episode["final_metrics"],
                    "num_steps": len(steps),
                },
                f,
                indent=2,
            )
        return episode

    @staticmethod
    def _collect_used_teacher_models(context: Any, steps: List[Dict[str, Any]]) -> List[str]:
        """Distinct teacher models that ACTUALLY served a call this episode.

        The configured ``teacher_model`` is only the first choice; the fallback router
        (``get_completion_with_fallback``) may serve a call from a later provider in
        ``teacher_router`` and records the real one under each call's ``"model"``. We walk
        only the genuinely teacher-served calls -- per-step teacher evals and the
        plan-review ``review_calls`` -- so the stored provenance reflects which teacher(s)
        produced the guidance. Note: plan-review ``initial_plan_calls`` and
        ``revision_calls`` are the STUDENT writing/revising its own plan (planner=student),
        so they are deliberately excluded here.
        """
        used: List[str] = []
        seen = set()

        def _add_from(calls: Any) -> None:
            for c in calls or []:
                m = (c or {}).get("model")
                if m and m not in seen:
                    seen.add(m)
                    used.append(m)

        for s in steps:
            _add_from(s.get("teacher_calls"))
        pr = context.metadata.get("plan_review") or {}
        for rnd in pr.get("rounds", []) or []:
            _add_from(rnd.get("review_calls"))
        return used

    # ------------------------------------------------------------------
    def _build_episode_record(self, context: Any) -> Dict[str, Any]:
        md = context.metadata
        gold = md.get("gold", {}) or {}
        steps = md.get("teacher_guided_steps", []) or []
        final_answer = md.get("final_answer", "") or ""

        corpus = self._gather_corpus(context, gold)
        final_metrics = compute_final_metrics(
            final_answer=final_answer,
            gold_answer=gold.get("answer", "") or "",
            retrieved_doc_ids=set(md.get("retrieved_doc_ids", []) or []),
            gold_doc_ids=set(gold.get("gold_doc_ids", []) or []),
            extracted_spans=md.get("extracted_facts", []) or [],
            gold_facts=gold.get("supporting_facts", []) or [],
            corpus=corpus,
        )
        # The teacher's own verdict on the final answer (it can see the gold answer),
        # reported alongside the deterministic cover-match metric. Absent when there was
        # no teacher (skip_teacher runs) or the teacher didn't return the fields.
        judgment = md.get("teacher_final_judgment") or {}
        final_metrics["teacher_answer_correct"] = judgment.get("correct")
        final_metrics["teacher_answer_score"] = judgment.get("score")

        # Paragraph-level sources (MuSiQue, StrategyQA) annotate supporting *paragraphs*,
        # never sentences, so there are no gold facts to recall. Report None rather than
        # the 0.0 the metric returns for an empty fact list -- 0.0 would read as "the
        # student recalled nothing" and would drag down any average taken over datasets.
        gold_granularity = md.get("gold_granularity", "sentence")
        if gold_granularity != "sentence":
            final_metrics["supporting_fact_recall"] = None

        guidance = md.get("guidance", {}) or {}
        plan_review = md.get("plan_review", {"enabled": False})

        # Budget = max student tool-use steps the episode was allowed; used_steps =
        # how many the student actually took before stopping. These are distinct.
        budget = md.get("budget")
        if budget is None:
            budget = len(steps)

        step_records = [
            {
                "t": s.get("t"),
                "student_prompt": s.get("student_prompt", ""),
                "student_raw": s.get("student_raw", ""),
                "student_calls": s.get("student_calls"),
                "student_call_ms": s.get("student_call_ms"),
                "student_action": s.get("student_action"),
                "tool_observation": s.get("tool_observation"),
                "teacher_prompt": s.get("teacher_prompt", ""),
                "teacher_raw": s.get("teacher_raw", ""),
                "teacher_calls": s.get("teacher_calls"),
                "teacher_call_ms": s.get("teacher_call_ms"),
                # The model(s) that actually served this step's teacher call(s).
                "teacher_models_used": [
                    c.get("model") for c in (s.get("teacher_calls") or []) if (c or {}).get("model")
                ],
                "teacher_skipped": s.get("teacher_skipped", False),
                "teacher_private_diagnosis": (s.get("teacher_full", {}) or {}).get("private_diagnosis", {}),
                "student_visible_guidance": s.get("student_visible_guidance"),
                "metrics": s.get("metrics"),
                "leakage_check": s.get("leakage_check"),
                "stop_condition": s.get("stop_condition", "CONTINUE"),
                "wiki_update_call": s.get("wiki_update_call"),
                "wiki_after": s.get("wiki_after"),
                "wiki_edit_ops": s.get("wiki_edit_ops"),
                "step_started_at": s.get("step_started_at"),
                "step_ended_at": s.get("step_ended_at"),
                "step_elapsed_ms": s.get("step_elapsed_ms"),
            }
            for s in steps
        ]

        sample = md.get("dataset_sample") or {}
        return {
            "episode_id": f"{md.get('sample_id', context.task_id)}",
            "qid": gold.get("qid", md.get("retrieval_scope", {}).get("qid", context.task_id)),
            "query": context.query,
            "gold_answer": gold.get("answer", ""),
            # Provenance: which source dataset/split produced this question, which schema
            # this record follows, which code produced it, and under which configuration.
            "dataset": md.get("dataset", "hotpotqa"),
            "split": md.get("dataset_split") or md.get("split", "validation"),
            "gold_granularity": gold_granularity,
            "answer_type": sample.get("answer_type", "span"),
            "num_hops": sample.get("num_hops"),
            "question_type": sample.get("type", ""),
            "schema_version": EPISODE_SCHEMA_VERSION,
            "framework_commit": framework_commit(),
            "config_hash": md.get("config_hash", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "budget": int(budget),
            "used_steps": len(steps),
            "guidance_level": int(guidance.get("level", 0)) if isinstance(guidance, dict) else 0,
            "student_model": md.get("student_model", ""),
            # teacher_model = the configured first-choice teacher; teacher_router = the
            # full fallback chain; teacher_models_used = the model(s) that ACTUALLY served
            # this episode's teacher calls (the router may fall through to a fallback), so
            # the used teacher is captured on the stored data, not just the request.
            "teacher_model": md.get("teacher_model", ""),
            "teacher_router": md.get("teacher_router") or [],
            "teacher_models_used": self._collect_used_teacher_models(context, steps),
            "plan_review": plan_review,
            "steps": step_records,
            "final_answer": final_answer,
            "final_metrics": final_metrics,
            "path_optimality": compute_path_optimality(step_records),
            "plan_adherence": md.get("plan_adherence"),
            "stop_reason": md.get("stop_reason", "error"),
            # Agent-wiki runs: record the flag and the wiki's final content so wiki
            # usage can be analyzed per episode (absent/false on non-wiki runs).
            "wiki_enabled": bool(md.get("wiki_enabled", False)),
            "wiki_mode": (md.get("wiki_mode", "tools") if md.get("wiki_enabled") else None),
            "wiki_final": (str(md.get("wiki", "") or "") if md.get("wiki_enabled") else None),
        }

    def _gather_corpus(self, context: Any, gold: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        retriever = getattr(context, "_tg_retriever", None)
        corpus: Dict[str, Dict[str, Any]] = {}
        if retriever is None:
            return corpus
        doc_ids = set(gold.get("gold_doc_ids", []) or []) | set(
            context.metadata.get("retrieved_doc_ids", []) or []
        )
        for doc_id in doc_ids:
            doc = retriever.get_doc(doc_id)
            if doc:
                corpus[doc_id] = doc
        return corpus

    # ------------------------------------------------------------------
    def _export_student_sft(self, steps: List[Dict[str, Any]], episode: Dict[str, Any], out: Path) -> None:
        for s in steps:
            _append_jsonl(
                out / "student_sft.jsonl",
                {
                    "input": s.get("student_prompt", ""),
                    "output": s.get("student_raw", ""),
                    "metadata": {
                        "qid": episode["qid"],
                        "step": s.get("t"),
                        "guidance_level": episode["guidance_level"],
                        "gold_answer_hidden": True,
                    },
                },
            )

    def _export_teacher_sft(self, steps: List[Dict[str, Any]], episode: Dict[str, Any], out: Path) -> None:
        for s in steps:
            _append_jsonl(
                out / "teacher_sft.jsonl",
                {
                    "input": s.get("teacher_prompt", ""),
                    "output": s.get("teacher_raw", ""),
                    "metadata": {
                        "qid": episode["qid"],
                        "step": s.get("t"),
                        "guidance_level": episode["guidance_level"],
                        "gold_answer_visible_to_teacher": True,
                        # Which teacher(s) actually produced this demonstration.
                        "teacher_model": episode.get("teacher_model", ""),
                        "teacher_models_used": [
                            c.get("model") for c in (s.get("teacher_calls") or []) if (c or {}).get("model")
                        ],
                    },
                },
            )

    def _export_guidance_rows(self, steps: List[Dict[str, Any]], episode: Dict[str, Any], out: Path) -> None:
        for s in steps:
            _append_jsonl(
                out / "student_visible_guidance.jsonl",
                {
                    "qid": episode["qid"],
                    "step": s.get("t"),
                    "guidance_level": episode["guidance_level"],
                    "rendered_guidance": s.get("student_visible_guidance"),
                    "leakage_check": s.get("leakage_check"),
                },
            )

    def _export_plan_review(self, context: Any, episode: Dict[str, Any], out: Path) -> None:
        plan_review = context.metadata.get("plan_review")
        if not plan_review or not plan_review.get("enabled"):
            return
        row = dict(plan_review)
        row["qid"] = episode["qid"]
        _append_jsonl(out / "plan_review_rows.jsonl", row)
