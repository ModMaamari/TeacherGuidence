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
from pathlib import Path
from typing import Any, Dict, List

from agentsim.teacher_guidance.metrics import compute_final_metrics


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

        guidance = md.get("guidance", {}) or {}
        plan_review = md.get("plan_review", {"enabled": False})

        return {
            "episode_id": f"{md.get('sample_id', context.task_id)}",
            "qid": gold.get("qid", md.get("retrieval_scope", {}).get("qid", context.task_id)),
            "query": context.query,
            "gold_answer": gold.get("answer", ""),
            "dataset": "hotpotqa",
            "split": md.get("split", "validation"),
            "budget": int(guidance.get("budget", md.get("budget", len(steps)))) if isinstance(guidance, dict) else len(steps),
            "guidance_level": int(guidance.get("level", 0)) if isinstance(guidance, dict) else 0,
            "student_model": md.get("student_model", ""),
            "teacher_model": md.get("teacher_model", ""),
            "plan_review": plan_review,
            "steps": [
                {
                    "t": s.get("t"),
                    "student_action": s.get("student_action"),
                    "tool_observation": s.get("tool_observation"),
                    "teacher_private_diagnosis": (s.get("teacher_full", {}) or {}).get("private_diagnosis", {}),
                    "student_visible_guidance": s.get("student_visible_guidance"),
                    "metrics": s.get("metrics"),
                    "leakage_check": s.get("leakage_check"),
                    "stop_condition": s.get("stop_condition", "CONTINUE"),
                }
                for s in steps
            ],
            "final_answer": final_answer,
            "final_metrics": final_metrics,
            "stop_reason": md.get("stop_reason", "error"),
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
