#!/usr/bin/env python
"""Collect teacher-guided episodes: a student solves questions while a teacher (that can
see the gold answer) reviews its plan and every step. This is how the training data in
``data/episodes`` was produced, and how to produce more with any student or teacher.

For each dataset the questions are split round-robin into ``--shards`` shards, one
template + one ``agentsim simulate`` worker per shard (concurrency = shards). Each
worker keeps a checkpoint and writes a ``_SUCCESS`` marker per question, so re-running
the same command resumes; a shard whose episodes are all present is skipped. Progress:
``<out>/_logs/<template>.log`` per shard and the per-question folders under ``<out>``.

Models are provider-prefixed ids (docs/PROVIDERS.md): the student is normally a local
vLLM server (``vllm/student``, see scripts/serve_vllm.sh), the teacher an API model;
both can be any provider. Afterwards run scripts/consolidate_episodes.py.

Usage::

    python scripts/collect_episodes.py --datasets hotpotqa musique --num-samples 2000 \
        --student vllm/student --teacher oai-teacher/deepseek-v4-flash --shards 6 \
        --out runs/collect
    python scripts/collect_episodes.py --datasets hotpotqa --num-samples 3 --smoke ...   # 3 questions
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import yaml

import tgd  # noqa: F401
from tgd import DATASETS, ROOT
from tgd.io import corpus_file, question_file, read_jsonl, write_jsonl

TEMPLATE_DIR = ROOT / "templates" / "simulations"


def build_template(*, template_id: str, student: str, teacher: str, questions_path: str,
                   corpus_path: str, output_dir: str, num_samples: int, budget: int,
                   disclose_budget: bool, planning_steps: int, max_plan_steps: int,
                   teacher_max_tokens: int, teacher_temperature: float, student_temperature: float) -> dict:
    """One simulation template = one worker's configuration (mirrors the harness's
    ``standard`` mode with plan review and guidance level 3, diagnostic feedback)."""
    mode_config = {
        "budget": budget,
        "student_model": student,
        "teacher_model": teacher,
        "teacher_router": [teacher],
        "student_use_response_schema": False,
        "disclose_budget": disclose_budget,
        "wiki_enabled": False,
        "wiki_mode": "tools",
        "corpus_path": corpus_path,
        "retrieval_backend": "hotpot_local",
        "skip_teacher": False,
        "teacher_max_tokens": teacher_max_tokens,
        "teacher_max_tokens_retry": teacher_max_tokens * 2,
        "teacher_temperature": teacher_temperature,
        "student_temperature": student_temperature,
        "guidance": {
            "level": 3, "name": "diagnostic_feedback", "score_mode": "continuous",
            "max_feedback_words": 60, "expose_next_action_hint": False, "expose_tool_hint": False,
            "expose_query_hint": False, "expose_doc_title_hint": False,
            "expose_gold_answer_hint": False, "leak_policy": "strict",
        },
        "plan_review": {
            "enabled": True, "planner": "student", "planning_steps": planning_steps,
            "formal_plan": False, "review_guidance_level": 3,
            "max_initial_plan_steps": max_plan_steps, "max_revised_plan_steps": max_plan_steps,
            "consume_budget": False, "include_revised_plan_in_student_context": True,
            "allow_teacher_to_suggest_tools": True, "allow_teacher_to_suggest_queries": False,
            "allow_teacher_to_reveal_gold_titles": False, "allow_teacher_to_reveal_gold_answer": False,
        },
    }
    return {
        "id": template_id,
        "name": f"teacher-guided collection ({student} student, {teacher} teacher, b={budget})",
        "mode": "standard",
        "teacher_models": [{"name": "student", "model_id": student, "role": "teacher",
                            "temperature": student_temperature}],
        "consultant_models": [],
        "workflows": [f"hotpot_teacher_guided_b{budget}_plan_review"],
        "datasets": [{"name": "hotpot_questions", "path": questions_path,
                      "num_samples": num_samples, "sample_strategy": "sequential"}],
        "max_iterations": 1,
        "similarity_metric": "token_overlap",
        "similarity_threshold": 0.0,
        "verification": {"enabled": False},
        "mode_config": mode_config,
        "output_dir": output_dir,
    }


def episodes_done(out_dir: Path) -> int:
    return sum(1 for _ in out_dir.rglob("_SUCCESS")) if out_dir.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--questions", default="data/questions")
    ap.add_argument("--num-samples", type=int, default=2000, help="questions per dataset (from the top of the file)")
    ap.add_argument("--student", required=True, help="student model id, e.g. vllm/student")
    ap.add_argument("--teacher", required=True, help="teacher model id, e.g. oai-teacher/<model>")
    ap.add_argument("--shards", type=int, default=6, help="concurrent workers per dataset")
    ap.add_argument("--budget", type=int, default=3, choices=[1, 2, 3, 4, 5, 9, 10, 12, 20, 30])
    ap.add_argument("--disclose-budget", action="store_true", help="tell the student its budget (default: hidden)")
    ap.add_argument("--planning-steps", type=int, default=3, help="plan-review rounds")
    ap.add_argument("--max-plan-steps", type=int, default=6)
    ap.add_argument("--teacher-max-tokens", type=int, default=2500)
    ap.add_argument("--teacher-temperature", type=float, default=0.1)
    ap.add_argument("--student-temperature", type=float, default=0.2)
    ap.add_argument("--out", default="runs/collect")
    ap.add_argument("--tag", default="collect", help="template/run name prefix")
    ap.add_argument("--smoke", action="store_true", help="1 shard, num-samples questions, separate tag")
    ap.add_argument("--plan-only", action="store_true", help="write templates, run nothing")
    args = ap.parse_args()

    workflow = ROOT / "templates" / "workflows" / f"hotpot_teacher_guided_b{args.budget}_plan_review.yaml"
    if not workflow.exists():
        ap.error(f"no workflow for budget {args.budget}: {workflow}")
    shards = 1 if args.smoke else args.shards
    tag = args.tag + ("_smoke" if args.smoke else "")
    out_root = Path(args.out)
    shard_root = out_root / "_shards"
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    plans: List[Dict] = []
    for ds in args.datasets:
        rows = list(read_jsonl(question_file(args.questions, ds), args.num_samples))
        for s in range(shards):
            shard_rows = rows[s::shards]          # round-robin, so no worker gets a
            sq = shard_root / ds / f"{ds}_s{s}.jsonl"   # contiguous (easier or harder) block
            write_jsonl(sq, shard_rows)
            tid = f"{tag}_{ds}_s{s}"
            out_dir = out_root / ds / tid
            tpl = build_template(
                template_id=tid, student=args.student, teacher=args.teacher,
                questions_path=str(sq), corpus_path=str(corpus_file(args.questions, ds)),
                output_dir=str(out_dir), num_samples=len(shard_rows), budget=args.budget,
                disclose_budget=args.disclose_budget, planning_steps=args.planning_steps,
                max_plan_steps=args.max_plan_steps, teacher_max_tokens=args.teacher_max_tokens,
                teacher_temperature=args.teacher_temperature, student_temperature=args.student_temperature)
            (TEMPLATE_DIR / f"{tid}.yaml").write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")
            plans.append({"dataset": ds, "template": tid, "questions": len(shard_rows), "out_dir": out_dir})
    total = sum(p["questions"] for p in plans)
    print(f"planned {len(plans)} worker(s) over {len(args.datasets)} dataset(s): {total} episodes -> {out_root}")
    if args.plan_only:
        return 0

    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for p in plans:
        done = episodes_done(p["out_dir"])
        if done >= p["questions"]:
            print(f"  skip {p['template']} ({done}/{p['questions']} done)")
            continue
        log = (log_dir / f"{p['template']}.log").open("a", encoding="utf-8")
        procs.append((p, subprocess.Popen([sys.executable, "-m", "agentsim.cli", "simulate", p["template"]],
                                          cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "PYTHONPATH": str(ROOT)}), log))
    print(f"launched {len(procs)} worker(s); logs in {log_dir}")
    t0 = time.time()
    while procs:
        time.sleep(30)
        still = []
        for p, proc, log in procs:
            if proc.poll() is None:
                still.append((p, proc, log))
            else:
                log.close()
                print(f"  {p['template']} exited {proc.returncode} ({episodes_done(p['out_dir'])}/{p['questions']} done)")
        procs = still
        done = sum(episodes_done(p["out_dir"]) for p in plans)
        print(f"  [{time.strftime('%H:%M:%S')}] {done}/{total} episodes, {len(procs)} workers running, "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)
    done = sum(episodes_done(p["out_dir"]) for p in plans)
    print(f"finished: {done}/{total} episodes. {'Re-run the same command to retry the missing ones.' if done < total else ''}")
    print(f"next: python scripts/consolidate_episodes.py --runs {out_root} --out data/episodes_new --gzip")
    return 0 if done >= total else 3


if __name__ == "__main__":
    raise SystemExit(main())
