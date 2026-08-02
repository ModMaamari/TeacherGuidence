"""End-to-end smoke test of the teacher x student collection matrix.

Runs a handful of real episodes per cell and audits what came out, so a mistake in the
teachers, the routing, the prompts or the export surfaces before a six-figure-episode run.

Two student modes:

* ``--student-mode vllm`` (default) -- serves the real student locally, exactly as
  collection will. This is the faithful check and needs working GPUs.
* ``--student-mode api`` -- substitutes a small hosted model for the student. It does NOT
  validate vLLM serving, but it validates everything else (prompt construction, action
  parsing, tool execution, teacher guidance, leakage gating, provenance, episode export)
  and needs no GPU. Use it when GPUs are unavailable, or as a fast pre-check.

Usage::

    .venv/bin/python scripts/smoke_matrix.py --plan resource_paper/plan/collection_plan.json
    .venv/bin/python scripts/smoke_matrix.py --teacher deepseek-v4-flash \
        --student qwen-0.8b --student-mode api --num-samples 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from agentsim.teacher_guidance.model_registry import (  # noqa: E402
    STUDENTS,
    TEACHERS,
    combination_id,
    get_student,
    get_teacher,
)
from scripts.gen_fau_smoke_template import build_fau_smoke_template  # noqa: E402
from scripts.smoke_datasets import audit_episodes  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "templates" / "simulations"

#: Small free hosted model used to stand in for a student in ``--student-mode api``.
API_STUDENT = "fau/Microsoft/Phi-4-mini-instruct"


def run_cell(
    teacher: str,
    student: str,
    questions: Path,
    corpus: Path,
    out_root: Path,
    *,
    num_samples: int,
    student_mode: str,
    budget: int,
    planning_steps: int,
    guidance_level: int,
    dataset_name: str,
) -> Dict[str, Any]:
    """Run one teacher x student cell and audit the episodes it produced."""
    tspec, sspec = get_teacher(teacher), get_student(student)
    cell = combination_id(teacher, student)
    # The simulate CLI resolves output_dir relative to the repo root, so keep an absolute
    # path for our own IO and a repo-relative one for the template.
    out_dir = (out_root if out_root.is_absolute() else REPO_ROOT / out_root) / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    student_model = API_STUDENT if student_mode == "api" else sspec.served_model
    template_id = f"mxsmoke_{cell.replace('.', '')}"[:60]

    # Trim the question file to the smoke size so the simulate CLI reads only what we want.
    q_lines = [l for l in questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    q_path = out_dir / "questions.jsonl"
    q_path.write_text("\n".join(q_lines[:num_samples]) + "\n", encoding="utf-8")

    template = build_fau_smoke_template(
        template_id=template_id,
        student_model=student_model,
        teacher_model=tspec.primary,
        teacher_router=tspec.router,
        num_samples=num_samples,
        questions_path=str(q_path),
        corpus_path=str(corpus),
        output_dir=f"./{out_dir.relative_to(REPO_ROOT)}",
        budget=budget,
        workflow=f"hotpot_teacher_guided_b{budget}_plan_review",
        planning_steps=planning_steps,
        max_plan_steps=budget,
        disclose_budget=False,
    )
    template["mode_config"]["guidance"]["level"] = guidance_level
    template["mode_config"]["teacher_max_tokens"] = tspec.max_tokens
    template["mode_config"]["teacher_max_tokens_retry"] = tspec.max_tokens_retry
    # Dataset provenance: without this every episode would be labelled hotpotqa.
    template["mode_config"]["dataset"] = dataset_name

    tpath = TEMPLATES_DIR / f"{template_id}.yaml"
    tpath.write_text(yaml.dump(template, sort_keys=False), encoding="utf-8")
    log_path = out_dir / "run.log"
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                [sys.executable, "-m", "agentsim.cli", "simulate", template_id],
                cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT,
            )
        rc = proc.returncode
    finally:
        tpath.unlink(missing_ok=True)

    elapsed = round(time.time() - t0, 1)
    problems, stats = audit_episodes(out_dir, dataset_name, num_samples)
    if rc != 0:
        problems.insert(0, f"{cell}: simulate exited {rc} (see {log_path})")
    return {
        "cell": cell, "teacher": teacher, "student": student,
        "student_model": student_model, "student_mode": student_mode,
        "rc": rc, "elapsed_s": elapsed, "problems": problems, "stats": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--teacher", nargs="*", default=None, help="default: all teachers")
    ap.add_argument("--student", nargs="*", default=None, help="default: all students")
    ap.add_argument("--student-mode", choices=["vllm", "api"], default="vllm")
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--planning-steps", type=int, default=3)
    ap.add_argument("--guidance-level", type=int, default=3)
    ap.add_argument("--dataset-name", default="hotpotqa")
    ap.add_argument("--questions",
                    default="data/datasets/tg_smoke/hotpotqa/hotpotqa_validation_questions.jsonl")
    ap.add_argument("--corpus",
                    default="data/datasets/tg_smoke/hotpotqa/hotpotqa_validation_corpus.jsonl")
    ap.add_argument("--out", default="data/simulation_output/matrix_smoke")
    args = ap.parse_args()

    teachers = args.teacher or list(TEACHERS)
    students = args.student or list(STUDENTS)
    questions, corpus = Path(args.questions), Path(args.corpus)
    for path in (questions, corpus):
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/prepare_dataset.py first")

    out_root = Path(args.out)
    print(f"=== matrix smoke: {len(teachers)} teacher(s) x {len(students)} student(s) "
          f"x {args.num_samples} episodes | student-mode={args.student_mode} ===")
    if args.student_mode == "api":
        print(f"    NOTE: students substituted by {API_STUDENT}; vLLM serving NOT covered")

    results: List[Dict[str, Any]] = []
    for teacher in teachers:
        for student in students:
            print(f"\n--- {teacher} x {student} ---")
            result = run_cell(
                teacher, student, questions, corpus, out_root,
                num_samples=args.num_samples, student_mode=args.student_mode,
                budget=args.budget, planning_steps=args.planning_steps,
                guidance_level=args.guidance_level, dataset_name=args.dataset_name,
            )
            results.append(result)
            stats = result["stats"]
            status = "ok  " if not result["problems"] else "FAIL"
            print(f"  [{status}] {result['elapsed_s']:>6}s  "
                  f"episodes={stats.get('episodes', 0)} "
                  f"correct={stats.get('answer_correct', 0)} "
                  f"parse_fail={stats.get('step_parse_failure_rate')} "
                  f"teachers={list(stats.get('teachers', {}))}")
            for problem in result["problems"][:4]:
                print(f"         - {problem}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "matrix_smoke_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    failed = [r for r in results if r["problems"]]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} cells ok ===")
    print(f"report -> {out_root / 'matrix_smoke_report.json'}")
    if failed:
        print(f"FAILING CELLS: {[r['cell'] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
