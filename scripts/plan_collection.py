"""Plan the teacher-guidance collection matrix: which questions each cell runs.

Running every teacher-student pair over every question would be enormous and mostly
redundant, so the corpus is *partitioned*: each of the 12 teacher x student combinations
gets its own batch of questions, plus a shared **anchor set** that every combination sees.

The anchor set is the scientifically important part. Because every combination answers the
same anchor questions, teachers and students can be compared *on identical items* (paired
tests, per-question deltas) instead of across different question samples, while the
non-overlapping remainder keeps question coverage broad and the cost bounded.

Each (combination, question) assignment is then run under several **configs** -- differing
in guidance level and whether plan review is on -- which is what turns 60k assignments into
a 120k-episode corpus and gives the counterfactual supervision ladder its paired structure.

This script only *plans*: it writes a manifest and never calls a model. Feed the manifest
to the collection runner.

Usage::

    .venv/bin/python scripts/plan_collection.py \
        --data-root data/datasets/tg_v1 --out resource_paper/plan \
        --per-combination 5000 --anchor 500 --configs g3_plan g0_plan
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.model_registry import (  # noqa: E402
    STUDENTS,
    TEACHERS,
    combination_id,
    get_student,
    get_teacher,
)
from agentsim.teacher_guidance.provenance import framework_commit  # noqa: E402

#: Named generation configs. Two are used by default; varying ONE axis at a time keeps the
#: contrast interpretable (g3_plan vs g0_plan isolates guidance richness with planning held
#: constant, which is the paper's dose-response comparison).
CONFIGS: Dict[str, Dict[str, Any]] = {
    "g3_plan": {
        "guidance_level": 3, "plan_review": True, "planning_steps": 3,
        "budget": 3, "disclose_budget": False, "skip_teacher": False,
        "description": "diagnostic feedback + plan review (workhorse)",
    },
    "g0_plan": {
        "guidance_level": 0, "plan_review": True, "planning_steps": 3,
        "budget": 3, "disclose_budget": False, "skip_teacher": False,
        "description": "binary score only + plan review (low-information contrast)",
    },
    "g3_noplan": {
        "guidance_level": 3, "plan_review": False, "planning_steps": 0,
        "budget": 3, "disclose_budget": False, "skip_teacher": False,
        "description": "diagnostic feedback, no plan review (isolates planning)",
    },
    "g4_plan": {
        "guidance_level": 4, "plan_review": True, "planning_steps": 3,
        "budget": 3, "disclose_budget": False, "skip_teacher": False,
        "description": "feedback + next-step hint (maximum exposure)",
    },
    "skip_teacher": {
        "guidance_level": 0, "plan_review": True, "planning_steps": 0,
        "budget": 3, "disclose_budget": False, "skip_teacher": True,
        "description": "no teacher at all (control arm)",
    },
}


def load_question_pool(data_root: Path) -> List[Dict[str, str]]:
    """Every prepared question, as ``{"qid", "dataset", "file"}``, in a stable order."""
    pool: List[Dict[str, str]] = []
    for q_path in sorted(data_root.rglob("*_questions.jsonl")):
        for line in q_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pool.append({
                "qid": str(row["id"]),
                "dataset": row.get("source", "unknown"),
                "file": str(q_path.relative_to(data_root)),
            })
    return pool


def _stratified_take(
    by_dataset: Dict[str, List[Dict[str, str]]], cursors: Dict[str, int], n: int
) -> List[Dict[str, str]]:
    """Take ``n`` questions spread proportionally across datasets, without replacement.

    Round-robins across datasets so every combination sees a mix rather than being handed
    one dataset's questions, and advances a per-dataset cursor so batches never overlap
    unless we deliberately make them.
    """
    datasets = [d for d in by_dataset if cursors[d] < len(by_dataset[d])]
    taken: List[Dict[str, str]] = []
    while len(taken) < n and datasets:
        for name in list(datasets):
            if len(taken) >= n:
                break
            idx = cursors[name]
            if idx >= len(by_dataset[name]):
                datasets.remove(name)
                continue
            taken.append(by_dataset[name][idx])
            cursors[name] = idx + 1
    return taken


def assign_batches(
    pool: Sequence[Dict[str, str]],
    combinations: Sequence[str],
    per_combination: int,
    anchor: int,
    seed: int = 13,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    """Split the pool into one batch per combination plus a shared anchor set.

    Returns ``(batches, anchor_questions)``. Every batch has exactly ``per_combination``
    questions: the ``anchor`` shared ones plus unique remainder. Raises if the pool is too
    small to do that without reusing questions across batches.
    """
    if anchor > per_combination:
        raise ValueError("anchor cannot exceed per-combination size")
    unique_needed = len(combinations) * (per_combination - anchor)
    if anchor + unique_needed > len(pool):
        raise ValueError(
            f"pool too small: need {anchor + unique_needed} questions "
            f"({anchor} anchor + {len(combinations)} x {per_combination - anchor} unique) "
            f"but only {len(pool)} available"
        )

    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    by_dataset: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for q in shuffled:
        by_dataset[q["dataset"]].append(q)
    cursors = {d: 0 for d in by_dataset}

    anchor_questions = _stratified_take(by_dataset, cursors, anchor)
    batches: Dict[str, List[Dict[str, str]]] = {}
    for combo in combinations:
        remainder = _stratified_take(by_dataset, cursors, per_combination - anchor)
        batches[combo] = anchor_questions + remainder
    return batches, anchor_questions


def build_plan(
    batches: Dict[str, List[Dict[str, str]]],
    anchor_questions: Sequence[Dict[str, str]],
    config_names: Sequence[str],
    per_combination: int,
    seed: int,
) -> Dict[str, Any]:
    """Assemble the manifest: one cell per (combination, config)."""
    cells = []
    for combo, questions in batches.items():
        teacher_name, student_name = combo.split("__")
        teacher, student = get_teacher(teacher_name), get_student(student_name)
        for config_name in config_names:
            cfg = CONFIGS[config_name]
            cells.append({
                "cell_id": f"{combo}__{config_name}",
                "combination": combo,
                "teacher": teacher_name,
                "teacher_model": teacher.primary,
                "teacher_router": teacher.router,
                "teacher_max_tokens": teacher.max_tokens,
                "student": student_name,
                "student_model": student.served_model,
                "student_hf_id": student.hf_id,
                "config": config_name,
                **{k: v for k, v in cfg.items() if k != "description"},
                "num_questions": len(questions),
                "qids": [q["qid"] for q in questions],
            })

    all_qids = {q["qid"] for qs in batches.values() for q in qs}
    episodes = sum(c["num_questions"] for c in cells)
    return {
        "framework_commit": framework_commit(),
        "seed": seed,
        "teachers": list(TEACHERS),
        "students": list(STUDENTS),
        "combinations": sorted(batches),
        "configs": {name: CONFIGS[name] for name in config_names},
        "per_combination": per_combination,
        "anchor_questions": len(anchor_questions),
        "anchor_qids": [q["qid"] for q in anchor_questions],
        "unique_questions": len(all_qids),
        "assignments": sum(len(qs) for qs in batches.values()),
        "total_episodes": episodes,
        "dataset_mix": dict(collections.Counter(
            q["dataset"] for qs in batches.values() for q in qs
        )),
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", required=True, help="dir of prepared datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-combination", type=int, default=5000)
    ap.add_argument("--anchor", type=int, default=500,
                    help="questions every combination shares, for paired comparison")
    ap.add_argument("--configs", nargs="+", default=["g3_plan", "g0_plan"],
                    choices=sorted(CONFIGS))
    ap.add_argument("--teachers", nargs="+", default=None)
    ap.add_argument("--students", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    teachers = args.teachers or list(TEACHERS)
    students = args.students or list(STUDENTS)
    for name in teachers:
        get_teacher(name)
    for name in students:
        get_student(name)
    combinations = [combination_id(t, s) for t in teachers for s in students]

    pool = load_question_pool(Path(args.data_root))
    if not pool:
        raise SystemExit(f"no prepared questions found under {args.data_root}")
    seen = collections.Counter(q["qid"] for q in pool)
    dupes = [q for q, n in seen.items() if n > 1]
    if dupes:
        raise SystemExit(f"question pool has {len(dupes)} duplicate qid(s), e.g. {dupes[:3]}")

    batches, anchor_questions = assign_batches(
        pool, combinations, args.per_combination, args.anchor, args.seed
    )
    plan = build_plan(batches, anchor_questions, args.configs, args.per_combination, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "collection_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(f"=== collection plan ===")
    print(f"  teachers x students : {len(teachers)} x {len(students)} = {len(combinations)} combinations")
    print(f"  configs             : {', '.join(args.configs)}")
    print(f"  per combination     : {args.per_combination} questions "
          f"({args.anchor} shared anchor + {args.per_combination - args.anchor} unique)")
    print(f"  assignments         : {plan['assignments']}")
    print(f"  TOTAL EPISODES      : {plan['total_episodes']}")
    print(f"  unique questions    : {plan['unique_questions']} (pool had {len(pool)})")
    print(f"  dataset mix         : {plan['dataset_mix']}")
    print(f"  cells               : {len(plan['cells'])}")
    print(f"  -> {plan_path}")


if __name__ == "__main__":
    main()
