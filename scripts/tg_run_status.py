"""Answered/unanswered tracking for a Teacher-Guidance run.

The episodes on disk are the single source of truth: a question is ANSWERED once an
episode row for its qid exists anywhere under the run's output root (whether still in a
worker shard ``w*/`` or already consolidated into ``<run_name>/``). Everything else in the
requested slice is UNANSWERED. Nothing is cached, so the status is always accurate even if
a run was killed mid-flight, and a resumed run simply re-derives the remaining work.

This module is imported by scripts/run_tg_vllm.py for resume, and is also a CLI to
monitor a run at any time:

    .venv/bin/python scripts/tg_run_status.py \
        --out-root data/simulation_output/tg_m3_granite_b3 \
        --questions data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl \
        --num-samples 3000 [--json] [--list-unanswered N]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"


def read_questions(questions_path: Path, num_samples: int | None = None) -> List[Dict[str, Any]]:
    """The requested slice of the question file, in file order (the runner shards the
    first ``num_samples`` questions, so the slice defines the run's scope)."""
    rows = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows[:num_samples] if num_samples else rows


def scan_episodes(out_root: Path) -> Dict[str, Dict[str, Any]]:
    """Map qid -> episode for every episode written under ``out_root``.

    Scans both worker shards and the consolidated run dir; if a qid somehow appears twice
    (e.g. an interrupted run re-ran it), the last one wins.
    """
    episodes: Dict[str, Dict[str, Any]] = {}
    if not out_root.exists():
        return episodes
    for ep_file in out_root.rglob(EPISODE_FILENAME):
        try:
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ep = json.loads(line)
                qid = ep.get("qid")
                if qid:
                    episodes[str(qid)] = ep
        except (OSError, json.JSONDecodeError):
            # A worker may be mid-write; a partial file just means "not answered yet".
            continue
    return episodes


def compute_status(out_root: Path, questions_path: Path, num_samples: int | None = None) -> Dict[str, Any]:
    questions = read_questions(questions_path, num_samples)
    wanted = [str(q["id"]) for q in questions]
    episodes = scan_episodes(out_root)

    answered = [q for q in wanted if q in episodes]
    unanswered = [q for q in wanted if q not in episodes]

    done = [episodes[q] for q in answered]
    correct = sum(1 for e in done if (e.get("final_metrics") or {}).get("answer_correct"))
    teacher_judged = sum(1 for e in done if (e.get("final_metrics") or {}).get("teacher_answer_correct"))
    unknown = sum(1 for e in done if str(e.get("final_answer", "")).strip().lower() == "unknown")

    teachers: collections.Counter = collections.Counter()
    for e in done:
        for m in e.get("teacher_models_used") or []:
            teachers[m] += 1
    steps: collections.Counter = collections.Counter(e.get("used_steps") for e in done)

    return {
        "out_root": str(out_root),
        "questions": str(questions_path),
        "requested": len(wanted),
        "answered": len(answered),
        "unanswered": len(unanswered),
        "pct_done": (100.0 * len(answered) / len(wanted)) if wanted else 0.0,
        "answer_correct": correct,
        "answer_correct_pct": (100.0 * correct / len(done)) if done else 0.0,
        "teacher_answer_correct": teacher_judged,
        "unknown_answers": unknown,
        "teacher_models_used": dict(teachers),
        "used_steps_distribution": {str(k): v for k, v in sorted(steps.items(), key=lambda x: (x[0] is None, x[0]))},
        "unanswered_qids": unanswered,
        "answered_qids": answered,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="emit the full status as JSON")
    ap.add_argument("--list-unanswered", type=int, default=0, metavar="N",
                    help="also print the first N unanswered qids")
    args = ap.parse_args()

    st = compute_status(Path(args.out_root), Path(args.questions), args.num_samples)
    if args.json:
        print(json.dumps(st, indent=2))
        return

    print(f"=== {st['out_root']} ===")
    print(f"answered  : {st['answered']}/{st['requested']} ({st['pct_done']:.1f}%)")
    print(f"unanswered: {st['unanswered']}")
    if st["answered"]:
        print(f"answer_correct        : {st['answer_correct']}/{st['answered']} ({st['answer_correct_pct']:.1f}%)")
        print(f"teacher_answer_correct: {st['teacher_answer_correct']}")
        print(f"unknown answers       : {st['unknown_answers']}")
        print(f"teacher_models_used   : {st['teacher_models_used']}")
        print(f"used_steps            : {st['used_steps_distribution']}")
    if args.list_unanswered:
        print(f"first {args.list_unanswered} unanswered: {st['unanswered_qids'][:args.list_unanswered]}")


if __name__ == "__main__":
    main()
