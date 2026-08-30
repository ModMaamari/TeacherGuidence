"""
Escalating retry campaign for the hard questions no trace-collection run answered.

Input: the questions whose best_answer_score is 0 in every run of the 3000x6 matrix
(no teacher-correct answer anywhere). Three stages, each only on what is still wrong:

    xr0  budget 4 (disclosed), 3 planning rounds -- repeated up to --xr0-repeats times,
         dropping a question as soon as one attempt is teacher-correct
    xr1  budget 20 (hidden), 5 planning rounds, one attempt on xr0 leftovers
    xr2  budget 30 (hidden), 5 planning rounds, one attempt on xr1 leftovers

Every answered question contributes its best attempt (by best_answer_score) to
``ds4_hard_q_answers_<nnn>.jsonl`` where nnn is the number answered out of the input.

Progress is written to <out-root>/status.json after every batch so the campaign can be
monitored live. Each batch is a normal run_batched_parallel run under
<out-root>/<stage>_r<round>/ with the usual live files.

Usage:
    python scripts/run_hard_questions_xr.py --out-root data/simulation_output/hard_q_xr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.best_answer_score import (  # noqa: E402
    best_answer_score,
    is_teacher_correct,
    load_run_episodes,
)

TRAIN_QUESTIONS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl"
TRAIN_CORPUS = REPO_ROOT / "data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl"
SCORE_MATRIX = REPO_ROOT / "reports/best_answer_v1/score_matrix.csv"

STAGES = {
    "xr0": {"budget": 4, "planning": 3, "hidden": False, "context": 16384,
            "workflow": "hotpot_teacher_guided_b4_plan_review"},
    "xr1": {"budget": 20, "planning": 5, "hidden": True, "context": 32768,
            "workflow": "hotpot_teacher_guided_b20_plan_review"},
    "xr2": {"budget": 30, "planning": 5, "hidden": True, "context": 32768,
            "workflow": "hotpot_teacher_guided_b30_plan_review"},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%F %T")


def hard_question_qids() -> List[str]:
    """Questions whose score is 0 in every run of the score matrix."""
    qids = []
    with open(SCORE_MATRIX, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(float(v) == 0.0 for k, v in row.items() if k != "qid"):
                qids.append(row["qid"])
    return qids


def write_question_file(qids: List[str], path: Path) -> None:
    wanted = set(qids)
    rows = [l for l in TRAIN_QUESTIONS.read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["id"] in wanted]
    if len(rows) != len(qids):
        raise RuntimeError(f"found {len(rows)} question rows for {len(qids)} qids")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def pick_gpus(n_questions: int, max_gpus: int = 4):
    """Top free GPUs and a worker count that leaves no worker with an empty shard."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    free = sorted((int(m), int(i)) for i, m in
                  (l.split(",") for l in out.strip().splitlines()))
    gpus = [str(i) for _, i in free[:max(1, min(max_gpus, math.ceil(n_questions / 8)))]]
    wpg = max(1, min(8, n_questions // len(gpus)))
    return ",".join(gpus), wpg


def run_batch(qids: List[str], stage: str, rnd: int, out_root: Path) -> Dict[str, Dict[str, Any]]:
    """One run_batched_parallel invocation; returns qid -> episode for what it produced."""
    cfg = STAGES[stage]
    batch_dir = out_root / f"{stage}_r{rnd}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    qfile = batch_dir / "questions.jsonl"
    write_question_file(qids, qfile)

    gpu_ids, wpg = pick_gpus(len(qids))
    env = dict(os.environ, FAU_TIMEOUT="45", CUSTOM_TIMEOUT="180",
               OLLAMA_CONTEXT_LENGTH=str(cfg["context"]))
    cmd = [
        sys.executable, "scripts/run_batched_parallel.py",
        "--num-samples", str(len(qids)), "--workers-per-gpu", str(wpg),
        "--gpu-ids", gpu_ids, "--base-port", "11800",
        "--student", "ollama/granite4.1:3b",
        "--budget", str(cfg["budget"]),
        "--planning-steps", str(cfg["planning"]),
        "--max-plan-steps", str(cfg["budget"]),
        "--workflow", cfg["workflow"],
        "--questions", str(qfile), "--corpus", str(TRAIN_CORPUS),
        "--out-root", str(batch_dir.relative_to(REPO_ROOT)),
        "--tag", f"hardq_{stage}_r{rnd}",
    ]
    if cfg["hidden"]:
        cmd.append("--hidden-budget")
    print(f"[{_now()}] {stage} round {rnd}: {len(qids)} questions | GPUs {gpu_ids} x{wpg}", flush=True)
    with open(batch_dir / "runner.log", "w") as log:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    # scan the whole batch dir (not just run/) so episodes count even if consolidation fails
    return load_run_episodes(batch_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="data/simulation_output/hard_q_xr")
    ap.add_argument("--xr0-repeats", type=int, default=10)
    ap.add_argument("--ds-dir", default="data/datasets/best_answer_v1")
    args = ap.parse_args()
    out_root = REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    qids = hard_question_qids()
    (out_root / "hard_question_ids.txt").write_text("\n".join(qids) + "\n")
    print(f"[{_now()}] {len(qids)} hard questions (0 in all runs of the score matrix)", flush=True)

    answered: Dict[str, Dict[str, Any]] = {}  # qid -> best episode (with provenance)
    status_path = out_root / "status.json"

    def note(ep: Dict[str, Any], stage: str, rnd: int) -> None:
        score, comps = best_answer_score(ep)
        prev = answered.get(ep["qid"])
        if prev is None or score > prev["bas_score"]:
            row = {k: v for k, v in ep.items() if k != "_sample_dir"}
            row.update(bas_run=f"{stage}_r{rnd}", bas_score=round(score, 6),
                       bas_components={k: round(v, 6) for k, v in comps.items()},
                       bas_sample_dir=ep.get("_sample_dir", ""))
            answered[ep["qid"]] = row

    def write_status(stage: str, rnd: int, remaining: int) -> None:
        status_path.write_text(json.dumps({
            "updated_at": _now(), "stage": stage, "round": rnd,
            "total_hard_questions": len(qids),
            "answered": len(answered), "remaining_unanswered": remaining,
        }, indent=2))

    remaining = list(qids)

    # Resume: fold in any batches that already ran (answered questions leave the queue,
    # already-used xr0 rounds count toward the repeat budget).
    xr0_rounds_done = 0
    resumed_stages = set()
    for batch_dir in sorted(out_root.iterdir()) if out_root.exists() else []:
        parts = batch_dir.name.rsplit("_r", 1)
        if not batch_dir.is_dir() or len(parts) != 2 or parts[0] not in STAGES:
            continue
        stage, rnd = parts[0], int(parts[1])
        episodes = load_run_episodes(batch_dir)
        if not episodes:
            continue
        hit = 0
        for qid in list(remaining):
            ep = episodes.get(qid)
            if ep is not None and is_teacher_correct(ep):
                note(ep, stage, rnd)
                remaining.remove(qid)
                hit += 1
        if stage == "xr0":
            xr0_rounds_done = max(xr0_rounds_done, rnd)
        resumed_stages.add(stage)
        print(f"[{_now()}] resumed {batch_dir.name}: {len(episodes)} episodes, "
              f"{hit} newly answered, remaining {len(remaining)}", flush=True)

    # xr0: repeat until answered, at most --xr0-repeats rounds
    for rnd in range(xr0_rounds_done + 1, args.xr0_repeats + 1):
        if not remaining:
            break
        episodes = run_batch(remaining, "xr0", rnd, out_root)
        still = []
        for qid in remaining:
            ep = episodes.get(qid)
            if ep is not None and is_teacher_correct(ep):
                note(ep, "xr0", rnd)
            else:
                still.append(qid)
        remaining = still
        write_status("xr0", rnd, len(remaining))
        print(f"[{_now()}] xr0 r{rnd}: answered so far {len(answered)}, remaining {len(remaining)}", flush=True)

    # xr1 / xr2: one escalating attempt each on what is left
    for stage in ("xr1", "xr2"):
        if not remaining:
            break
        if stage in resumed_stages:
            continue  # this stage already ran; its failures stay for the next stage
        episodes = run_batch(remaining, stage, 1, out_root)
        still = []
        for qid in remaining:
            ep = episodes.get(qid)
            if ep is not None and is_teacher_correct(ep):
                note(ep, stage, 1)
            else:
                still.append(qid)
        remaining = still
        write_status(stage, 1, len(remaining))
        print(f"[{_now()}] {stage}: answered so far {len(answered)}, remaining {len(remaining)}", flush=True)

    # ds4: best answer per answered question
    nnn = len(answered)
    ds_dir = REPO_ROOT / args.ds_dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ds_dir / f"ds4_hard_q_answers_{nnn}.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for qid in sorted(answered):
            f.write(json.dumps(answered[qid], ensure_ascii=False) + "\n")
    stage_counts: Dict[str, int] = {}
    for row in answered.values():
        stage_counts[row["bas_run"]] = stage_counts.get(row["bas_run"], 0) + 1
    summary = {
        "total_hard_questions": len(qids),
        "answered": nnn,
        "still_unanswered": len(remaining),
        "answered_by_batch": dict(sorted(stage_counts.items())),
        "dataset": str(ds_path.relative_to(REPO_ROOT)),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{_now()}] DONE: {json.dumps(summary, indent=2)}", flush=True)
    print("HARD QUESTION CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
