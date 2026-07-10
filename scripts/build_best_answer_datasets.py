"""
Build the three best-answer trace datasets from the six granite trace collections.

    ds1  horizontal best: for each question, the best-scoring answer across the runs
         (questions where no run is teacher-correct are skipped)
    ds2  vertical best: the single run with the highest mean best_answer_score over
         all questions (wrong answers counted as 0), keeping its teacher-correct episodes
    ds3  overall best: the top-N episodes by best_answer_score over all runs x questions
         (the same question may appear once per run, so up to 6 times)

Each output row is the full episode record plus provenance fields:
    bas_run          run label the episode came from
    bas_score        best_answer_score
    bas_components   the six score components
    bas_sample_dir   sample dir inside the run (for tracing back to raw exports)

Usage:
    python scripts/build_best_answer_datasets.py \
        --out-dir data/datasets/best_answer_v1 --top-n 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.best_answer_score import (
    REPO_ROOT,
    build_score_table,
    is_teacher_correct,
    load_all_runs,
)


def _row(episode: Dict[str, Any], run: str, score: float, comps: Dict[str, float]) -> Dict[str, Any]:
    row = {k: v for k, v in episode.items() if k != "_sample_dir"}
    row["bas_run"] = run
    row["bas_score"] = round(score, 6)
    row["bas_components"] = {k: round(v, 6) for k, v in comps.items()}
    row["bas_sample_dir"] = episode.get("_sample_dir", "")
    return row


def _rank_key(loaded, scores, comps, qid: str, label: str, label_order: List[str]):
    """Sort key: higher score, then fewer steps, then higher f1, then cheaper run."""
    ep = loaded[label][qid]
    return (
        scores[qid][label],
        -(ep.get("used_steps") or len(ep.get("steps", []) or [])),
        comps[qid][label]["f1"],
        -label_order.index(label),
    )


def build_datasets(top_n: int = 3000):
    loaded = load_all_runs()
    labels = list(loaded)
    qids, scores, comps = build_score_table(loaded)

    # ds1: per-question argmax over runs, positive scores only
    ds1, ds1_winners = [], {l: 0 for l in labels}
    for qid in qids:
        best = max(labels, key=lambda l: _rank_key(loaded, scores, comps, qid, l, labels))
        if scores[qid][best] <= 0:
            continue
        ds1.append(_row(loaded[best][qid], best, scores[qid][best], comps[qid][best]))
        ds1_winners[best] += 1

    # ds2: run with the highest mean score over all questions, its teacher-correct episodes
    run_means = {l: sum(scores[q][l] for q in qids) / len(qids) for l in labels}
    best_run = max(labels, key=lambda l: run_means[l])
    ds2 = [
        _row(loaded[best_run][qid], best_run, scores[qid][best_run], comps[qid][best_run])
        for qid in qids
        if is_teacher_correct(loaded[best_run][qid])
    ]

    # ds3: global top-N episodes across all runs x questions
    pool = [(qid, l) for qid in qids for l in labels if scores[qid][l] > 0]
    pool.sort(key=lambda t: _rank_key(loaded, scores, comps, t[0], t[1], labels), reverse=True)
    ds3 = [
        _row(loaded[l][qid], l, scores[qid][l], comps[qid][l])
        for qid, l in pool[:top_n]
    ]

    summary = {
        "runs": labels,
        "run_mean_scores": {l: round(run_means[l], 6) for l in labels},
        "ds1": {
            "n": len(ds1),
            "skipped_no_correct_answer": len(qids) - len(ds1),
            "winner_counts": ds1_winners,
            "mean_score": round(sum(r["bas_score"] for r in ds1) / max(1, len(ds1)), 6),
        },
        "ds2": {
            "n": len(ds2),
            "run": best_run,
            "mean_score": round(sum(r["bas_score"] for r in ds2) / max(1, len(ds2)), 6),
        },
        "ds3": {
            "n": len(ds3),
            "run_counts": {l: sum(1 for r in ds3 if r["bas_run"] == l) for l in labels},
            "unique_questions": len({r["qid"] for r in ds3}),
            "mean_score": round(sum(r["bas_score"] for r in ds3) / max(1, len(ds3)), 6),
            "min_score": round(min((r["bas_score"] for r in ds3), default=0.0), 6),
        },
    }
    return {"ds1_horizontal_best": ds1, "ds2_vertical_best": ds2, "ds3_overall_best": ds3}, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/datasets/best_answer_v1")
    ap.add_argument("--top-n", type=int, default=3000)
    args = ap.parse_args()
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    datasets, summary = build_datasets(top_n=args.top_n)
    for name, rows in datasets.items():
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(rows)} rows)")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
