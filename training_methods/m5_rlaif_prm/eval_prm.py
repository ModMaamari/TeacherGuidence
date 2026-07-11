"""Evaluate the trained PRM against held-out teacher step scores.

Reports Pearson/Spearman correlation, MAE, and ROC-AUC of the binarized judgment
(teacher score >= 0.5). This is the go/no-go gate before spending GPU-days on
PRM-reward RL: a PRM that can't track the teacher offline will only mislead RL.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/m5_rlaif_prm/eval_prm.py \
        --adapter <prm adapter> [--limit 500]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dev-file", default=str(Path(__file__).parent / "data" / "prm_dev.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "runs" / "prm_eval"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out, "corr")
    log = setup_logger("m5_eval_prm", run_dir / "eval.log")
    log.info(f"args: {vars(args)}")

    import torch  # noqa: F401  (ensures CUDA context before scorer)
    from training_methods.m5_rlaif_prm.prm_score import PRMScorer

    rows = []
    with open(args.dev_file) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break
    log.info(f"scoring {len(rows)} held-out judge examples")

    scorer = PRMScorer(args.adapter, device=args.device)
    preds, golds = [], []
    for k, r in enumerate(rows, 1):
        # the stored prompt already contains the full judge message; re-scoring via
        # the raw user content keeps prompts identical to training
        user = r["prompt"][1]["content"]
        state, action_part = user.split("=== PROPOSED ACTION ===", 1)
        action_json = action_part.strip().split("\n\nRate this action", 1)[0]
        preds.append(scorer.score(state, json.loads(action_json)))
        golds.append(r["metadata"]["teacher_score"])
        if k % 50 == 0:
            log.info(f"[{k}/{len(rows)}]")

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    mae = sum(abs(p - g) for p, g in zip(preds, golds)) / len(preds)
    binary = [1 if g >= 0.5 else 0 for g in golds]
    metrics = {
        "n": len(preds),
        "pearson": round(float(pearsonr(preds, golds)[0]), 4),
        "spearman": round(float(spearmanr(preds, golds)[0]), 4),
        "mae": round(mae, 4),
        "auc_vs_binary_teacher": round(float(roc_auc_score(binary, preds)), 4)
        if 0 < sum(binary) < len(binary) else None,
        "adapter": args.adapter,
    }
    write_json(run_dir / "prm_correlation.json", metrics)
    log.info(f"PRM CORRELATION: {json.dumps(metrics)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
