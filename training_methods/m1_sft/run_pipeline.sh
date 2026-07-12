#!/usr/bin/env bash
# m1_sft end-to-end pipeline: dataset -> train -> eval (dev + golden-100, base vs
# trained) -> results report. One A100 is enough; set GPU=<id> to choose it.
#
# Usage:
#   bash training_methods/m1_sft/run_pipeline.sh            # full run (~6-8 h total)
#   SMOKE=1 bash training_methods/m1_sft/run_pipeline.sh    # fast validation (~15 min)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
M=training_methods/m1_sft
SMOKE="${SMOKE:-0}"
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

log "=== m1_sft pipeline start (GPU=$GPU SMOKE=$SMOKE) ==="

log "[1/6] building golden-100 (idempotent) + m1 dataset"
$PY training_methods/common/build_golden100.py
$PY $M/build_dataset.py

log "[2/6] training"
if [ "$SMOKE" = "1" ]; then
  $PY $M/train.py --smoke
else
  $PY $M/train.py --epochs 2
fi
ADAPTER=$(ls -dt $M/runs/*/adapter | head -1)
log "adapter: $ADAPTER"

EVAL_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 3" || echo "")
GOLD_Q=training_methods/common/data/golden100/golden100_questions.jsonl
GOLD_C=training_methods/common/data/golden100/golden100_corpus.jsonl
DEV_Q=$M/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

log "[3/6] eval: BASE model on dev questions"
$PY training_methods/common/eval_agent.py --questions $DEV_Q --corpus $TRAIN_C \
  --out $M/runs/eval_dev --tag base --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"
log "[4/6] eval: TRAINED adapter on dev questions"
$PY training_methods/common/eval_agent.py --adapter "$ADAPTER" --questions $DEV_Q \
  --corpus $TRAIN_C --out $M/runs/eval_dev --tag m1 --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"

log "[5/6] GOLDEN TEST: base + trained on the 100 hardest (never-answered) questions"
$PY training_methods/common/eval_agent.py --questions $GOLD_Q --corpus $GOLD_C \
  --out $M/runs/eval_golden100 --tag base --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"
$PY training_methods/common/eval_agent.py --adapter "$ADAPTER" --questions $GOLD_Q \
  --corpus $GOLD_C --out $M/runs/eval_golden100 --tag m1 --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"

log "[6/6] results report"
$PY training_methods/common/compare_evals.py --title "m1_sft results ($TS)" \
  --out $M/runs/results_$TS.md \
  dev_base="$(ls -dt $M/runs/eval_dev/*_base | head -1)" \
  dev_m1="$(ls -dt $M/runs/eval_dev/*_m1 | head -1)" \
  golden_base="$(ls -dt $M/runs/eval_golden100/*_base | head -1)" \
  golden_m1="$(ls -dt $M/runs/eval_golden100/*_m1 | head -1)"

log "=== m1_sft pipeline COMPLETE — report: $M/runs/results_$TS.md ==="
