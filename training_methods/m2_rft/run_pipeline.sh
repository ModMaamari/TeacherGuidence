#!/usr/bin/env bash
# m2_rft end-to-end pipeline (ONE rejection-sampling round):
#   fresh questions -> rollouts with the current policy -> filter correct ->
#   merge with the m1 dataset -> retrain -> eval (dev + golden-100) -> report.
# Re-run with ROUND=2 (and ADAPTER pointing at round-1's adapter) to iterate.
#
# Usage:
#   bash training_methods/m2_rft/run_pipeline.sh              # full round (~10-14 h)
#   SMOKE=1 bash training_methods/m2_rft/run_pipeline.sh      # validation (~25 min)
#   ROUND=2 ADAPTER=<round1 adapter> bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
M=training_methods/m2_rft
SMOKE="${SMOKE:-0}"
ROUND="${ROUND:-1}"
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

log "=== m2_rft pipeline start (GPU=$GPU SMOKE=$SMOKE ROUND=$ROUND) ==="

log "[1/6] fresh question pool + golden-100"
$PY training_methods/common/build_golden100.py
# smoke uses its own tiny pool so it never pollutes the full 2000-question pool
FRESH_DIR=$([ "$SMOKE" = "1" ] && echo "$M/data/fresh_smoke" || echo "$M/data/fresh")
LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 20" || echo "--limit 2000")
[ -f $FRESH_DIR/fresh_questions.jsonl ] || $PY $M/prepare_fresh_questions.py $LIMIT --out "$FRESH_DIR"

# rollout policy: explicit ADAPTER > newest m1 adapter > base model
ADAPTER="${ADAPTER:-$(ls -dt training_methods/m1_sft/runs/*/adapter 2>/dev/null | head -1 || true)}"
ADAPTER_ARG=""
[ -n "$ADAPTER" ] && { log "rollout policy adapter: $ADAPTER"; ADAPTER_ARG="--adapter $ADAPTER"; }

log "[2/6] generating rollouts"
ROLLOUT_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 4" || echo "")
$PY $M/generate_rollouts.py $ADAPTER_ARG --tag "r$ROUND" $ROLLOUT_LIMIT \
  --questions "$FRESH_DIR/fresh_questions.jsonl" --corpus "$FRESH_DIR/fresh_corpus.jsonl" \
  --batch-size "${ROLLOUT_BATCH:-8}" \
  $([ "$SMOKE" = "1" ] && echo "--budget 2 --n-per-question 1" || echo "")
ROLLOUTS=$(ls -dt $M/runs/rollouts/*_r$ROUND*/rollouts.jsonl | head -1)
log "rollouts: $ROLLOUTS"

log "[3/6] filtering + merging with the m1 dataset"
$PY $M/filter_rollouts.py --rollouts "$ROLLOUTS" --out $M/data/round$ROUND \
  --merge-with training_methods/m1_sft/data/train.jsonl

log "[4/6] retraining on the merged mixture (reuses the m1 trainer)"
TRAIN_ARGS=$([ "$SMOKE" = "1" ] && echo "--smoke" || echo "--epochs 1")
$PY training_methods/m1_sft/train.py --train-file $M/data/round$ROUND/merged_train.jsonl \
  --out-base $M/runs --tag "rft_round$ROUND" $TRAIN_ARGS
NEW_ADAPTER=$(ls -dt $M/runs/*rft_round$ROUND*/adapter | head -1)
log "new adapter: $NEW_ADAPTER"

EVAL_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 3" || echo "")
GOLD=training_methods/common/data/golden100
DEV_Q=training_methods/m1_sft/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

log "[5/6] evals: dev + GOLDEN-100"
$PY training_methods/common/eval_agent.py --adapter "$NEW_ADAPTER" \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag "rft$ROUND" --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"
$PY training_methods/common/eval_agent.py --adapter "$NEW_ADAPTER" \
  --questions $GOLD/golden100_questions.jsonl --corpus $GOLD/golden100_corpus.jsonl \
  --out $M/runs/eval_golden100 --tag "rft$ROUND" --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"

log "[6/6] results report"
$PY training_methods/common/compare_evals.py --title "m2_rft round $ROUND results ($TS)" \
  --out $M/runs/results_round${ROUND}_$TS.md \
  dev_rft="$(ls -dt $M/runs/eval_dev/*_rft$ROUND | head -1)" \
  golden_rft="$(ls -dt $M/runs/eval_golden100/*_rft$ROUND | head -1)"

log "=== m2_rft round $ROUND COMPLETE — report: $M/runs/results_round${ROUND}_$TS.md ==="
