#!/usr/bin/env bash
# m5_rlaif_prm end-to-end pipeline:
#   PRM dataset -> PRM training -> PRM correlation gate -> RLAIF (GRPO with PRM
#   reward, reusing the m4 trainer; policy GPU + PRM GPU) -> eval -> report.
#
# Usage:
#   bash training_methods/m5_rlaif_prm/run_pipeline.sh              # full (~30 h)
#   SMOKE=1 bash training_methods/m5_rlaif_prm/run_pipeline.sh      # validation (~35 min)
#   GPU=0 PRM_GPU=1 UPDATES=60 bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
GPU="${GPU:-0}"
PRM_GPU="${PRM_GPU:-1}"
M=training_methods/m5_rlaif_prm
SMOKE="${SMOKE:-0}"
UPDATES="${UPDATES:-60}"
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

log "=== m5_rlaif_prm pipeline start (GPU=$GPU PRM_GPU=$PRM_GPU SMOKE=$SMOKE) ==="

log "[1/6] building PRM dataset (all six runs) + golden-100"
$PY training_methods/common/build_golden100.py
[ -f $M/data/prm_train.jsonl ] || $PY $M/build_dataset.py

log "[2/6] training the PRM"
SMOKE_ARG=$([ "$SMOKE" = "1" ] && echo "--smoke" || echo "")
CUDA_VISIBLE_DEVICES=$GPU $PY $M/train_prm.py $SMOKE_ARG
PRM_ADAPTER=$(ls -dt $M/runs/*prm*/adapter | head -1)
log "PRM adapter: $PRM_ADAPTER"

log "[3/6] PRM correlation gate (held-out teacher scores)"
CORR_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 12" || echo "--limit 500")
CUDA_VISIBLE_DEVICES=$GPU $PY $M/eval_prm.py --adapter "$PRM_ADAPTER" $CORR_LIMIT

log "[4/6] RLAIF: GRPO with the PRM as reward (policy on GPU $GPU, PRM on GPU $PRM_GPU)"
$PY training_methods/m4_grpo/build_dataset.py
INIT_ADAPTER="${INIT_ADAPTER:-$(ls -dt training_methods/m1_sft/runs/*/adapter 2>/dev/null | head -1 || true)}"
INIT_ARG=""
[ -n "$INIT_ADAPTER" ] && INIT_ARG="--init-adapter $INIT_ADAPTER"
GRPO_ARGS=$([ "$SMOKE" = "1" ] && echo "--smoke" || echo "--updates $UPDATES")
PRM_ADAPTER="$PRM_ADAPTER" PRM_DEVICE=cuda:1 CUDA_VISIBLE_DEVICES=$GPU,$PRM_GPU \
  PRM_ADAPTER="$PRM_ADAPTER" $PY training_methods/m4_grpo/train_grpo.py $INIT_ARG $GRPO_ARGS \
  --reward-fn training_methods.m5_rlaif_prm.prm_reward.episode_reward \
  --out-base $M/runs --tag rlaif
ADAPTER=$(ls -dt $M/runs/*rlaif*/adapter | head -1)
log "RLAIF adapter: $ADAPTER"

EVAL_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 3" || echo "")
GOLD=training_methods/common/data/golden100
DEV_Q=training_methods/m1_sft/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

MODEL_ARG=""
if [ -n "$INIT_ADAPTER" ]; then
  MERGED=$M/runs/merged_init_$TS
  log "merging init adapter for eval base: $MERGED"
  $PY - "$INIT_ADAPTER" "$MERGED" <<'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
init, out = sys.argv[1], sys.argv[2]
m = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.1-3b", dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, init).merge_and_unload()
m.save_pretrained(out); AutoTokenizer.from_pretrained(init).save_pretrained(out)
print("merged ->", out)
PYEOF
  MODEL_ARG="--model $MERGED"
fi

log "[5/6] evals: dev + GOLDEN-100"
CUDA_VISIBLE_DEVICES=$GPU $PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$ADAPTER" \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag rlaif --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"
CUDA_VISIBLE_DEVICES=$GPU $PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$ADAPTER" \
  --questions $GOLD/golden100_questions.jsonl --corpus $GOLD/golden100_corpus.jsonl \
  --out $M/runs/eval_golden100 --tag rlaif --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"

log "[6/6] results report"
$PY training_methods/common/compare_evals.py --title "m5_rlaif_prm results ($TS)" \
  --out $M/runs/results_$TS.md \
  dev_rlaif="$(ls -dt $M/runs/eval_dev/*_rlaif | head -1)" \
  golden_rlaif="$(ls -dt $M/runs/eval_golden100/*_rlaif | head -1)"

log "=== m5_rlaif_prm pipeline COMPLETE — report: $M/runs/results_$TS.md ==="
