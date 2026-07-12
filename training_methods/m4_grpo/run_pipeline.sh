#!/usr/bin/env bash
# m4_grpo end-to-end pipeline: question pool -> GRPO training (on top of the m1
# adapter if available) -> eval (dev + golden-100) -> results report.
#
# Usage:
#   bash training_methods/m4_grpo/run_pipeline.sh              # full (~24 h at 100 updates)
#   SMOKE=1 bash training_methods/m4_grpo/run_pipeline.sh      # validation (~20 min)
#   UPDATES=50 GPU=3 bash training_methods/m4_grpo/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
M=training_methods/m4_grpo
SMOKE="${SMOKE:-0}"
UPDATES="${UPDATES:-100}"
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

log "=== m4_grpo pipeline start (GPU=$GPU SMOKE=$SMOKE UPDATES=$UPDATES) ==="

log "[1/4] building question pool + golden-100"
$PY training_methods/common/build_golden100.py
$PY $M/build_dataset.py

INIT_ADAPTER="${INIT_ADAPTER:-$(ls -dt training_methods/m1_sft/runs/*/adapter 2>/dev/null | head -1 || true)}"
INIT_ARG=""
if [ -n "$INIT_ADAPTER" ]; then
  log "starting from SFT adapter: $INIT_ADAPTER"
  INIT_ARG="--init-adapter $INIT_ADAPTER"
else
  log "WARNING: no m1 adapter found — GRPO from the raw base model (not recommended)"
fi

log "[2/4] GRPO training"
if [ "$SMOKE" = "1" ]; then
  $PY $M/train_grpo.py $INIT_ARG --smoke
else
  $PY $M/train_grpo.py $INIT_ARG --updates "$UPDATES"
fi
ADAPTER=$(ls -dt $M/runs/*/adapter | head -1)
log "adapter: $ADAPTER"

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

log "[3/4] evals: dev + GOLDEN-100"
$PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$ADAPTER" \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag grpo --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"
$PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$ADAPTER" \
  --questions $GOLD/golden100_questions.jsonl --corpus $GOLD/golden100_corpus.jsonl \
  --out $M/runs/eval_golden100 --tag grpo --budget 4 $EVAL_LIMIT --batch-size "${EVAL_BATCH:-8}"

log "[4/4] results report"
$PY training_methods/common/compare_evals.py --title "m4_grpo results ($TS)" \
  --out $M/runs/results_$TS.md \
  dev_grpo="$(ls -dt $M/runs/eval_dev/*_grpo | head -1)" \
  golden_grpo="$(ls -dt $M/runs/eval_golden100/*_grpo | head -1)"

log "=== m4_grpo pipeline COMPLETE — report: $M/runs/results_$TS.md ==="
