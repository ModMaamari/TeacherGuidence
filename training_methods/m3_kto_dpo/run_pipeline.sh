#!/usr/bin/env bash
# m3_kto_dpo end-to-end pipeline: dataset -> KTO train (on top of the m1 adapter if
# available) -> DPO train -> eval (dev + golden-100) -> results report.
#
# Usage:
#   bash training_methods/m3_kto_dpo/run_pipeline.sh              # full (~12-16 h)
#   SMOKE=1 bash training_methods/m3_kto_dpo/run_pipeline.sh      # validation (~25 min)
#   INIT_ADAPTER=path/to/m1/adapter bash ...                      # explicit SFT init
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
M=training_methods/m3_kto_dpo
SMOKE="${SMOKE:-0}"
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

log "=== m3_kto_dpo pipeline start (GPU=$GPU SMOKE=$SMOKE) ==="

log "[1/5] building datasets (KTO labels + DPO pairs; scans all six runs, ~1 min)"
$PY training_methods/common/build_golden100.py
$PY $M/build_dataset.py

# Stack on the newest m1 SFT adapter unless the caller overrides / none exists.
INIT_ADAPTER="${INIT_ADAPTER:-$(ls -dt training_methods/m1_sft/runs/*/adapter 2>/dev/null | head -1 || true)}"
INIT_ARG=""
if [ -n "$INIT_ADAPTER" ]; then
  log "stacking on SFT adapter: $INIT_ADAPTER"
  INIT_ARG="--init-adapter $INIT_ADAPTER"
else
  log "WARNING: no m1 adapter found — training preference methods from the raw base model"
fi

SMOKE_ARG=$([ "$SMOKE" = "1" ] && echo "--smoke" || echo "")
log "[2/5] KTO training"
$PY $M/kto_train.py $INIT_ARG $SMOKE_ARG
KTO_ADAPTER=$(ls -dt $M/runs/*kto*/adapter | head -1)

log "[3/5] DPO training"
$PY $M/dpo_train.py $INIT_ARG $SMOKE_ARG
DPO_ADAPTER=$(ls -dt $M/runs/*dpo*/adapter | head -1)

EVAL_LIMIT=$([ "$SMOKE" = "1" ] && echo "--limit 3" || echo "")
GOLD_Q=training_methods/common/data/golden100/golden100_questions.jsonl
GOLD_C=training_methods/common/data/golden100/golden100_corpus.jsonl
DEV_Q=training_methods/m1_sft/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

# NOTE: kto/dpo adapters were trained on (base + merged init adapter); evaluate them
# on the same composition via --model of a merged checkpoint when INIT_ADAPTER is set.
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

log "[4/5] evals: dev + GOLDEN-100 for kto and dpo adapters"
for NAME in kto dpo; do
  A=$([ "$NAME" = kto ] && echo "$KTO_ADAPTER" || echo "$DPO_ADAPTER")
  $PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$A" \
    --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag $NAME --budget 4 $EVAL_LIMIT
  $PY training_methods/common/eval_agent.py $MODEL_ARG --adapter "$A" \
    --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden100 --tag $NAME --budget 4 $EVAL_LIMIT
done

log "[5/5] results report"
$PY training_methods/common/compare_evals.py --title "m3_kto_dpo results ($TS)" \
  --out $M/runs/results_$TS.md \
  dev_kto="$(ls -dt $M/runs/eval_dev/*_kto | head -1)" \
  dev_dpo="$(ls -dt $M/runs/eval_dev/*_dpo | head -1)" \
  golden_kto="$(ls -dt $M/runs/eval_golden100/*_kto | head -1)" \
  golden_dpo="$(ls -dt $M/runs/eval_golden100/*_dpo | head -1)"

log "=== m3_kto_dpo pipeline COMPLETE — report: $M/runs/results_$TS.md ==="
