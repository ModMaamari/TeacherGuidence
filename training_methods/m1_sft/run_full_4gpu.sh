#!/usr/bin/env bash
# m1_sft FULL pipeline accelerated on the top 4 free GPUs:
#   dataset build -> DDP training (torchrun, 4 ranks) -> 4 evals IN PARALLEL
#   (dev-base, dev-m1, golden-base, golden-m1; one GPU each) -> results report.
#
# Usage:  tmux new-session -d -s m1_full 'bash training_methods/m1_sft/run_full_4gpu.sh'
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
TORCHRUN=.venv_train/bin/torchrun
M=training_methods/m1_sft
TS=$(date -u '+%Y%m%dT%H%M%SZ')
log() { echo "[$(date -u '+%F %T')Z] $*"; }

# Top 4 free GPUs at launch (lowest memory.used), fixed for the whole run.
GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr -d ' ' | paste -sd,)
IFS=',' read -r G0 G1 G2 G3 <<< "$GPUS"
log "=== m1_sft FULL 4-GPU pipeline start | GPUs=$GPUS | ts=$TS ==="

log "[1/4] building golden-100 + m1 dataset"
$PY training_methods/common/build_golden100.py
$PY $M/build_dataset.py

log "[2/4] DDP training on 4 GPUs (torchrun): 2 epochs, effective batch 2x4x2=16"
RUN_DIR=$M/runs/${TS}_train4gpu
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node=4 --master_port=29713 \
  $M/train.py --epochs 2 --batch-size 2 --grad-accum 2 --run-dir "$RUN_DIR" \
  2>&1 | tee "$M/runs/${TS}_train4gpu_console.log"
ADAPTER=$RUN_DIR/adapter
[ -d "$ADAPTER" ] || { log "FATAL: adapter not found at $ADAPTER"; exit 1; }
log "adapter: $ADAPTER"

GOLD_Q=training_methods/common/data/golden100/golden100_questions.jsonl
GOLD_C=training_methods/common/data/golden100/golden100_corpus.jsonl
DEV_Q=$M/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

log "[3/4] four evals in parallel (one GPU each): dev/golden x base/trained"
CUDA_VISIBLE_DEVICES=$G0 $PY training_methods/common/eval_agent.py \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag base --budget 4 \
  > $M/runs/${TS}_eval_dev_base.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=$G1 $PY training_methods/common/eval_agent.py --adapter "$ADAPTER" \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag m1 --budget 4 \
  > $M/runs/${TS}_eval_dev_m1.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=$G2 $PY training_methods/common/eval_agent.py \
  --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden100 --tag base --budget 4 \
  > $M/runs/${TS}_eval_golden_base.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=$G3 $PY training_methods/common/eval_agent.py --adapter "$ADAPTER" \
  --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden100 --tag m1 --budget 4 \
  > $M/runs/${TS}_eval_golden_m1.log 2>&1 &
P4=$!
RC=0
wait $P1 || RC=1; wait $P2 || RC=1; wait $P3 || RC=1; wait $P4 || RC=1
[ $RC -eq 0 ] || log "WARNING: at least one eval exited non-zero (see eval logs)"

log "[4/4] results report"
$PY training_methods/common/compare_evals.py --title "m1_sft FULL results ($TS)" \
  --out $M/runs/results_full_$TS.md \
  dev_base="$(ls -dt $M/runs/eval_dev/*_base | head -1)" \
  dev_m1="$(ls -dt $M/runs/eval_dev/*_m1 | head -1)" \
  golden_base="$(ls -dt $M/runs/eval_golden100/*_base | head -1)" \
  golden_m1="$(ls -dt $M/runs/eval_golden100/*_m1 | head -1)"

log "=== m1_sft FULL PIPELINE COMPLETE — report: $M/runs/results_full_$TS.md ==="
