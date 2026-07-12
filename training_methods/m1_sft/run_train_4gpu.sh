#!/usr/bin/env bash
# m1_sft DDP training only (dataset must already be built): torchrun on the top
# 4 free GPUs, then training-curve PNGs. Evals are launched separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
TORCHRUN=.venv_train/bin/torchrun
M=training_methods/m1_sft
TS=${TS:-$(date -u '+%Y%m%dT%H%M%SZ')}
log() { echo "[$(date -u '+%F %T')Z] $*"; }

GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr -d ' ' | paste -sd,)
log "=== m1_sft TRAIN (4-GPU DDP) | GPUs=$GPUS | ts=$TS ==="

RUN_DIR=$M/runs/${TS}_train4gpu
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN --nproc_per_node=4 --master_port=29713 \
  $M/train.py --epochs 2 --batch-size 2 --grad-accum 2 --run-dir "$RUN_DIR" \
  2>&1 | tee "$M/runs/${TS}_train4gpu_console.log"
[ -d "$RUN_DIR/adapter" ] || { log "FATAL: adapter not found at $RUN_DIR/adapter"; exit 1; }

log "training-curve PNGs"
$PY training_methods/common/plot_training_curves.py --run-dir "$RUN_DIR" || log "WARNING: plotting failed"

log "=== m1_sft TRAIN COMPLETE — adapter: $RUN_DIR/adapter ==="
