#!/usr/bin/env bash
# exp_teacher_only PART D + Qwen merge: train the three students on the teacher-only m1
# dataset (same m1_sft trainer, same per-model hyperparameters as the guidance-m1 runs, so
# the only difference is the training data), then merge the two Qwen adapters into full
# composite checkpoints for vLLM serving (G-5). Granite serves its LoRA directly (no merge).
#
# Runs sequentially -- each training uses all 4 given GPUs. Writes adapters/curves under
# training_methods/exp_teacher_only/runs/<ts>_<model>/ and a manifest of the run dirs.
#
# Usage: bash training_methods/exp_teacher_only/run_train_merge.sh "1,2,3,5"
set -euo pipefail

GPUS="${1:-1,2,3,5}"
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
PY="$REPO/.venv_train/bin"
DATA="training_methods/exp_teacher_only/data"
RUNS="training_methods/exp_teacher_only/runs"
mkdir -p "$RUNS"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
NG="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
MANIFEST="$RUNS/${TS}_manifest.env"

echo "=== teacher-only train+merge | ts=$TS gpus=$GPUS nproc=$NG ==="
echo "train=$(wc -l < "$DATA/train.jsonl") dev=$(wc -l < "$DATA/dev.jsonl") examples"

train_one () {
  local model="$1" tag="$2" bs="$3" ga="$4" port="$5"
  local rundir="$RUNS/${TS}_${tag}"
  echo ""
  echo "=== TRAIN $tag ($model) bs=$bs ga=$ga -> $rundir ==="
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY/torchrun" --nproc_per_node="$NG" --master_port="$port" \
    training_methods/m1_sft/train.py \
    --model "$model" \
    --train-file "$DATA/train.jsonl" --dev-file "$DATA/dev.jsonl" \
    --epochs 2 --lr 1e-4 --batch-size "$bs" --grad-accum "$ga" \
    --max-length 8192 --lora-r 32 --lora-alpha 64 --seed 13 \
    --run-dir "$rundir"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY/python" \
    training_methods/common/plot_training_curves.py --run-dir "$rundir" || true
}

merge_one () {
  local base="$1" tag="$2"
  local rundir="$RUNS/${TS}_${tag}"
  echo ""
  echo "=== MERGE $tag ($base) -> $rundir/merged ==="
  "$PY/python" training_methods/common/merge_qwen_composite.py \
    --base "$base" --adapter "$rundir/adapter" --out "$rundir/merged"
}

# 1. granite-3B (LoRA served directly at eval)
train_one "ibm-granite/granite-4.1-3b" "granite" 2 2 29701

# 2. Qwen3.5-0.8B (+ composite merge)
train_one "Qwen/Qwen3.5-0.8B" "qwen05b" 4 1 29702
merge_one "Qwen/Qwen3.5-0.8B" "qwen05b"

# 3. Qwen3.5-2B (+ composite merge)
train_one "Qwen/Qwen3.5-2B" "qwen2b" 4 1 29703
merge_one "Qwen/Qwen3.5-2B" "qwen2b"

{
  echo "TS=$TS"
  echo "GRANITE_RUN=$RUNS/${TS}_granite"
  echo "QWEN05B_RUN=$RUNS/${TS}_qwen05b"
  echo "QWEN2B_RUN=$RUNS/${TS}_qwen2b"
} > "$MANIFEST"
echo ""
echo "=== TRAIN+MERGE DONE. manifest -> $MANIFEST ==="
cat "$MANIFEST"
echo "TRAIN_MERGE_DONE_RC=0"
