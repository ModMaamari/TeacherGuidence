#!/usr/bin/env bash
# exp_cross_student PART 2: train Qwen3.5-0.8B TWICE -- once on the dataset generated
# with Qwen itself as the student ("self"), once on the dataset generated with
# Granite-4.1-3B as the student ("cross") -- then merge each LoRA adapter into a full
# composite checkpoint for vLLM serving (vLLM cannot LoRA-serve Qwen3.5's composite
# architecture; same G-5 workaround as exp_teacher_only).
#
# Hyperparameters match the guidance-m1 recipe exactly (LoRA r=32 alpha=64, lr 1e-4,
# cosine, 2 epochs, bf16, completion-only loss) with effective batch 16
# (micro-batch 2 x grad-accum 2 x 4 GPUs), so the ONLY difference between the two
# trained models is which student generated the training data.
#
# Usage: bash training_methods/exp_cross_student/run_train_merge.sh "1,5,6,7"
set -euo pipefail

GPUS="${1:-}"
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
PY="$REPO/.venv_train/bin"
DATA="training_methods/exp_cross_student/data"
RUNS="training_methods/exp_cross_student/runs"
mkdir -p "$RUNS"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# top 4 free GPUs (>40GB free) unless pinned explicitly
if [ -z "$GPUS" ]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | "$PY/python" -c '
import sys
rows=[(int(i),int(f)) for i,f in (l.split(",") for l in sys.stdin if l.strip())]
rows=[r for r in rows if r[1] > 40960]
rows.sort(key=lambda x:-x[1])
print(",".join(str(i) for i,_ in rows[:4]))')"
fi
NG="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== exp_cross_student train+merge | ts=$TS gpus=$GPUS nproc=$NG ==="
wc -l "$DATA"/qwen/train.jsonl "$DATA"/granite/train.jsonl

# 15s nvidia-smi sampler for the whole training window (efficiency reporting)
GPU_CSV="$RUNS/${TS}_gpu_samples.csv"
echo "ts_utc,gpu,mem_used_mib,util_pct" > "$GPU_CSV"
(
  while true; do
    ts="$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -v t="$ts" -F', *' '{print t","$1","$2","$3}' >> "$GPU_CSV"
    sleep 15
  done
) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

train_merge_one () {
  local dataset_src="$1" tag="$2" port="$3"
  local rundir="$RUNS/${TS}_${tag}"
  echo ""
  echo "=== TRAIN $tag (Qwen/Qwen3.5-0.8B on $dataset_src data) -> $rundir ==="
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY/torchrun" --nproc_per_node="$NG" --master_port="$port" \
    training_methods/m1_sft/train.py \
    --model "Qwen/Qwen3.5-0.8B" \
    --train-file "$DATA/$dataset_src/train.jsonl" --dev-file "$DATA/$dataset_src/dev.jsonl" \
    --epochs 2 --lr 1e-4 --batch-size 2 --grad-accum 2 \
    --max-length 8192 --lora-r 32 --lora-alpha 64 --seed 13 \
    --run-dir "$rundir"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY/python" \
    training_methods/common/plot_training_curves.py --run-dir "$rundir" || true
  echo "=== MERGE $tag -> $rundir/merged ==="
  "$PY/python" training_methods/common/merge_qwen_composite.py \
    --base "Qwen/Qwen3.5-0.8B" --adapter "$rundir/adapter" --out "$rundir/merged"
}

train_merge_one qwen    "self_qwen_data"     29811
train_merge_one granite "cross_granite_data" 29812

{
  echo "TS=$TS"
  echo "SELF_RUN=$RUNS/${TS}_self_qwen_data"
  echo "CROSS_RUN=$RUNS/${TS}_cross_granite_data"
} > "$RUNS/${TS}_manifest.env"
echo ""
echo "=== TRAIN+MERGE DONE. manifest -> $RUNS/${TS}_manifest.env ==="
cat "$RUNS/${TS}_manifest.env"
echo "CROSS_STUDENT_TRAIN_MERGE_DONE_RC=0"
