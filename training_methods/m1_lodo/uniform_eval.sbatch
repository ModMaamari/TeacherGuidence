#!/usr/bin/env bash
#SBATCH -J m1_uniform
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH --mem=192G
#SBATCH -t 23:59:00
#SBATCH -o training_methods/m1_lodo/uniform_%j.log
# Four-arm comparison on ONE uniform test set: the 747 held-in questions (10% of each
# dataset), which no model of any kind has trained on.
#
#   1 base                    granite-4.1-3b, no teacher            [already measured]
#   2 teacher-guided student  base + live teacher at inference      [this run]
#   3 teacher alone           DeepSeek-V4-Flash-0731 as the agent   [already measured]
#   4 trained student (all4)  trained on all four datasets          [train + eval here]
#
# Arms 1 and 3 already have episodes on exactly these 747 questions, so they are reused
# rather than re-run -- same questions, same protocol, same metrics.
set -euo pipefail
cd /shared/al-maamari/DeKIS/teacher-guidence
export HF_HOME=$PWD/.hf_cache HF_HUB_CACHE=$PWD/.hf_cache/hub
export XDG_CACHE_HOME=$PWD/.cache VLLM_CACHE_ROOT=$PWD/.cache/vllm
export TRITON_CACHE_DIR=$PWD/.cache/triton TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
# /home is read-only on compute nodes: vLLM's usage-stats thread and flashinfer still
# resolve '~' (this killed the 4047988 eval phase) -> point HOME at /shared too.
export HOME=$PWD/.cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p .cache/vllm .cache/triton .cache/inductor .hf_cache/hub
OUT=training_methods/m1_lodo/runs/uniform
mkdir -p "$OUT/logs"
# pre-flight: fail before burning GPU time if either eval agent cannot even import
.venv_train/bin/python -c "import training_methods.common.eval_agent, training_methods.common.teacher_eval_agent" \
  2>/dev/null || { echo "PRE-FLIGHT FAILED: eval agents do not import in .venv_train"; exit 1; }
echo "node $(hostname) | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- 1. train the all-4 student (skipped if the adapter already exists) -------------
if [ ! -f "$OUT/train/adapter/adapter_config.json" ]; then
  echo "=== training fold_all4 $(date -u +%FT%TZ) ==="
  .venv_train/bin/python training_methods/m1_sft/train.py \
      --model ibm-granite/granite-4.1-3b \
      --train-file training_methods/m1_lodo/data/fold_all4/train.jsonl \
      --dev-file  training_methods/m1_lodo/data/fold_all4/dev.jsonl \
      --run-dir "$OUT/train" \
      --epochs 2 --lr 1e-4 --batch-size 4 --grad-accum 4 --lora-r 32 --lora-alpha 64 --seed 13
else
  echo "=== all4 adapter already present, skipping training ==="
fi

# ---- 2. serve base + the all4 adapter ----------------------------------------------
echo "=== starting vLLM $(date -u +%FT%TZ) ==="
.venv_vllm/bin/vllm serve ibm-granite/granite-4.1-3b \
    --served-model-name student --port 8500 --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --enable-lora --max-lora-rank 32 --max-loras 1 \
    --lora-modules "all4=$OUT/train/adapter" > "$OUT/logs/vllm.log" 2>&1 &
VLLM=$!
ready=0
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:8500/v1/models >/dev/null && { ready=1; break; }
  kill -0 $VLLM 2>/dev/null || break            # server died -> stop waiting
  sleep 10
done
if [ "$ready" != 1 ]; then
  echo "vLLM FAILED to come up after $((i*10))s -- see $OUT/logs/vllm.log"; tail -30 "$OUT/logs/vllm.log"; exit 1
fi
echo "vllm ready after $((i*10))s"
trap 'kill $VLLM 2>/dev/null || true' EXIT

Q=training_methods/m1_lodo/data/uniform
# ---- 3. arm 4: trained all-4 student, teacherless ----------------------------------
for ds in hotpotqa 2wikimultihopqa musique strategyqa; do
  tag="all4__heldin_${ds}"
  [ -f "$OUT/eval/$tag.done" ] && { echo "skip $tag"; continue; }
  .venv_train/bin/python training_methods/common/eval_agent.py \
      --model ibm-granite/granite-4.1-3b \
      --questions "training_methods/m1_lodo/data/tests/heldin_${ds}_questions.jsonl" \
      --corpus "data/datasets/tg_v1/${ds}/${ds}_train_corpus.jsonl" \
      --out "$OUT/eval" --tag "$tag" --budget 3 --hidden-budget --temperature 0 --seed 13 \
      --backend vllm --server-url http://127.0.0.1:8500 --served-model all4 \
      --batch-size 16 > "$OUT/logs/$tag.log" 2>&1 && touch "$OUT/eval/$tag.done"
  echo "  done $tag"
done

# ---- 4. arm 2: teacher-guided BASE student (live teacher at inference) --------------
for ds in hotpotqa 2wikimultihopqa musique strategyqa; do
  tag="guided__heldin_${ds}"
  [ -f "$OUT/eval/$tag.done" ] && { echo "skip $tag"; continue; }
  .venv_train/bin/python training_methods/common/teacher_eval_agent.py \
      --model ibm-granite/granite-4.1-3b \
      --questions "training_methods/m1_lodo/data/tests/heldin_${ds}_questions.jsonl" \
      --corpus "data/datasets/tg_v1/${ds}/${ds}_train_corpus.jsonl" \
      --out "$OUT/eval" --tag "$tag" --budget 3 --hidden-budget --student-temperature 0 \
      --backend vllm --server-url http://127.0.0.1:8500 --served-model student \
      --concurrency 6 --teacher-router "edenchat/flexai/DeepSeek-V4-Flash-0731" > "$OUT/logs/$tag.log" 2>&1 && touch "$OUT/eval/$tag.done"
  echo "  done $tag"
done
echo "=== uniform eval complete $(date -u +%FT%TZ) ==="
