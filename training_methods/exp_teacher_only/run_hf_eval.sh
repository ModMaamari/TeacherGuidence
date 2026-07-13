#!/usr/bin/env bash
# One HF-backend four-arm eval (or an arm subset) for a single trained student. HF backend
# loads the base + LoRA adapter in-process per job -- robust on a shared/chaotic machine
# (no long-lived vLLM servers) and applies the adapter correctly for both granite and Qwen
# (unlike vLLM, which mis-serves granite LoRA and ignores Qwen LoRA). Passing --model via a
# script arg avoids the arg-mangling that a long concatenated `tmux send-keys` caused.
#
# Usage: bash run_hf_eval.sh <base_model> <adapter_dir> <tag> [gpus] [arms]
set -euo pipefail
MODEL="$1"; ADAPTER="$2"; TAG="$3"; GPUS="${4:-1,2,3,5}"; ARMS="${5:-base,m1}"
cd /root/DeKIS/teacher-guidence
echo "[hf-eval] model=$MODEL adapter=$ADAPTER tag=$TAG gpus=$GPUS arms=$ARMS"
.venv_train/bin/python training_methods/exp_unseen100/run_experiment.py \
  --backend hf --gpus "$GPUS" --arms "$ARMS" \
  --model "$MODEL" --adapter "$ADAPTER" \
  --exp-tag "exp_teacheronly_hf_$TAG"
echo "HF_EVAL_DONE_$TAG"
