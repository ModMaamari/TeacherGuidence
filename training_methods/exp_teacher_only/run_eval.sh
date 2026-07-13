#!/usr/bin/env bash
# exp_teacher_only PART E: four-arm unseen-100 eval for each teacher-only student, reusing
# the exp_unseen100 stack unchanged (only the adapter/merged weights change). Granite serves
# its LoRA directly; the two Qwen students serve their merged composite checkpoints on the
# twin (+100) ports (G-5). Each run auto-judges all 2000 answers and writes analysis inputs.
#
# Runs the three models sequentially on the four given GPUs. Records each experiment dir to
# an eval manifest for the analyze/compare stage.
#
# Usage: bash training_methods/exp_teacher_only/run_eval.sh "1,2,3,5" <train_manifest.env>
set -euo pipefail

GPUS="${1:?gpus e.g. 1,2,3,5}"
MANIFEST="${2:?path to the *_manifest.env written by run_train_merge.sh}"
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
PY="$REPO/.venv_train/bin/python"
EXPRUNS="training_methods/exp_unseen100/runs"
# shellcheck disable=SC1090
source "$MANIFEST"   # provides TS, GRANITE_RUN, QWEN05B_RUN, QWEN2B_RUN
EVAL_MANIFEST="$(dirname "$MANIFEST")/${TS}_eval_manifest.env"
: > "$EVAL_MANIFEST"

run_eval () {
  local tag="$1"; shift
  echo ""
  echo "=== EVAL $tag on gpus $GPUS ==="
  "$PY" training_methods/exp_unseen100/run_experiment.py \
    --backend vllm --gpus "$GPUS" --clients-per-server 4 \
    --exp-tag "exp_teacheronly_$tag" "$@"
  local d
  d="$(ls -dt "$EXPRUNS"/*_exp_teacheronly_"$tag" | head -1)"
  echo "$(echo "$tag" | tr '[:lower:]' '[:upper:]')_EXP=$d" >> "$EVAL_MANIFEST"
  echo "eval $tag -> $d"
}

# granite-3B: vLLM serves the base + LoRA module 'm1'
run_eval granite --model ibm-granite/granite-4.1-3b \
  --adapter "$GRANITE_RUN/adapter" --served-adapter-name m1

# Qwen students: vLLM serves base + merged composite twin
run_eval qwen05b --model Qwen/Qwen3.5-0.8B --merged-model "$QWEN05B_RUN/merged"
run_eval qwen2b  --model Qwen/Qwen3.5-2B   --merged-model "$QWEN2B_RUN/merged"

echo ""
echo "=== EVAL DONE. manifest -> $EVAL_MANIFEST ==="
cat "$EVAL_MANIFEST"
echo "EVAL_ALL_DONE_RC=0"
