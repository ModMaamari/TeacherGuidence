#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for eval serving (bf16, optional LoRA).
#
#   GPU=6 PORT=8300 MODEL=ibm-granite/granite-4.1-3b \
#     ADAPTERS="m1=training_methods/m1_sft/runs/<ts>/adapter" \
#     bash training_methods/common/serve_vllm.sh
#
# Env vars:
#   GPU        (required) CUDA device index
#   PORT       default 8300
#   MODEL      default ibm-granite/granite-4.1-3b; served under name "student"
#   ADAPTERS   optional space-separated name=path LoRA modules (served by name)
#   MEM_UTIL   default 0.85 of the GPU (lower it when sharing a GPU)
#   MAX_LEN    default 16384 context length
#   LOG        default training_methods/common/vllm_<PORT>.log
#
# The server runs in the foreground (use tmux/nohup); readiness: GET /v1/models
# or vllm_backend.wait_ready().
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:?set GPU=<index>}
PORT=${PORT:-8300}
MODEL=${MODEL:-ibm-granite/granite-4.1-3b}
MEM_UTIL=${MEM_UTIL:-0.85}
MAX_LEN=${MAX_LEN:-16384}
LOG=${LOG:-training_methods/common/vllm_${PORT}.log}
ADAPTERS=${ADAPTERS:-}

ARGS=(serve "$MODEL"
  --served-model-name student
  --host 127.0.0.1 --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$MEM_UTIL")
if [ -n "$ADAPTERS" ]; then
  ARGS+=(--enable-lora --max-lora-rank 64)
  for kv in $ADAPTERS; do ARGS+=(--lora-modules "$kv"); done
fi

echo "[serve_vllm] GPU=$GPU PORT=$PORT MODEL=$MODEL ADAPTERS='$ADAPTERS' -> $LOG"
# vLLM JIT-compiles kernels at startup and needs the venv's ninja on PATH
export PATH="$(pwd)/.venv_vllm/bin:$PATH"
CUDA_VISIBLE_DEVICES=$GPU exec .venv_vllm/bin/vllm "${ARGS[@]}" 2>&1 | tee "$LOG"
