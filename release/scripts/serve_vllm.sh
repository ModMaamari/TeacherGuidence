#!/usr/bin/env bash
# Serve the student with vLLM (OpenAI-compatible), optionally with LoRA adapters.
#
#   scripts/serve_vllm.sh                                # base model only, port 8300
#   scripts/serve_vllm.sh --lora uniform=runs/train/uniform/adapter \
#                         --lora fold_musique=runs/train/fold_musique/adapter
#   PORT=8301 MODEL=Qwen/Qwen2.5-3B-Instruct scripts/serve_vllm.sh
#
# The base model is served under the name "student"; each adapter under the name given
# before "=". Evaluate the base with --served-model student and an adapter with
# --served-model <name>. Blocks until the server is ready, then keeps running in the
# foreground (put it in the background yourself, e.g. `scripts/serve_vllm.sh ... &`).
#
# Env: MODEL (default ibm-granite/granite-4.1-3b), PORT (8300), GPU_MEM (0.85),
#      MAX_LEN (16384), VLLM_PYTHON (.venv_vllm/bin/python), CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-ibm-granite/granite-4.1-3b}
PORT=${PORT:-8300}
GPU_MEM=${GPU_MEM:-0.85}
MAX_LEN=${MAX_LEN:-16384}
VLLM_PYTHON=${VLLM_PYTHON:-.venv_vllm/bin/python}
LOG=${LOG:-runs/vllm_${PORT}.log}
mkdir -p "$(dirname "$LOG")"
# caches: keep everything under the project unless the caller set them
export HF_HOME=${HF_HOME:-$PWD/.cache/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$PWD/.cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/.cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PWD/.cache/inductor}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

LORAS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --lora) LORAS+=("$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
ARGS=(serve "$MODEL" --served-model-name student --port "$PORT" --dtype bfloat16
      --gpu-memory-utilization "$GPU_MEM" --max-model-len "$MAX_LEN")
if [ ${#LORAS[@]} -gt 0 ]; then
  ARGS+=(--enable-lora --max-lora-rank 64 --max-loras "${#LORAS[@]}" --lora-modules "${LORAS[@]}")
fi
echo "[serve_vllm] $MODEL on :$PORT loras=${LORAS[*]:-none} log=$LOG"
"$(dirname "$VLLM_PYTHON")/vllm" "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then
    echo "[serve_vllm] ready after $((i*5))s (pid $PID)"; break
  fi
  kill -0 $PID 2>/dev/null || { echo "[serve_vllm] server died, tail of $LOG:"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "[serve_vllm] not ready after 20 min"; exit 1; }
wait $PID
