# Sourced by every sbatch script. Sets caches to the project directory (compute nodes
# often mount $HOME read-only or tiny), loads .env, and defines helpers.
cd "${ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}"
export PROJECT=$PWD
export HF_HOME=${HF_HOME:-$PROJECT/.cache/hf}
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PROJECT/.cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$PROJECT/.cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PROJECT/.cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$PROJECT/.cache/inductor}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
# some libraries still resolve '~' directly; point it somewhere writable too
export HOME=${TGD_HOME_OVERRIDE:-$PROJECT/.cache/home}
mkdir -p "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HOME"
[ -f .env ] && set -a && . ./.env && set +a
export PYTHONPATH=$PROJECT
PY_BASE=${PY_BASE:-$PROJECT/.venv/bin/python}
PY_TRAIN=${PY_TRAIN:-$PROJECT/.venv_train/bin/python}
PY_VLLM=${PY_VLLM:-$PROJECT/.venv_vllm/bin/python}
echo "== $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_JOB_ID:-none} project=$PROJECT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# start_server <port> [--lora name=dir ...]  -> exports SERVER_PID; exits 1 if it dies
start_server() {
  local port=$1; shift
  PORT=$port VLLM_PYTHON=$PY_VLLM LOG=$PROJECT/runs/vllm_${port}_${SLURM_JOB_ID:-local}.log \
    bash scripts/serve_vllm.sh "$@" &
  SERVER_PID=$!
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && return 0
    kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM server failed to start (see runs/vllm_${port}_*.log)"; return 1; }
    sleep 5
  done
  echo "vLLM server not ready after 20 min"; return 1
}
stop_server() { [ -n "${SERVER_PID:-}" ] && kill $SERVER_PID 2>/dev/null || true; }
