#!/usr/bin/env bash
# Build the GPU environments m1-LODO needs. Run ON A GPU NODE (srun/sbatch): the CUDA
# wheels probe the driver at install time and vLLM cannot import without a visible device.
#
#   sbatch -p gpu --gres=gpu:h200:1 -c 16 --mem=64G -t 03:00:00 \
#     --wrap="bash training_methods/m1_lodo/setup_envs.sh"
#
# .venv_train  training stack (torch cu124, transformers, trl, peft)
# .venv_vllm   inference server, separate so a vLLM upgrade cannot break training.
#              vLLM's Python support lags; if it will not install, eval falls back to the
#              HF backend (slower but identical semantics) and this script still succeeds.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=${HF_HOME:-/shared/al-maamari/DeKIS/teacher-guidence/.hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
PY=${PY:-python3}
echo "== node $(hostname) | $($PY -V) =="

if [ ! -x .venv_train/bin/python ]; then
  $PY -m venv .venv_train
  .venv_train/bin/pip -q install -U pip wheel
  .venv_train/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cu124 \
    || .venv_train/bin/pip -q install torch
  # Pinned to the stack training_methods/ was written against: TRL's SFTConfig surface
  # changes between minors (1.12 dropped warmup_ratio), and an unpinned install silently
  # breaks the trainer. See training_methods/README.md "Environment".
  .venv_train/bin/pip -q install "transformers==5.13.*" "trl==1.8.*" "peft==0.19.*" \
      "datasets==5.0.*" accelerate \
      pyyaml loguru python-dotenv httpx pydantic json-repair rank-bm25 sentencepiece
fi
.venv_train/bin/python - <<'PYEOF'
import torch, transformers, trl, peft
print(f"train ok | torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"| gpus {torch.cuda.device_count()} | transformers {transformers.__version__} "
      f"| trl {trl.__version__} | peft {peft.__version__}")
PYEOF

if [ ! -x .venv_vllm/bin/python ]; then
  $PY -m venv .venv_vllm
  .venv_vllm/bin/pip -q install -U pip wheel
  .venv_vllm/bin/pip -q install vllm || echo "VLLM_INSTALL_FAILED"
fi
if .venv_vllm/bin/python -c "import vllm" 2>/dev/null; then
  .venv_vllm/bin/python -c "import vllm; print('vllm ok', vllm.__version__)"
else
  echo "vllm UNAVAILABLE on $($PY -V) -- eval will use the HF backend"
fi
