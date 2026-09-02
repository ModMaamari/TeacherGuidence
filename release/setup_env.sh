#!/usr/bin/env bash
# Create the three virtual environments. Run on a machine with the GPU driver visible
# (a GPU node / interactive GPU session) so the CUDA wheels resolve correctly.
#
#   bash setup_env.sh            # all three
#   ONLY=base bash setup_env.sh  # just the CPU env (data, judge, teacher arm, results)
#
# Env vars: PY (python >= 3.11, default python3), TORCH_INDEX (default cu124 wheels),
#           ONLY (base | train | serve).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
ONLY=${ONLY:-}
export HF_HOME=${HF_HOME:-$PWD/.cache/hf}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PWD/.cache/pip}
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
echo "== $($PY --version) | HF_HOME=$HF_HOME =="

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

if want base && [ ! -x .venv/bin/python ]; then
  $PY -m venv .venv
  .venv/bin/pip -q install -U pip wheel
  .venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip -q install -r requirements/base.txt
fi
if want base; then .venv/bin/python -c "import agentsim, tgd; print('base env ok')"; fi

if want train && [ ! -x .venv_train/bin/python ]; then
  $PY -m venv .venv_train
  .venv_train/bin/pip -q install -U pip wheel
  .venv_train/bin/pip -q install "torch==2.6.*" --index-url "$TORCH_INDEX" || .venv_train/bin/pip -q install torch
  .venv_train/bin/pip -q install -r requirements/train.txt
fi
if want train; then .venv_train/bin/python - <<'PYEOF'
import torch, transformers, trl, peft
print(f"train env ok | torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
      f"transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")
PYEOF
fi

if want serve && [ ! -x .venv_vllm/bin/python ]; then
  $PY -m venv .venv_vllm
  .venv_vllm/bin/pip -q install -U pip wheel
  .venv_vllm/bin/pip -q install -r requirements/serve.txt || echo "vLLM install failed: use --student hf for evaluation"
fi
if want serve; then .venv_vllm/bin/python -c "import vllm; print('serve env ok | vllm', vllm.__version__)" || true; fi

echo
echo "environments ready. Next: 'make data' to build the SFT train/dev files, then 'make smoke'."
exit 0
