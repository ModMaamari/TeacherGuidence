#!/usr/bin/env bash
#SBATCH -J m1lodo_eval_a100
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-sxm:3
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 23:59:00
#SBATCH -o training_methods/m1_lodo/eval_a100_%j.log
# Evaluation on 3xA100-80GB. The adapters are already trained; this stage is resumable
# (a finished shard leaves a .done marker) and fault-tolerant (a server that fails to come
# up is dropped, not fatal; the stage retries inside the allocation).
#
# A100 is chosen over H200 purely for queue time -- a 3.4B model under vLLM fits either.
set -euo pipefail
cd /shared/al-maamari/DeKIS/teacher-guidence
export HF_HOME=$PWD/.hf_cache HF_HUB_CACHE=$PWD/.hf_cache/hub
echo "node $(hostname) | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Never hand the allocation back on a transient failure: retry the stage in-place.
for attempt in 1 2 3; do
  echo "=== orchestrator attempt $attempt $(date -u +%FT%TZ) ==="
  if .venv_train/bin/python training_methods/m1_lodo/run_lodo.py \
        --stage eval --gpus 0,1,2 \
        --clients-per-server 4 --unseen-shards 3 --eval-batch 16 \
        --eval-attempts 3 \
        --out training_methods/m1_lodo/runs/lodo; then
    echo "=== eval complete on attempt $attempt ==="; break
  fi
  echo "=== attempt $attempt failed; retrying in 120s ==="; sleep 120
done
# Report from whatever completed, even on a partial run.
.venv_train/bin/python training_methods/m1_lodo/build_lodo_report.py \
    --out training_methods/m1_lodo/runs/lodo --md reports/m1_lodo/REPORT.md || true
echo "done $(date -u +%FT%TZ)"
