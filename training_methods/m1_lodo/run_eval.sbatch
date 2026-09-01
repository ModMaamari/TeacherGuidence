#!/usr/bin/env bash
#SBATCH -J m1lodo_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:3
#SBATCH -c 48
#SBATCH --mem=384G
#SBATCH -t 23:59:00
#SBATCH -o training_methods/m1_lodo/eval_%j.log
# Evaluation only -- the four adapters are already trained. Resumable: finished shards
# leave a .done marker and are skipped.
set -euo pipefail
cd /shared/al-maamari/DeKIS/teacher-guidence
export HF_HOME=$PWD/.hf_cache HF_HUB_CACHE=$PWD/.hf_cache/hub
echo "node $(hostname) | $(date -u +%FT%TZ)"
.venv_train/bin/python training_methods/m1_lodo/run_lodo.py \
    --stage eval --gpus 0,1,2 \
    --clients-per-server 4 --unseen-shards 3 --eval-batch 16 \
    --out training_methods/m1_lodo/runs/lodo
