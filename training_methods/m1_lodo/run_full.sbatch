#!/usr/bin/env bash
#SBATCH -J m1lodo
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:3
#SBATCH -c 48
#SBATCH --mem=384G
#SBATCH -t 23:59:00
#SBATCH -o training_methods/m1_lodo/full_%j.log
# Full m1-LODO run: 4 folds trained (2 co-located per GPU), then 24 eval configurations
# sharded into 40 jobs across 3 vLLM servers. Resumable: re-submitting skips finished work.
set -euo pipefail
cd /shared/al-maamari/DeKIS/teacher-guidence
export HF_HOME=$PWD/.hf_cache HF_HUB_CACHE=$PWD/.hf_cache/hub
echo "node $(hostname) | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# batch 4 x grad-accum 4 keeps the effective batch at 16 (the validated recipe) while
# halving activation memory, so two training jobs share a card safely.
.venv_train/bin/python training_methods/m1_lodo/run_lodo.py \
    --stage all --gpus 0,1,2 \
    --jobs-per-gpu 2 --batch-size 4 --grad-accum 4 \
    --clients-per-server 4 --unseen-shards 3 --eval-batch 16 \
    --out training_methods/m1_lodo/runs/lodo
