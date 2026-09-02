#!/usr/bin/env bash
#SBATCH --job-name=tgd_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/train_%j.log
# Train one student. Usage (from the project root):
#   sbatch [-p <partition> -A <account>] slurm/train.sbatch <split-dir> <out-dir> [extra train_sft args]
#   sbatch slurm/train.sbatch data/splits/uniform runs/train/uniform
#   sbatch slurm/train.sbatch data/splits/lodo/fold_musique runs/train/fold_musique --model Qwen/Qwen2.5-3B-Instruct
# Resumable: resubmitting resumes from the last checkpoint; a finished run is skipped.
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
SPLIT=${1:?split dir}; OUT=${2:?out dir}; shift 2
$PY_TRAIN scripts/train_sft.py --train-file "$SPLIT/train.jsonl" --dev-file "$SPLIT/dev.jsonl" --out "$OUT" "$@"
