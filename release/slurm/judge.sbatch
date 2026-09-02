#!/usr/bin/env bash
#SBATCH --job-name=tgd_judge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/judge_%j.log
# Judge every episode file under runs/eval with an API judge, then build the results.
#   sbatch slurm/judge.sbatch "oai-judge/<model>[,fallback]"
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
JUDGE=${1:?judge model id}; shift
$PY_BASE scripts/judge.py --judge "$JUDGE" --episodes 'runs/eval/*/*/episodes.jsonl' --out runs/judge "$@" || true
$PY_BASE scripts/collect_results.py --runs runs/eval --judge runs/judge/verdicts.jsonl --out runs/results
