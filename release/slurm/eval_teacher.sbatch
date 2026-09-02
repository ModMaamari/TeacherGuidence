#!/usr/bin/env bash
#SBATCH --job-name=tgd_eval_teacher
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/eval_teacher_%j.log
# Teacher-alone arm: an API model is the agent. No GPU. Can also run on a login node
# or laptop: bash slurm/eval_teacher.sbatch <arm-name> <agent-id> "<test-set> ..."
#   sbatch slurm/eval_teacher.sbatch teacher "oai-teacher/deepseek-v4-flash" "heldout_hotpotqa heldout_musique"
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
NAME=${1:?arm name}; AGENT=${2:?agent model id}; TESTS=${3:?test sets}; shift 3
for t in $TESTS; do
  ds=${t#*_}
  [ -f "runs/eval/$NAME/$t/.done" ] && { echo "skip $NAME/$t (done)"; continue; }
  $PY_BASE scripts/eval.py --arm teacher --agent-model "$AGENT" --concurrency "${CONCURRENCY:-4}" \
    --questions "data/splits/test/${t}_questions.jsonl" --corpus "data/questions/$ds/${ds}_corpus.jsonl.gz" \
    --out "runs/eval/$NAME/$t" "$@" || echo "!! $NAME/$t incomplete (rc=$?) - rerun to retry"
done
echo "== done $(date -u +%FT%TZ)"
