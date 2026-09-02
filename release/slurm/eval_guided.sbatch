#!/usr/bin/env bash
#SBATCH --job-name=tgd_eval_guided
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/eval_guided_%j.log
# Teacher-guided student arm: the student (base or adapter) acts, an API teacher reviews.
#   sbatch slurm/eval_guided.sbatch <arm-name> <served-name[:adapter-dir]> "<teacher-id[,fallback]>" "<test-set> ..." [extra eval args]
#   sbatch slurm/eval_guided.sbatch guided_base student "oai-teacher/deepseek-v4-flash" "heldout_musique heldout_hotpotqa"
# Needs the teacher's credentials in .env (docs/PROVIDERS.md). Resumable.
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
NAME=${1:?arm name}; SPEC=${2:?served-name[:adapter]}; TEACHER=${3:?teacher id}; TESTS=${4:?test sets}; shift 4
PORT=${PORT:-8300}
served=${SPEC%%:*}; LORAS=()
[[ "$SPEC" == *:* ]] && LORAS=(--lora "$served=${SPEC#*:}")
start_server "$PORT" ${LORAS[@]+"${LORAS[@]}"}
trap stop_server EXIT
for t in $TESTS; do
  ds=${t#*_}
  [ -f "runs/eval/$NAME/$t/.done" ] && { echo "skip $NAME/$t (done)"; continue; }
  $PY_TRAIN scripts/eval.py --arm guided --student vllm --server-url "http://127.0.0.1:$PORT" \
    --served-model "$served" --teacher "$TEACHER" --concurrency "${CONCURRENCY:-6}" \
    --questions "data/splits/test/${t}_questions.jsonl" --corpus "data/questions/$ds/${ds}_corpus.jsonl.gz" \
    --out "runs/eval/$NAME/$t" "$@" || echo "!! $NAME/$t incomplete (rc=$?) - resubmit to retry"
done
echo "== done $(date -u +%FT%TZ)"
