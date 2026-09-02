#!/usr/bin/env bash
#SBATCH --job-name=tgd_eval_student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/eval_student_%j.log
# Student arms (base and/or trained) on one or more test sets, one vLLM server.
#   sbatch slurm/eval_student.sbatch "<arm=served-name[:adapter-dir]> ..." "<test-set> ..." [extra eval args]
# Examples:
#   # base student on all four held-out sets
#   sbatch slurm/eval_student.sbatch "base=student" "heldout_hotpotqa heldout_2wikimultihopqa heldout_musique heldout_strategyqa"
#   # base + trained (adapter served as "uniform") in one job
#   sbatch slurm/eval_student.sbatch "base=student trained_uniform=uniform:runs/train/uniform/adapter" "heldout_musique"
#   # LODO fold on its unseen dataset
#   sbatch slurm/eval_student.sbatch "fold_musique=fold_musique:runs/train/fold_musique/adapter" "full_musique"
# Output: runs/eval/<arm>/<test-set>/ ; resubmitting resumes unfinished sets.
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
ARMS=${1:?arm specs}; TESTS=${2:?test sets}; shift 2
PORT=${PORT:-8300}
LORAS=()
for spec in $ARMS; do
  name=${spec%%=*}; rhs=${spec#*=}; served=${rhs%%:*}
  if [[ "$rhs" == *:* ]]; then LORAS+=(--lora "$served=${rhs#*:}"); fi
done
start_server "$PORT" ${LORAS[@]+"${LORAS[@]}"}
trap stop_server EXIT
for spec in $ARMS; do
  name=${spec%%=*}; rhs=${spec#*=}; served=${rhs%%:*}
  for t in $TESTS; do
    ds=${t#*_}   # heldout_<ds> / full_<ds>
    [ -f "runs/eval/$name/$t/.done" ] && { echo "skip $name/$t (done)"; continue; }
    $PY_TRAIN scripts/eval.py --arm student --student vllm --server-url "http://127.0.0.1:$PORT" \
      --served-model "$served" --questions "data/splits/test/${t}_questions.jsonl" \
      --corpus "data/questions/$ds/${ds}_corpus.jsonl.gz" --out "runs/eval/$name/$t" "$@" || echo "!! $name/$t incomplete (rc=$?) - resubmit to retry"
  done
done
echo "== done $(date -u +%FT%TZ)"
