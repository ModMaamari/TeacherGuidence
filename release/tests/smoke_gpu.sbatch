#!/usr/bin/env bash
#SBATCH --job-name=tgd_smoke_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm/smoke_gpu_%j.log
# GPU smoke test (~20-30 min): trains the real student for 8 optimizer steps, serves
# base + smoke adapter with vLLM, evaluates the base and the adapter on 4 questions,
# runs the guided arm with the mock teacher, judges with the mock judge, builds results.
#   sbatch [-p <gpu partition> -A <account>] tests/smoke_gpu.sbatch
#   bash tests/smoke_gpu.sbatch          # on any machine with a GPU
set -euo pipefail
mkdir -p runs/slurm
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}   # submit from the project root
. "$ROOT/slurm/common.sh"
W=runs/smoke_gpu
rm -rf "$W"; mkdir -p "$W"
Q=data/splits/test/heldout_hotpotqa_questions.jsonl
C=data/questions/hotpotqa/hotpotqa_corpus.jsonl.gz
STUDENT_MODEL=${STUDENT_MODEL:-ibm-granite/granite-4.1-3b}
for f in data/splits/uniform/train.jsonl data/splits/uniform/dev.jsonl "$Q" "$C"; do
  [ -f "$f" ] || { echo "missing $f -- run 'make data' first (it builds the SFT train/dev files)"; exit 1; }
done
step() { echo; echo "=== $(date -u +%T) $* ==="; }

step "1/5 train (smoke: 64 examples, 8 steps)"
$PY_TRAIN scripts/train_sft.py --smoke --model "$STUDENT_MODEL" \
    --train-file data/splits/uniform/train.jsonl --dev-file data/splits/uniform/dev.jsonl --out "$W/train"
[ -f "$W/train/adapter/.done" ] || { echo "training produced no adapter"; exit 1; }
step "1b/5 re-run must skip"
$PY_TRAIN scripts/train_sft.py --smoke --model "$STUDENT_MODEL" --train-file data/splits/uniform/train.jsonl --out "$W/train" | grep -q "already trained"

step "2/5 serve base + smoke adapter"
PORT=${PORT:-8310}
MODEL=$STUDENT_MODEL start_server "$PORT" --lora "smoke=$W/train/adapter"
trap stop_server EXIT

step "3/5 student arms (base, adapter) on 4 questions"
$PY_TRAIN scripts/eval.py --arm student --student vllm --server-url "http://127.0.0.1:$PORT" --served-model student \
    --questions "$Q" --corpus "$C" --out "$W/eval/base/heldout_hotpotqa" --limit 4 --batch-size 4 --model "$STUDENT_MODEL"
$PY_TRAIN scripts/eval.py --arm student --student vllm --server-url "http://127.0.0.1:$PORT" --served-model smoke \
    --questions "$Q" --corpus "$C" --out "$W/eval/trained_smoke/heldout_hotpotqa" --limit 4 --batch-size 4 --model "$STUDENT_MODEL"
step "3b/5 resume must be a no-op"
$PY_TRAIN scripts/eval.py --arm student --student vllm --server-url "http://127.0.0.1:$PORT" --served-model student \
    --questions "$Q" --corpus "$C" --out "$W/eval/base/heldout_hotpotqa" --limit 4 | grep -q "already complete"

step "4/5 guided arm (real student, mock teacher) on 2 questions"
$PY_TRAIN scripts/eval.py --arm guided --student vllm --server-url "http://127.0.0.1:$PORT" --served-model student \
    --teacher mock/teacher --questions "$Q" --corpus "$C" --out "$W/eval/guided_mock/heldout_hotpotqa" --limit 2 --concurrency 2 --model "$STUDENT_MODEL"

step "5/5 judge (mock) + results"
$PY_BASE scripts/judge.py --judge mock/judge --episodes "$W/eval/*/*/episodes.jsonl" --out "$W/judge"
$PY_BASE scripts/collect_results.py --runs "$W/eval" --judge "$W/judge/verdicts.jsonl" --out "$W/results"
for f in "$W/eval/base/heldout_hotpotqa/.done" "$W/eval/trained_smoke/heldout_hotpotqa/.done" \
         "$W/eval/guided_mock/heldout_hotpotqa/.done" "$W/results/RESULTS.md"; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done
echo; echo "SMOKE GPU: OK  (artifacts in $W)"
