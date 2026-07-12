#!/usr/bin/env bash
# All three m1 eval arms in parallel after training:
#   arm 1 (no guidance):        dev-base, golden-base          [teacherless]
#   arm 2 (internalized):       dev-m1, golden-m1              [teacherless]
#   arm 3 (m1 + real teacher):  dev-m1-teacher, golden-m1-teacher (2 shards each,
#                               teacher = gpt-oss-120b FAU->OpenRouter router)
# GPU layout: 4 teacherless evals on the first four free GPUs; the two teacher-arm
# shard workers on the next two (dev shards first, then golden shards).
#
# Usage: bash training_methods/m1_sft/run_evals_all.sh <train-TS>
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
M=training_methods/m1_sft
TS=${1:?usage: run_evals_all.sh <train-TS>}
ADAPTER=$M/runs/${TS}_train4gpu/adapter
log() { echo "[$(date -u '+%F %T')Z] $*"; }
[ -d "$ADAPTER" ] || { log "FATAL: adapter not found at $ADAPTER"; exit 1; }

GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -6 | cut -d, -f1 | tr -d ' ' | paste -sd,)
IFS=',' read -r G0 G1 G2 G3 G4 G5 <<< "$GPUS"
log "=== m1_sft ALL EVAL ARMS | GPUs=$GPUS | adapter=$ADAPTER ==="

GOLD_Q=training_methods/common/data/golden100/golden100_questions.jsonl
GOLD_C=training_methods/common/data/golden100/golden100_corpus.jsonl
DEV_Q=$M/data/dev_questions.jsonl
TRAIN_C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl

log "[1/3] four teacherless evals (arms 1+2) on GPUs $G0,$G1,$G2,$G3"
CUDA_VISIBLE_DEVICES=$G0 $PY training_methods/common/eval_agent.py \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag base --budget 4 --batch-size "${EVAL_BATCH:-8}" \
  > $M/runs/${TS}_eval_dev_base.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=$G1 $PY training_methods/common/eval_agent.py --adapter "$ADAPTER" \
  --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev --tag m1 --budget 4 --batch-size "${EVAL_BATCH:-8}" \
  > $M/runs/${TS}_eval_dev_m1.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=$G2 $PY training_methods/common/eval_agent.py \
  --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden100 --tag base --budget 4 --batch-size "${EVAL_BATCH:-8}" \
  > $M/runs/${TS}_eval_golden_base.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=$G3 $PY training_methods/common/eval_agent.py --adapter "$ADAPTER" \
  --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden100 --tag m1 --budget 4 --batch-size "${EVAL_BATCH:-8}" \
  > $M/runs/${TS}_eval_golden_m1.log 2>&1 &
P4=$!

log "[2/3] teacher-in-loop arm (m1 + gpt-oss-120b) on GPUs $G4,$G5 (2 shards, dev then golden)"
teacher_worker() {  # $1 = gpu, $2 = shard i/n
  local gpu=$1 shard=$2 sid=${2%%/*}
  CUDA_VISIBLE_DEVICES=$gpu $PY training_methods/common/teacher_eval_agent.py --adapter "$ADAPTER" \
    --questions $DEV_Q --corpus $TRAIN_C --out $M/runs/eval_dev_teacher \
    --tag "m1_teacher_s$sid" --budget 4 --shard "$shard" --concurrency "${TEACHER_CONCURRENCY:-3}" \
    > $M/runs/${TS}_eval_dev_teacher_s$sid.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY training_methods/common/teacher_eval_agent.py --adapter "$ADAPTER" \
    --questions $GOLD_Q --corpus $GOLD_C --out $M/runs/eval_golden_teacher \
    --tag "m1_teacher_s$sid" --budget 4 --shard "$shard" --concurrency "${TEACHER_CONCURRENCY:-3}" \
    > $M/runs/${TS}_eval_golden_teacher_s$sid.log 2>&1
}
teacher_worker "$G4" 0/2 & P5=$!
teacher_worker "$G5" 1/2 & P6=$!

RC=0
wait $P1 || { RC=1; log "WARNING: dev-base eval failed"; }
wait $P2 || { RC=1; log "WARNING: dev-m1 eval failed"; }
wait $P3 || { RC=1; log "WARNING: golden-base eval failed"; }
wait $P4 || { RC=1; log "WARNING: golden-m1 eval failed"; }
wait $P5 || { RC=1; log "WARNING: teacher shard 0 failed"; }
wait $P6 || { RC=1; log "WARNING: teacher shard 1 failed"; }
[ $RC -eq 0 ] && log "all six eval jobs finished cleanly" || log "WARNING: some eval jobs failed (see logs)"

log "[3/3] results report (teacherless arms; the teacher arm is merged by the analysis step)"
$PY training_methods/common/compare_evals.py --title "m1_sft FULL results ($TS)" \
  --out $M/runs/results_full_$TS.md \
  dev_base="$(ls -dt $M/runs/eval_dev/*_base | head -1)" \
  dev_m1="$(ls -dt $M/runs/eval_dev/*_m1 | head -1)" \
  golden_base="$(ls -dt $M/runs/eval_golden100/*_base | head -1)" \
  golden_m1="$(ls -dt $M/runs/eval_golden100/*_m1 | head -1)"

log "=== m1_sft ALL EVALS COMPLETE — report: $M/runs/results_full_$TS.md ==="
