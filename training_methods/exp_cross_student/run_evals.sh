#!/usr/bin/env bash
# exp_cross_student PART 3: end-to-end evaluation of the cross-student question.
#
# 3 models x 2 test sets x 3 seeds = 18 teacherless agent runs of 200 questions each:
#   base      Qwen/Qwen3.5-0.8B untrained (control)
#   m1_self   Qwen trained on data generated with QWEN as the student
#   m1_cross  Qwen trained on data generated with GRANITE as the student
# Test sets: qwen-derived and granite-derived (100 correct + 100 wrong each), both
# unseen by BOTH trained models (their qids were banned from both training sets).
#
# Eval conditions mirror data collection: budget 3, hidden budget, plan first,
# temperature 0.2 (sampled; 3 seeds for variance), same train3000 local corpus.
# Each model is served by its own vLLM server on its own GPU (merged weights -- vLLM
# cannot LoRA-serve Qwen3.5's composite architecture); jobs are HTTP clients, 2
# concurrent per server (continuous batching absorbs them). A 15s nvidia-smi sampler
# records GPU memory/util for the whole experiment. Afterwards every final answer is
# judged post-hoc (same FAU gpt-oss judge as previous experiments).
#
# Usage: bash training_methods/exp_cross_student/run_evals.sh [manifest.env]
set -uo pipefail
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
PY="$REPO/.venv_train/bin/python"
EXP=training_methods/exp_cross_student
DATA="$EXP/data"
CORPUS=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl
SEEDS="11 23 37"

MANIFEST="${1:-$(ls -t "$EXP"/runs/*_manifest.env | head -1)}"
# shellcheck disable=SC1090
source "$MANIFEST"
echo "=== exp_cross_student evals | manifest=$MANIFEST ==="
echo "SELF_RUN=$SELF_RUN"
echo "CROSS_RUN=$CROSS_RUN"
[ -d "$SELF_RUN/merged" ] && [ -d "$CROSS_RUN/merged" ] || { echo "merged checkpoints missing"; exit 1; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$EXP/runs/${TS}_evals"
mkdir -p "$OUT"

# ---- top-3 free GPUs (>40GB), one vLLM server per model -------------------------
GPUS=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | "$PY" -c '
import sys
rows=[(int(i),int(f)) for i,f in (l.split(",") for l in sys.stdin if l.strip())]
rows=[r for r in rows if r[1] > 40960]
rows.sort(key=lambda x:-x[1])
print(" ".join(str(i) for i,_ in rows[:3]))'))
echo "server gpus: ${GPUS[*]}"

declare -A MODEL_PATH=( [base]="Qwen/Qwen3.5-0.8B" [m1_self]="$SELF_RUN/merged" [m1_cross]="$CROSS_RUN/merged" )
declare -A PORT=( [base]=8501 [m1_self]=8502 [m1_cross]=8503 )
MODELS=(base m1_self m1_cross)

# 15s GPU sampler for the whole eval window
GPU_CSV="$OUT/gpu_samples.csv"
echo "ts_utc,gpu,mem_used_mib,util_pct" > "$GPU_CSV"
( while true; do
    ts="$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -v t="$ts" -F', *' '{print t","$1","$2","$3}' >> "$GPU_CSV"
    sleep 15
  done ) &
SAMPLER_PID=$!

SERVER_PIDS=()
cleanup () {
  kill "$SAMPLER_PID" 2>/dev/null || true
  for p in "${SERVER_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  sleep 3
  # serve_vllm.sh pipes through tee, which detaches the real vLLM PID -- kill ours by venv path
  ps -eo pid,args | grep -E "\.venv_vllm/bin/vllm serve|VLLM::EngineCore" | grep -v grep \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
}
trap cleanup EXIT

for i in "${!MODELS[@]}"; do
  m="${MODELS[$i]}"
  GPU="${GPUS[$i]}" PORT="${PORT[$m]}" MODEL="${MODEL_PATH[$m]}" MEM_UTIL=0.85 \
    LOG="$OUT/vllm_${m}.log" bash training_methods/common/serve_vllm.sh \
    > /dev/null 2>&1 &
  SERVER_PIDS+=($!)
  echo "serving $m (${MODEL_PATH[$m]}) on GPU ${GPUS[$i]} :${PORT[$m]}"
done
for m in "${MODELS[@]}"; do
  "$PY" -c "
from training_methods.common.vllm_backend import wait_ready
print('  ready:', '$m', wait_ready('http://127.0.0.1:${PORT[$m]}', 'student', timeout_s=900))"
done

# ---- 6 jobs per model (2 test sets x 3 seeds), 2 concurrent per server ----------
run_model_jobs () {  # model
  local m="$1" port="${PORT[$1]}"
  local pids=()
  for testset in qwen granite; do
    for seed in $SEEDS; do
      local tag="${m}_${testset}test_s${seed}"
      "$PY" training_methods/common/eval_agent.py \
        --backend vllm --server-url "http://127.0.0.1:$port" --served-model student \
        --model "Qwen/Qwen3.5-0.8B" \
        --questions "$DATA/$testset/test_questions.jsonl" --corpus "$CORPUS" \
        --budget 3 --hidden-budget --temperature 0.2 --seed "$seed" --batch-size 16 \
        --out "$OUT/$m" --tag "$tag" > "$OUT/job_${tag}.log" 2>&1 &
      pids+=($!)
      # at most 2 concurrent jobs per server (this function runs as its own subshell,
      # so `jobs` counts only THIS model's eval jobs)
      while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
    done
  done
  for p in "${pids[@]}"; do wait "$p" || echo "[evals] job pid $p rc=$?"; done
}

T0=$(date +%s)
for m in "${MODELS[@]}"; do run_model_jobs "$m" & done
wait
echo "=== all eval jobs done in $(( ($(date +%s) - T0) / 60 )) min ==="

# ---- post-hoc judge over every run (free FAU gpt-oss router) --------------------
mapfile -t EPFILES < <(ls "$OUT"/*/*/episodes.jsonl 2>/dev/null)
echo "judging ${#EPFILES[@]} runs ..."
"$PY" training_methods/common/judge_final_answers.py \
  --out "$OUT/judge" --concurrency 10 "${EPFILES[@]}" \
  > "$OUT/judge_console.log" 2>&1 || echo "judge rc=$?"

echo "OUT=$OUT" > "$EXP/runs/latest_evals.env"
echo "CROSS_STUDENT_EVALS_DONE_RC=0"
