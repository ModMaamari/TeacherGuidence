#!/usr/bin/env bash
# Autonomous continuation of exp_teacher_only. Waits for run_train_merge.sh to finish, then
# runs the four-arm unseen-100 eval for all three students, per-model analysis, and the
# cross-recipe comparison against the guidance-m1 baselines -- entirely in tmux, so the whole
# experiment completes end-to-end independent of any SSH connection or interactive driver.
#
# GPU/memory are chosen adaptively at eval time (the machine is shared): the four GPUs with
# the most free memory, and a vLLM memory fraction sized to the tightest of them so the
# servers fit alongside any co-located job. Eval accuracy/token metrics are unaffected by
# co-location (only wall time is), so this yields valid results.
#
# Usage: bash training_methods/exp_teacher_only/run_rest.sh
set -uo pipefail
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
PY="$REPO/.venv_train/bin/python"
RUNS=training_methods/exp_teacher_only/runs
EXPRUNS=training_methods/exp_unseen100/runs

# guidance-m1 baselines to compare against (PLAN.md §13)
GUID_GRANITE="$EXPRUNS/20260712T114423Z_exp"
GUID_QWEN05B="$EXPRUNS/20260712T163544Z_exp_qwen05b"
GUID_QWEN2B="$EXPRUNS/20260712T203511Z_exp_qwen2b"

echo "[rest] $(date -u +%H:%M:%S) waiting for train+merge to finish ..."
until grep -q "TRAIN_MERGE_EXIT=" "$RUNS/train_console.log" 2>/dev/null; do sleep 60; done
if ! grep -q "TRAIN_MERGE_DONE_RC=0" "$RUNS/train_console.log" 2>/dev/null; then
  echo "[rest] train+merge did NOT finish cleanly -- aborting. Check $RUNS/train_console.log"
  exit 1
fi
MANIFEST="$(ls -t "$RUNS"/*_manifest.env | head -1)"
# shellcheck disable=SC1090
source "$MANIFEST"
echo "[rest] train+merge done: $MANIFEST"
sleep 20   # let training free its GPU memory before vLLM profiles free memory

# ---- adaptive GPU + vLLM memory-fraction selection ----
read -r GPUS MEM < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | "$PY" -c '
import sys
rows=[(int(i),int(f)) for i,f in (l.split(",") for l in sys.stdin if l.strip())]
rows=[r for r in rows if r[1] > 40960]           # >40GB free
rows.sort(key=lambda x:-x[1])
sel=rows[:4]
gpus=",".join(str(i) for i,_ in sel)
minfree=min((f for _,f in sel), default=0)
mem=max(0.30, min(0.85, round(0.9*minfree/81920, 2)))
print(gpus, mem)
')
echo "[rest] eval gpus=$GPUS vllm-mem-util=$MEM"

eval_one () {  # tag, run_experiment args...
  local tag="$1"; shift
  echo "[rest] === EVAL $tag ($(date -u +%H:%M:%S)) ==="
  "$PY" training_methods/exp_unseen100/run_experiment.py \
    --backend vllm --gpus "$GPUS" --vllm-mem-util "$MEM" --clients-per-server 4 \
    --exp-tag "exp_teacheronly_$tag" "$@" || echo "[rest] eval $tag returned rc=$?"
}

eval_one granite --model ibm-granite/granite-4.1-3b \
  --adapter "$GRANITE_RUN/adapter" --served-adapter-name m1
eval_one qwen05b --model Qwen/Qwen3.5-0.8B --merged-model "$QWEN05B_RUN/merged"
eval_one qwen2b  --model Qwen/Qwen3.5-2B   --merged-model "$QWEN2B_RUN/merged"

G_EXP="$(ls -dt "$EXPRUNS"/*_exp_teacheronly_granite 2>/dev/null | head -1)"
Q05_EXP="$(ls -dt "$EXPRUNS"/*_exp_teacheronly_qwen05b 2>/dev/null | head -1)"
Q2_EXP="$(ls -dt "$EXPRUNS"/*_exp_teacheronly_qwen2b 2>/dev/null | head -1)"
echo "[rest] exp dirs: granite=$G_EXP qwen05b=$Q05_EXP qwen2b=$Q2_EXP"

for e in "$G_EXP" "$Q05_EXP" "$Q2_EXP"; do
  [ -n "$e" ] || continue
  "$PY" training_methods/exp_teacher_only/check_eval.py --exp-dir "$e" || true
  "$PY" training_methods/exp_unseen100/analyze_experiment.py --exp-dir "$e" || true
done

# ---- cross-recipe comparison (teacher-only vs guidance, per student) ----
CMP_DIR=training_methods/exp_teacher_only/report_compare
mkdir -p "$CMP_DIR"
"$PY" training_methods/exp_unseen100/compare_models.py \
  --out "$CMP_DIR/report.html" \
  --title "m1 data recipe: teacher-only expert demos vs teacher guidance" \
  "granite-3B guidance"="$GUID_GRANITE"   "granite-3B teacher-only"="$G_EXP" \
  "Qwen0.8B guidance"="$GUID_QWEN05B"      "Qwen0.8B teacher-only"="$Q05_EXP" \
  "Qwen2B guidance"="$GUID_QWEN2B"         "Qwen2B teacher-only"="$Q2_EXP" || true

echo "[rest] wrote comparison -> $CMP_DIR/report.html"
{
  echo "GRANITE_EXP=$G_EXP"
  echo "QWEN05B_EXP=$Q05_EXP"
  echo "QWEN2B_EXP=$Q2_EXP"
} > "$RUNS/eval_exp_dirs.env"
echo "PIPELINE_REST_DONE_RC=0"
