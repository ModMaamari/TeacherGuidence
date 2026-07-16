#!/usr/bin/env bash
# Autonomous continuation: wait for run_train_merge.sh to finish cleanly, then run the
# full eval suite. Lives in its own tmux session so the whole experiment completes
# end-to-end without an SSH connection or interactive driver.
set -uo pipefail
REPO=/root/DeKIS/teacher-guidence
cd "$REPO"
LOG=training_methods/exp_cross_student/runs/train_console.log

echo "[after-train] waiting for train+merge to finish ..."
until grep -q "CROSS_STUDENT_TRAIN_MERGE_DONE_RC=0" "$LOG" 2>/dev/null; do
  if grep -qE "Error|Traceback" <(tail -5 "$LOG" 2>/dev/null) && ! tmux has-session -t xstudent 2>/dev/null; then
    echo "[after-train] training session died without the DONE marker -- aborting"
    exit 1
  fi
  sleep 60
done
echo "[after-train] train+merge complete; starting evals"
sleep 20   # let training free its GPU memory before vLLM profiles free memory

bash training_methods/exp_cross_student/run_evals.sh 2>&1 | \
  tee -a training_methods/exp_cross_student/runs/evals_console.log
echo "AFTER_TRAIN_PIPELINE_DONE_RC=0"
