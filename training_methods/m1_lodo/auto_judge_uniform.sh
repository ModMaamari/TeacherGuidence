#!/usr/bin/env bash
# Watch the uniform four-arm eval (uniform_eval.sbatch) and judge each eval tag
# as soon as its .done marker appears.  Idempotent + resumable: judge_runs.py
# skips (source,qid) pairs already in verdicts.jsonl, and a tag is only marked
# judged when a pass ends with 0 unresolved.  Run detached (setsid nohup) so it
# survives the interactive session; events go to $EVENTS (one line each).
#
#   usage: auto_judge_uniform.sh <slurm_jobid>
set -u
cd "$(dirname "$0")/../.."                     # repo root
JOB=${1:?slurm jobid}
OUT=training_methods/m1_lodo/runs/uniform
JUDGE=$OUT/judge
EVENTS=$OUT/logs/auto_judge.events
SLURM_LOG=training_methods/m1_lodo/uniform_${JOB}.log
mkdir -p "$JUDGE" "$OUT/logs"
TAGS="all4__heldin_hotpotqa all4__heldin_2wikimultihopqa all4__heldin_musique all4__heldin_strategyqa \
guided__heldin_hotpotqa guided__heldin_2wikimultihopqa guided__heldin_musique guided__heldin_strategyqa"

ev() { echo "$(date -u +%FT%TZ) | $*" >> "$EVENTS"; }
ev "watcher started for job $JOB (pid $$)"

err_seen=0
while true; do
  # --- surface failures in the slurm log (once per new occurrence) ------------
  n_err=$(grep -cE "Traceback|CUDA out of memory|CANCELLED|DUE TO TIME LIMIT|srun: error|Killed" "$SLURM_LOG" 2>/dev/null || echo 0)
  if [ "$n_err" -gt "$err_seen" ]; then
    ev "ERROR signature in $SLURM_LOG: $(grep -E 'Traceback|CUDA out of memory|CANCELLED|DUE TO TIME LIMIT|srun: error|Killed' "$SLURM_LOG" | tail -1 | cut -c1-160)"
    err_seen=$n_err
  fi

  # --- judge every finished, not-yet-judged tag --------------------------------
  for tag in $TAGS; do
    [ -f "$OUT/eval/$tag.done" ] || continue
    [ -f "$JUDGE/$tag.judged" ] && continue
    ep=$(ls -d "$OUT"/eval/*"$tag"/episodes.jsonl 2>/dev/null | head -1)
    [ -n "$ep" ] || { ev "WARN $tag.done present but no episodes.jsonl"; continue; }
    ev "EVAL DONE $tag ($(wc -l < "$ep") episodes) -> judging"
    for pass in 1 2 3; do
      .venv_probe/bin/python training_methods/m1_lodo/judge_runs.py \
          --out "$JUDGE" --glob "$ep" --concurrency 12 --report-every 100 \
          >> "$OUT/logs/judge_$tag.log" 2>&1
      last=$(grep -E "done: .* unresolved|nothing to do" "$OUT/logs/judge_$tag.log" | tail -1)
      if echo "$last" | grep -qE "nothing to do| 0 unresolved"; then
        touch "$JUDGE/$tag.judged"
        n=$(grep -c "\"source\": \"$ep\"" "$JUDGE/verdicts.jsonl" 2>/dev/null || echo 0)
        ev "JUDGED $tag: $n verdicts in $JUDGE/verdicts.jsonl (pass $pass)"
        break
      fi
      ev "judge pass $pass for $tag left unresolved items: ${last:-no summary line}; retrying in 60s"
      sleep 60
    done
    [ -f "$JUDGE/$tag.judged" ] || ev "FAILED to fully judge $tag after 3 passes (see $OUT/logs/judge_$tag.log)"
  done

  # --- exit conditions ------------------------------------------------------------
  all_judged=1
  for tag in $TAGS; do [ -f "$JUDGE/$tag.judged" ] || all_judged=0; done
  if [ "$all_judged" = 1 ]; then ev "ALL 8 TAGS JUDGED - watcher exiting"; exit 0; fi

  if ! squeue -h -j "$JOB" 2>/dev/null | grep -q .; then
    # job left the queue; one last sweep already happened above
    st=$(sacct -j "$JOB" -n -o State -X 2>/dev/null | head -1 | tr -d ' ')
    missing=$(for t in $TAGS; do [ -f "$OUT/eval/$t.done" ] || echo -n "$t "; done)
    ev "JOB $JOB LEFT QUEUE (state=${st:-unknown}); eval tags without .done: ${missing:-none}"
    if [ -n "$missing" ]; then ev "watcher exiting: job ended with unfinished evals - resubmit uniform_eval.sbatch (it skips .done tags)"; exit 2; fi
  fi
  sleep 60
done
