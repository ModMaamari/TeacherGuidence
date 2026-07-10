#!/usr/bin/env bash
# Big trace collection: granite4.1:3b student, 3000 HotpotQA train-split questions,
# b9 plan-review workflow (3 planning steps, 9 running budget, hidden), no wiki,
# teacher on the full FAU -> OpenRouter-free -> OpenRouter-paid fallback router with
# hard wall-clock timeouts. Top 4 free GPUs, 8 parallel workers each (32 concurrent).
#
# Live monitoring files (inside $OUT):
#   used_sample_ids.txt   all 3000 qids, written before launch
#   results_live.jsonl    one line per finished episode, appended in real time
#   stats_live.json       rolling aggregates (done/expected, rates, ETA)
#   missing_ids.txt       expected qids with no episode yet
#
# Usage:  tmux new-session -d -s trace_g3b 'bash scripts/run_trace_collection_g3b.sh'

set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT=data/simulation_output/traces_g3b_3000
DATASET=data/datasets/hotpot_teacher_guidance_train3000
QUESTIONS=$DATASET/hotpot_distractor_train_questions.jsonl
CORPUS=$DATASET/hotpot_distractor_train_corpus.jsonl
N=3000
mkdir -p "$OUT"

# Hard wall-clock caps so a hung provider call always raises instead of blocking a
# worker forever; a hard timeout also trips that provider's circuit breaker for the
# rest of the worker process, so the router skips it instead of re-eating the timeout.
export FAU_TIMEOUT=45
export CUSTOM_TIMEOUT=180

# Cap the per-slot context. granite4.1:3b's default context is so large that 8 parallel
# slots need >80GB of KV cache -- the allocation fails and Ollama silently degrades to
# CPU/hybrid inference (~30 tok/s prefill, every student call times out). 16k tokens is
# ample for the longest step prompt; 8 slots x 16k fits comfortably on one A100.
export OLLAMA_CONTEXT_LENGTH=16384

# Top 4 free GPUs at launch (lowest memory.used), fixed for the whole run.
GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr -d ' ' | paste -sd,)

echo "[$(date -u '+%F %T')] trace collection start | GPUs=$GPUS | N=$N | out=$OUT"

# Manifest of every sample id used (first N question lines, the runner's order).
# Question rows carry the HotpotQA id as "id"; episode exports call the same value "qid".
head -n "$N" "$QUESTIONS" | python -c '
import json, sys
for line in sys.stdin:
    if line.strip():
        print(json.loads(line)["id"])
' > "$OUT/used_sample_ids.txt"
echo "[$(date -u '+%F %T')] wrote $(wc -l < "$OUT/used_sample_ids.txt") sample ids to $OUT/used_sample_ids.txt"

# Live results watcher (its own process; survives the runner, exits via RUN_DONE marker).
rm -f "$OUT/RUN_DONE"
python scripts/watch_trace_collection.py --out-root "$OUT" \
  --expected-ids "$OUT/used_sample_ids.txt" --interval 30 \
  > "$OUT/watcher.log" 2>&1 &
WATCHER=$!
echo "[$(date -u '+%F %T')] watcher pid=$WATCHER (stats_live.json refreshes every 30s)"

# Main run. No --teacher-router: uses the default FAU -> free -> paid fallback chain.
python scripts/run_batched_parallel.py \
  --num-samples "$N" --workers-per-gpu 8 --gpu-ids "$GPUS" --base-port 11800 \
  --student ollama/granite4.1:3b \
  --budget 9 --planning-steps 3 --max-plan-steps 9 \
  --workflow hotpot_teacher_guided_b9_plan_review --hidden-budget \
  --questions "$QUESTIONS" --corpus "$CORPUS" \
  --out-root "$OUT" --tag tr3k \
  > "$OUT/runner.log" 2>&1
RC=$?
echo "[$(date -u '+%F %T')] runner finished exit=$RC (runner.log tail below)"
tail -6 "$OUT/runner.log"

# Let the watcher take a final sweep over the consolidated run/ dir, then stop it.
touch "$OUT/RUN_DONE"
for _ in $(seq 1 12); do kill -0 "$WATCHER" 2>/dev/null || break; sleep 10; done
kill "$WATCHER" 2>/dev/null

echo "[$(date -u '+%F %T')] final stats:"
cat "$OUT/stats_live.json"
echo "[$(date -u '+%F %T')] missing ids: $(grep -c . "$OUT/missing_ids.txt" 2>/dev/null || echo '?')"
echo "[$(date -u '+%F %T')] TRACE COLLECTION COMPLETE"
