#!/usr/bin/env bash
# Teacher-as-agent arm: DeepSeek-V4-Flash-0731 (EdenAI/flexai) solves the test questions
# itself, no guidance, same protocol as every student run. API-only -- no GPU needed.
#
# Concurrency is deliberately modest: EdenAI throttles this model, and 429s absorbed by
# backoff are fine while 429s that exhaust retries become errored episodes.
set -uo pipefail
cd "$(dirname "$0")/../.."
SHARDS=${SHARDS:-4}
LOG=data/simulation_output/teacher_arm/_logs
mkdir -p "$LOG"
for set_name in heldin_hotpotqa unseen_hotpotqa heldin_2wikimultihopqa unseen_2wikimultihopqa \
                heldin_musique unseen_musique heldin_strategyqa unseen_strategyqa; do
  echo "=== $set_name $(date -u +%FT%TZ) ==="
  pids=()
  for s in $(seq 0 $((SHARDS-1))); do
    tid="tarm_${set_name}_s${s}"
    [ -f "templates/simulations/${tid}.yaml" ] || continue
    # resumable: agentsim keeps a per-template checkpoint and skips finished samples
    .venv_probe/bin/python -m agentsim.cli simulate "$tid" > "$LOG/${tid}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  n=$(find data/simulation_output/teacher_arm -name teacher_guidance_episodes.jsonl -exec cat {} + 2>/dev/null | grep -c .)
  echo "  cumulative episodes: $n"
done
echo "=== teacher arm complete $(date -u +%FT%TZ) ==="
