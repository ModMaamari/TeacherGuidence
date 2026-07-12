#!/usr/bin/env bash
# Shard one eval across several GPUs and merge the results into a single run dir.
#
#   GPUS=1,2,3,5 bash training_methods/common/run_eval_sharded.sh \
#       <eval|teacher> <out-base> <tag> [extra eval args...]
#
# - <eval|teacher> picks eval_agent.py (add --batch-size via extra args for the big
#   speedup) or teacher_eval_agent.py (add --concurrency via extra args).
# - GPUS: comma-separated GPU ids; one shard per GPU (default: top 4 free).
# - Result: <out-base>/merged_<tag>/{episodes.jsonl,metrics.json}
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_train/bin/python
KIND=${1:?usage: run_eval_sharded.sh <eval|teacher> <out-base> <tag> [args...]}
OUT=${2:?out-base required}
TAG=${3:?tag required}
shift 3
case "$KIND" in
  eval)    AGENT=training_methods/common/eval_agent.py ;;
  teacher) AGENT=training_methods/common/teacher_eval_agent.py ;;
  *) echo "unknown kind: $KIND (want eval|teacher)"; exit 1 ;;
esac
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr -d ' ' | paste -sd,)}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
mkdir -p "$OUT"
log() { echo "[$(date -u '+%F %T')Z] $*"; }
log "=== sharded $KIND eval: $N shards on GPUs $GPUS | tag=$TAG ==="

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" $PY "$AGENT" "$@" \
    --out "$OUT" --tag "${TAG}_s$i" --shard "$i/$N" \
    > "$OUT/shard_${TAG}_s$i.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { RC=1; log "WARNING: shard $i failed (see $OUT/shard_${TAG}_s$i.log)"; }
done

SHARD_DIRS=()
for i in "${!GPU_ARR[@]}"; do
  d=$(ls -dt "$OUT"/*"_${TAG}_s$i" 2>/dev/null | head -1)
  [ -n "$d" ] && SHARD_DIRS+=("$d")
done
[ ${#SHARD_DIRS[@]} -gt 0 ] || { log "FATAL: no shard dirs produced"; exit 1; }
$PY training_methods/common/merge_eval_shards.py --out "$OUT/merged_$TAG" "${SHARD_DIRS[@]}"
log "=== sharded eval done: $OUT/merged_$TAG (rc=$RC) ==="
exit $RC
