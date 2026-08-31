#!/usr/bin/env bash
# Live status of the m1-LODO run. Safe to run any time, from anywhere.
#   bash training_methods/m1_lodo/watch.sh          # one snapshot
#   watch -n30 bash training_methods/m1_lodo/watch.sh
cd "$(dirname "$0")/../.."
OUT=${OUT:-training_methods/m1_lodo/runs/lodo}
JOB=$(cat training_methods/m1_lodo/runs/.jobid 2>/dev/null || echo "")
echo "=== m1-LODO @ $(date -u +%FT%TZ) ==="
[ -n "$JOB" ] && squeue -j "$JOB" -h -o "slurm %i %T %M elapsed on %R" 2>/dev/null
echo
echo "-- training --"
for d in "$OUT"/train/fold_*; do
  [ -d "$d" ] || continue
  f=$(basename "$d")
  if [ -f "$d/adapter/adapter_config.json" ]; then
    echo "  $f  DONE"
  else
    last=$(grep -oE "'loss': '[0-9.e+-]+'.*'epoch': '[0-9.]+'" "$OUT/logs/train_${f#fold_}.log" 2>/dev/null | tail -1)
    echo "  $f  running  ${last:-starting}"
  fi
done
echo
echo "-- evaluation --"
done_n=$(ls "$OUT"/eval/*.done 2>/dev/null | wc -l)
echo "  completed job shards: $done_n / 40"
echo
echo "-- recent --"
tail -6 "$OUT/run.log" 2>/dev/null | sed 's/^/  /'
echo
echo "-- gpu --"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (login node: no GPUs; check inside the job)"
