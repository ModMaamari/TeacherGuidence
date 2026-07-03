#!/usr/bin/env bash
# One-liner status check for the experiment-matrix run started by
# scripts/run_experiment_matrix.py.
#
# Usage:
#   ./scripts/check_experiment_matrix.sh
#   ./scripts/check_experiment_matrix.sh --out-dir reports/experiment_matrix_2026-07-03 --tmux tg_matrix
#
# Shows: how many of the 20 configs are done (with per-config wall time / exit
# code), which config is currently running and its last few log lines, whether
# the tmux session / driving processes are still alive, and current GPU usage.

set -euo pipefail

OUT_DIR="reports/experiment_matrix_2026-07-03"
TMUX_SESSION="tg_matrix"
TOTAL_CONFIGS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --tmux) TMUX_SESSION="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

MANIFEST="$OUT_DIR/manifest.jsonl"

echo "== Experiment matrix status ($(date -u '+%Y-%m-%d %H:%M:%S UTC')) =="
echo

if [[ -f "$MANIFEST" ]]; then
  DONE=$(wc -l < "$MANIFEST" | tr -d ' ')
else
  DONE=0
fi
echo "Configs finished: $DONE/$TOTAL_CONFIGS"
echo

if [[ -f "$MANIFEST" ]]; then
  echo "-- Finished configs (model / setting / wall time / exit code) --"
  python3 -c "
import json
for line in open('$MANIFEST'):
    row = json.loads(line)
    mins = row['wall_seconds'] / 60
    status = 'ok' if row['exit_code'] == 0 else f\"FAILED (exit {row['exit_code']})\"
    print(f\"  {row['model_slug']:10s} {row['setting']:2s}  {mins:6.1f} min  {status}\")
"
  echo
fi

echo "-- tmux session '$TMUX_SESSION' --"
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "  alive"
  echo "  last lines from the driver:"
  tmux capture-pane -t "${TMUX_SESSION}:run" -p 2>/dev/null | tail -6 | sed 's/^/    /'
else
  echo "  NOT RUNNING (session ended or was never started)"
fi
echo

echo "-- driving processes --"
pgrep -fa "agentsim.cli simulate" | sed 's/^/  /' || echo "  (no simulate process running)"
pgrep -fa "ollama serve" | sed 's/^/  /' || echo "  (no ollama server running)"
echo

echo "-- currently-running template log (last 5 lines) --"
CURRENT_LOG=$(ls -t "$OUT_DIR"/logs/exp_matrix_*.log 2>/dev/null | head -1 || true)
if [[ -n "$CURRENT_LOG" ]]; then
  echo "  $CURRENT_LOG"
  tail -5 "$CURRENT_LOG" | sed 's/^/    /'
else
  echo "  (no template logs found yet)"
fi
echo

echo "-- GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | sed 's/^/  /'
