#!/usr/bin/env bash
# 100-sample teacher-guidance smoke test with the NEW teacher = MiniMax-M3.
#
# Teacher router (fall through on failure): FAU MiniMax-M3 (primary) ->
# EdenAI lilac/minimaxai/minimax-m3 (fallback1) -> OpenRouter minimax/minimax-m3
# (fallback2). Student = local Ollama Granite. Guidance level 3 with plan review:
# 3 planning steps + 5 running-step budget. The actually-used teacher is recorded on
# every stored episode (teacher_model / teacher_router / teacher_models_used).
#
# Runs on the 4 highest-free-memory GPUs (--num-gpus 4). Disconnect-safe: launched under
# tmux by the caller.
set -uo pipefail
cd /root/DeKIS/teacher-guidence

OUT_ROOT=data/simulation_output/minimax_m3_smoke100
TEACHER=fau/MiniMaxAI/MiniMax-M3-MXFP8
ROUTER="fau/MiniMaxAI/MiniMax-M3-MXFP8,edenai/lilac/minimaxai/minimax-m3,custom/minimax/minimax-m3"

echo "=== MiniMax-M3 teacher-guidance smoke | $(date -u +%FT%TZ) ==="
echo "teacher=$TEACHER"
echo "router=$ROUTER"

.venv/bin/python scripts/run_batched_parallel.py \
  --num-samples 100 \
  --num-gpus 4 \
  --workers-per-gpu 4 \
  --student ollama/granite4.1:3b \
  --budget 5 \
  --planning-steps 3 \
  --teacher-model "$TEACHER" \
  --teacher-router "$ROUTER" \
  --out-root "$OUT_ROOT" \
  --run-name run \
  --tag mmx3smoke

echo "MINIMAX_M3_SMOKE_DONE_RC=$?"
