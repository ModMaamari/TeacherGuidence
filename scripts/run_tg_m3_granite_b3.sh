#!/usr/bin/env bash
# Teacher-Guidance run: MiniMax-M3 teacher + Granite-4.1-3B student (vLLM-served).
#
#   teacher : fau/MiniMaxAI/MiniMax-M3-MXFP8 -> edenai lilac/minimaxai/minimax-m3
#             -> openrouter minimax/minimax-m3   (router falls through on failure)
#   student : ibm-granite/granite-4.1-3b served by vLLM (one server per GPU)
#   budgets : 3 planning rounds + 3 running steps, budget HIDDEN from the student
#   scope   : 3000 questions (hotpot_teacher_guidance_train3000)
#
# RESUMABLE: re-running this exact script continues from wherever it stopped -- the
# episodes on disk decide what is still unanswered, so nothing is redone. Monitor anytime:
#
#   .venv/bin/python scripts/tg_run_status.py \
#     --out-root data/simulation_output/tg_m3_granite_b3 \
#     --questions data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl \
#     --num-samples 3000
#
# The GPUs are not the bottleneck (a 3B student on an 80GB A100 leaves nearly all memory
# to the KV cache); the remote reasoning teacher is. --workers-per-gpu therefore sets the
# teacher concurrency that actually drives wall time.
set -uo pipefail
cd /root/DeKIS/teacher-guidence

echo "=== TG MiniMax-M3 / Granite b3 | $(date -u +%FT%TZ) ==="

.venv/bin/python scripts/run_tg_vllm.py \
  --num-samples 3000 \
  --num-gpus 3 \
  --workers-per-gpu 12 \
  --student-model ibm-granite/granite-4.1-3b \
  --budget 3 \
  --planning-steps 3 \
  --hidden-budget \
  --mem-util 0.90 \
  --max-num-seqs 256 \
  --teacher-model "fau/MiniMaxAI/MiniMax-M3-MXFP8" \
  --teacher-router "fau/MiniMaxAI/MiniMax-M3-MXFP8,edenai/lilac/minimaxai/minimax-m3,custom/minimax/minimax-m3" \
  --out-root data/simulation_output/tg_m3_granite_b3 \
  --run-name run \
  --tag tgm3b3

echo "TG_M3_GRANITE_B3_DONE_RC=$?"
