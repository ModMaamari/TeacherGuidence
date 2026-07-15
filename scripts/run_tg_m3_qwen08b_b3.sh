#!/usr/bin/env bash
# Teacher-Guidance run: MiniMax-M3 teacher + Qwen3.5-0.8B student (vLLM-served).
#
# Identical to scripts/run_tg_m3_granite_b3.sh in every respect EXCEPT the student model,
# so the two runs are directly comparable (same teacher, router, budgets, prompts, dataset,
# concurrency):
#
#   teacher : edenai/lilac/minimaxai/minimax-m3  (primary, EdenAI Responses API, billed)
#             -> fau/MiniMaxAI/MiniMax-M3-MXFP8  (fallback, free NHR@FAU gateway)
#             OpenRouter is deliberately NOT in the chain: its key limit is exhausted.
#   student : Qwen/Qwen3.5-0.8B served by vLLM (one server per GPU, served under its real
#             HF id so every trace records the actual student model)
#   budgets : 3 planning rounds + 3 running steps, budget HIDDEN from the student
#   scope   : 3000 questions (hotpot_teacher_guidance_train3000)
#
# The teacher treats the budget as a ceiling, not a quota: a sound plan that solves the
# question in fewer steps is accepted rather than padded out.
#
# RESUMABLE: re-running this exact script continues from wherever it stopped -- the
# episodes on disk decide what is still unanswered, so nothing is redone. Monitor anytime:
#
#   .venv/bin/python scripts/tg_run_status.py \
#     --out-root data/simulation_output/tg_m3_qwen08b_b3_eden \
#     --questions data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl \
#     --num-samples 3000
set -uo pipefail
cd /root/DeKIS/teacher-guidence

echo "=== TG MiniMax-M3 (EdenAI->FAU) / Qwen3.5-0.8B b3 | $(date -u +%FT%TZ) ==="

.venv/bin/python scripts/run_tg_vllm.py \
  --num-samples 3000 \
  --num-gpus 4 \
  --workers-per-gpu 6 \
  --student-model Qwen/Qwen3.5-0.8B \
  --budget 3 \
  --planning-steps 3 \
  --hidden-budget \
  --mem-util 0.90 \
  --max-num-seqs 256 \
  --teacher-model "edenai/lilac/minimaxai/minimax-m3" \
  --teacher-router "edenai/lilac/minimaxai/minimax-m3,fau/MiniMaxAI/MiniMax-M3-MXFP8" \
  --out-root data/simulation_output/tg_m3_qwen08b_b3_eden \
  --run-name run \
  --tag tgm3qwen

echo "TG_M3_QWEN08B_B3_EDEN_DONE_RC=$?"
