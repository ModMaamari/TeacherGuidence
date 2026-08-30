#!/usr/bin/env bash
# Wiki-v2 A/B suite: 3 students x 5 paired reps (baseline vs auto-wiki with surgical
# edits), run sequentially on 4 GPUs (2 per arm), then aggregate stats and have the
# claude CLI write reports/wiki_ab_v2/REPORT.md. Designed to run unattended in tmux.
#
# Usage:  tmux new-session -d -s wiki_ab_v2 'bash scripts/run_wiki_ab_v2_suite.sh'

set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT=reports/wiki_ab_v2
LOGDIR=$OUT/logs
mkdir -p "$LOGDIR"

# Top 4 free GPUs at launch (lowest memory.used). Fixed for the whole suite.
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n | head -4 | cut -d, -f1 | tr -d ' ')
BASE_GPUS="${GPUS[0]},${GPUS[1]}"
WIKI_GPUS="${GPUS[2]},${GPUS[3]}"

COMMON="--num-samples 10 --workers-per-gpu 5 --budget 9 --planning-steps 3 --max-plan-steps 9 \
  --workflow hotpot_teacher_guided_b9_plan_review --hidden-budget \
  --teacher-model custom/openai/gpt-oss-120b --teacher-router custom/openai/gpt-oss-120b"

echo "[$(date -u '+%F %T')] suite start | baseline GPUs=$BASE_GPUS wiki GPUs=$WIKI_GPUS"

run_pair() { # $1=model $2=tag $3=round
  local model=$1 tag=$2 r=$3 p1 p2 e1 e2
  echo "[$(date -u '+%F %T')] $tag round $r launching"
  python scripts/run_batched_parallel.py $COMMON --student "$model" \
    --gpu-ids "$BASE_GPUS" --base-port 11600 \
    --out-root "data/simulation_output/wikiv2_${tag}_nowiki10_r$r" \
    --tag "v2${tag}nw$r" > "$LOGDIR/${tag}_nowiki_r$r.log" 2>&1 &
  p1=$!
  python scripts/run_batched_parallel.py $COMMON --student "$model" --wiki --wiki-mode auto \
    --gpu-ids "$WIKI_GPUS" --base-port 11700 \
    --out-root "data/simulation_output/wikiv2_${tag}_wiki10_r$r" \
    --tag "v2${tag}w$r" > "$LOGDIR/${tag}_wiki_r$r.log" 2>&1 &
  p2=$!
  wait "$p1"; e1=$?
  wait "$p2"; e2=$?
  echo "[$(date -u '+%F %T')] $tag round $r done (nowiki exit=$e1, wiki exit=$e2)"
}

# Experiment 1: qwen3.5:0.8b | 2: qwen3.5:2b | 3: granite4.1:3b
for spec in "ollama/qwen3.5:0.8b q08b" "ollama/qwen3.5:2b q2b" "ollama/granite4.1:3b g3b"; do
  set -- $spec
  model=$1; tag=$2
  echo "[$(date -u '+%F %T')] ===== experiment $tag ($model) ====="
  for r in 1 2 3 4 5; do
    run_pair "$model" "$tag" "$r"
  done
done

echo "[$(date -u '+%F %T')] all experiments done; aggregating stats"
python scripts/analyze_wiki_ab_v2.py > "$OUT/stats.txt" 2>&1
tail -5 "$OUT/stats.txt"

echo "[$(date -u '+%F %T')] asking claude to write the report"
claude -p --dangerously-skip-permissions --model sonnet "
You are in the TeacherGuidence repo. The wiki-v2 A/B experiment suite just finished:
3 students (qwen3.5:0.8b, qwen3.5:2b, granite4.1:3b), 5 paired repetitions each,
10 HotpotQA questions per run, baseline (no wiki) vs the v2 auto-wiki (surgical
ADD/EDIT/DEL/ANSWER/NEXT edit commands, wiki injected into every step prompt and the
forced-finish path, ANSWER-line fallback in derive_final_answer).

Read reports/wiki_ab_v2/stats.json (aggregates; teacher_rate = teacher-verdict correct
rate at score >= 0.40, cover = deterministic cover-match correct out of 10,
gold_in_wiki_missed = episodes whose final wiki literally contained the gold answer but
answered wrong, edit_ops = surgical-edit usage) and reports/wiki_ab_v2/stats.txt
(readable dump). For context from the v1 experiments on the same questions: v1 wiki
hurt qwen2b badly (teacher-rate 22.7% vs 44.3% baseline, 23 gold-in-wiki-missed
episodes) and was a noisy wash for qwen0.8b (+9pts, std 29%).

Write reports/wiki_ab_v2/REPORT.md: a professional, honest experiment report with (1) a
summary verdict up front, (2) per-student result tables (mean ± std, per-rep rows,
paired per-round deltas), (3) whether v2 fixed the v1 failure modes (gold-in-wiki-missed
count, unknown answers, edit-op quality incl. rewrites/ignored ops), (4) limitations
(n=10 per rep, teacher-scored denominators vary), and (5) concrete recommendations.
Keep it factual; no overclaiming.
" > "$LOGDIR/claude_report.log" 2>&1
echo "[$(date -u '+%F %T')] claude report exit=$? (see $LOGDIR/claude_report.log)"

ls -la "$OUT"
echo "[$(date -u '+%F %T')] SUITE COMPLETE"
