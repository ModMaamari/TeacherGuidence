#!/usr/bin/env bash
# End-to-end pipeline smoke test with NO GPU and NO API key (mock models):
#   collect 2 episodes -> consolidate -> split + leakage check ->
#   eval all three arm types on 3 questions -> judge -> results table.
# Takes ~2-4 minutes on a laptop. Usage: bash tests/smoke_offline.sh [work-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
W=${1:-runs/smoke_offline}
rm -rf "$W"; mkdir -p "$W"
Q=data/splits/test/heldout_musique_questions.jsonl
C=data/questions/musique/musique_corpus.jsonl.gz
[ -f "$Q" ] || { echo "run scripts/build_splits.py first (data/splits/test missing)"; exit 1; }
step() { echo; echo "=== $* ==="; }

step "1/7 collect episodes (mock student + mock teacher)"
$PY scripts/collect_episodes.py --smoke --datasets hotpotqa --num-samples 2 \
    --student mock/student --teacher mock/teacher --out "$W/collect" --tag smoke
step "2/7 consolidate"
$PY scripts/consolidate_episodes.py --runs "$W/collect" --out "$W/episodes" --gzip
step "3/7 build splits from the shipped episodes (200-episode sample) + leakage audit"
$PY scripts/build_splits.py --out "$W/splits" --limit 200
$PY scripts/check_leakage.py --splits "$W/splits"
step "4/7 arm: student alone (mock policy)"
$PY scripts/eval.py --arm student --student mock --questions "$Q" --corpus "$C" --out "$W/eval/base/heldout_musique" --limit 3 --batch-size 2
step "5/7 arm: guided student (mock teacher)"
$PY scripts/eval.py --arm guided --student mock --teacher mock/teacher --questions "$Q" --corpus "$C" --out "$W/eval/guided/heldout_musique" --limit 3 --concurrency 2
step "6/7 arm: teacher alone (mock agent)"
$PY scripts/eval.py --arm teacher --agent-model mock/agent --questions "$Q" --corpus "$C" --out "$W/eval/teacher/heldout_musique" --limit 3 --concurrency 2
step "7/7 judge + results"
$PY scripts/judge.py --judge mock/judge --episodes "$W/eval/*/*/episodes.jsonl" --out "$W/judge"
$PY scripts/collect_results.py --runs "$W/eval" --judge "$W/judge/verdicts.jsonl" --out "$W/results" > /dev/null
for f in "$W/episodes/episodes.jsonl.gz" "$W/splits/leakage_report.json" "$W/eval/base/heldout_musique/.done" \
         "$W/eval/guided/heldout_musique/.done" "$W/eval/teacher/heldout_musique/.done" "$W/judge/verdicts.jsonl" "$W/results/RESULTS.md"; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done
echo; echo "SMOKE OFFLINE: OK  (artifacts in $W)"
