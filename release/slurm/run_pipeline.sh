#!/usr/bin/env bash
# Submit the whole four-arm evaluation as a dependency chain (group B / uniform split):
#   train uniform student -> [base + trained student evals] -> guided eval -> judge + results
#   teacher-alone eval runs in parallel on a CPU job.
#
#   TEACHER=oai-teacher/<model> JUDGE=oai-judge/<model> bash slurm/run_pipeline.sh [sbatch opts]
#   e.g. TEACHER=... JUDGE=... bash slurm/run_pipeline.sh -p gpu -A myaccount
#
# Every step is resumable: re-running this script resubmits only what is unfinished.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TEACHER:?set TEACHER=<teacher model id>}"; : "${JUDGE:?set JUDGE=<judge model id>}"
STUDENT_MODEL=${STUDENT_MODEL:-ibm-granite/granite-4.1-3b}
TESTS=${TESTS:-"heldout_hotpotqa heldout_2wikimultihopqa heldout_musique heldout_strategyqa"}
OPTS=("$@")
mkdir -p runs/slurm

train=$(sbatch --parsable "${OPTS[@]}" slurm/train.sbatch data/splits/uniform runs/train/uniform --model "$STUDENT_MODEL")
echo "train        job $train"
students=$(sbatch --parsable "${OPTS[@]}" --dependency=afterok:$train slurm/eval_student.sbatch \
  "base=student trained_uniform=uniform:runs/train/uniform/adapter" "$TESTS" --model "$STUDENT_MODEL")
echo "student arms job $students (after $train)"
guided=$(sbatch --parsable "${OPTS[@]}" --dependency=afterok:$students slurm/eval_guided.sbatch \
  guided_base student "$TEACHER" "$TESTS" --model "$STUDENT_MODEL")
echo "guided arm   job $guided (after $students)"
teacher=$(sbatch --parsable "${OPTS[@]}" slurm/eval_teacher.sbatch teacher "$TEACHER" "$TESTS")
echo "teacher arm  job $teacher (CPU, parallel)"
judge=$(sbatch --parsable "${OPTS[@]}" --dependency=afterany:$guided:$teacher slurm/judge.sbatch "$JUDGE")
echo "judge+results job $judge (after $guided,$teacher) -> runs/results/RESULTS.md"
echo "monitor: squeue -u \$USER ; tail -f runs/slurm/*.log ; cat runs/eval/*/*/status.json"
