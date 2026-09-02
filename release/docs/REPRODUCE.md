# Reproducing the reference results

Everything below runs from the project root. Expected wall time on one 80 GB-class GPU:
data 2 min · training 4 h (uniform) + 4 × 3 h (lodo folds) · evaluation ≈ 1 h per arm
and split group · judging minutes. API spend for the teacher/judge depends on the
provider; with a low-priced teacher the whole four-arm comparison cost well under $1.

## 0. Environment and credentials

```bash
bash setup_env.sh                     # on a machine that sees the GPU driver
cp .env.example .env                  # fill in OAI_TEACHER_* and OAI_JUDGE_* (docs/PROVIDERS.md)
.venv/bin/python -m pytest tests -q   # 6 unit tests
bash tests/smoke_offline.sh           # ~3 min, no GPU, no API: whole pipeline with mock models
sbatch -p <gpu-partition> tests/smoke_gpu.sbatch   # ~25 min: real student, tiny run
```

## 1. Data (shipped; rebuild is optional)

```bash
.venv/bin/python scripts/build_splits.py            # data/splits (12 s)
.venv/bin/python scripts/check_leakage.py           # must print "no leakage found"
```

To regenerate the episodes themselves with your own student/teacher, see
`docs/DATA.md` §4 (≈ 8,000 teacher-guided episodes ≈ 12 GPU-hours of student serving
plus the teacher's API calls).

## 2. Group B — uniform split, four arms (the headline comparison)

One command on Slurm:

```bash
TEACHER=oai-teacher/<model> JUDGE=oai-judge/<model> bash slurm/run_pipeline.sh -p <gpu-partition> [-A <account>]
```

which submits, with dependencies: `slurm/train.sbatch` (uniform student) →
`slurm/eval_student.sbatch` (base + trained on the four held-out sets) →
`slurm/eval_guided.sbatch` (guided base) ; `slurm/eval_teacher.sbatch` (teacher alone,
CPU, in parallel) → `slurm/judge.sbatch` (judge + `runs/results/RESULTS.md`).

Step by step without Slurm:

```bash
.venv_train/bin/python scripts/train_sft.py --train-file data/splits/uniform/train.jsonl \
    --dev-file data/splits/uniform/dev.jsonl --out runs/train/uniform
scripts/serve_vllm.sh --lora uniform=runs/train/uniform/adapter &
# then the loops in docs/EVALUATION.md "Running the arms"
```

Monitor: `squeue -u $USER`, `tail -f runs/slurm/*.log`, `cat runs/train/uniform/status.json`,
`cat runs/eval/*/*/status.json`, `cat runs/judge/status.json`.

Resume: re-run the same `run_pipeline.sh` (or the same sbatch / python command); every
stage skips what is finished.

## 3. Group A — leave-one-dataset-out

```bash
for ds in hotpotqa 2wikimultihopqa musique strategyqa; do
  sbatch -p <gpu-partition> slurm/train.sbatch data/splits/lodo/fold_$ds runs/train/fold_$ds
done
# when trained: each fold on its unseen dataset + the held-out sets of its training datasets
sbatch -p <gpu-partition> slurm/eval_student.sbatch \
  "fold_hotpotqa=fold_hotpotqa:runs/train/fold_hotpotqa/adapter" "full_hotpotqa heldout_2wikimultihopqa heldout_musique heldout_strategyqa"
sbatch -p <gpu-partition> slurm/eval_student.sbatch \
  "fold_2wikimultihopqa=fold_2wikimultihopqa:runs/train/fold_2wikimultihopqa/adapter" "full_2wikimultihopqa heldout_hotpotqa heldout_musique heldout_strategyqa"
sbatch -p <gpu-partition> slurm/eval_student.sbatch \
  "fold_musique=fold_musique:runs/train/fold_musique/adapter" "full_musique heldout_hotpotqa heldout_2wikimultihopqa heldout_strategyqa"
sbatch -p <gpu-partition> slurm/eval_student.sbatch \
  "fold_strategyqa=fold_strategyqa:runs/train/fold_strategyqa/adapter" "full_strategyqa heldout_hotpotqa heldout_2wikimultihopqa heldout_musique"
# paired baselines on the full (unseen) sets
sbatch -p <gpu-partition> slurm/eval_student.sbatch "base=student" "full_hotpotqa full_2wikimultihopqa full_musique full_strategyqa"
# optional: the teacher on a sample of each full set (they are 2,000 questions each)
sbatch slurm/eval_teacher.sbatch teacher "oai-teacher/<model>" "full_musique" --limit 250
sbatch slurm/judge.sbatch "oai-judge/<model>"
```

## 4. Changing the student, teacher or judge

* Student: `--model <hf id>` on `train_sft.py`; `MODEL=<hf id> scripts/serve_vllm.sh`;
  `--model <hf id>` on `eval.py` (recorded in `metrics.json`). `run_pipeline.sh` takes
  `STUDENT_MODEL=<hf id>`.
* Teacher: any provider-prefixed id for `--teacher` / `--agent-model` / `TEACHER=`.
* Judge: any provider-prefixed id for `--judge` / `JUDGE=`. A custom rubric:
  `judge.py --prompt-file my_prompt.txt` (must contain `{question}`, `{gold}`, `{answer}`).
* New episodes with the new pair: `docs/DATA.md` §4, then train on the new splits.

## 5. Where things end up

```
runs/train/<name>/adapter/            trained adapters (+ status.json, train.log, final_metrics.json)
runs/eval/<arm>/<test-set>/           episodes.jsonl, metrics.json, status.json, eval.log, .done
runs/judge/verdicts.jsonl             one verdict per (episode file, qid)
runs/results/results.json, RESULTS.md tables + paired statistics
runs/slurm/*.log                      job logs
runs/vllm_<port>_<job>.log            server logs
```

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PermissionError: ... /home/...` on a compute node | home is read-only there: the Slurm templates already redirect caches and `HOME` to the project; do the same in your own shell (`HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `HOME`) |
| `vLLM server failed to start` | see `runs/vllm_<port>_*.log`; common: another server on the port (`PORT=`), not enough GPU memory (`GPU_MEM=0.7`), LoRA rank above `--max-lora-rank` (64) |
| `no OpenAI-compatible endpoint configured for 'oai-x/...'` | set `OAI_X_BASE_URL` (and key) in `.env` |
| judge/teacher calls fall through to the last model and fail | credentials, or the model name is wrong for that provider; test with the snippet in `docs/PROVIDERS.md` |
| `eval.py` exits with code 3 | some questions failed after retries; re-run the same command to retry only those |
| trainer errors about `SFTConfig` arguments | wrong TRL version; install `requirements/train.txt` as pinned |
| `ModuleNotFoundError: sentence_transformers` in the train env | run `setup_env.sh` again or `pip install -r requirements/base.txt` into `.venv_train` |
