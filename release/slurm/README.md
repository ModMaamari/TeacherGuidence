# Slurm templates

| Script | Resource | Purpose |
|---|---|---|
| `train.sbatch <split-dir> <out-dir> [train args]` | 1 GPU | train one student (resumes from checkpoints) |
| `eval_student.sbatch "<arm=served[:adapter]> ..." "<test-set> ..."` | 1 GPU | base and/or trained students, one vLLM server |
| `eval_guided.sbatch <arm> <served[:adapter]> "<teacher>" "<test-set> ..."` | 1 GPU + API | student guided by a teacher |
| `eval_teacher.sbatch <arm> "<agent>" "<test-set> ..."` | CPU + API | the teacher as the agent |
| `judge.sbatch "<judge>"` | CPU + API | judge all episodes, build `runs/results` |
| `run_pipeline.sh` | — | submits the uniform four-arm comparison as a dependency chain |

Pass partition/account as normal sbatch options: `sbatch -p <partition> -A <account> slurm/train.sbatch ...`.
Edit the `#SBATCH` headers if your cluster needs different memory/time defaults or a
`--gres` syntax such as `--gres=gpu:a100:1`.

`common.sh` (sourced by all of them) points every cache and `HOME` into the project
directory, loads `.env`, prints the node and GPU, and provides `start_server` /
`stop_server` for the vLLM student server. Gated Hugging Face models: put
`HF_TOKEN=...` in `.env`.

All scripts are idempotent: finished test sets (`.done`) and finished training runs
are skipped, so resubmitting after a time-limit kill continues where it stopped.
Logs: `runs/slurm/<job>_<id>.log`, `runs/vllm_<port>_<id>.log`, and each run's own
`status.json` / `eval.log` / `train.log`.
