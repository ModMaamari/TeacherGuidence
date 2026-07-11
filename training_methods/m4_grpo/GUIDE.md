# m4_grpo — Step-by-step guide

Prerequisite: the m1 SFT adapter (strongly recommended as the starting policy).
All commands from the repo root, `.venv_train/bin/python`. One A100 is enough.

## 1. Everything at once

```bash
SMOKE=1 GPU=3 bash training_methods/m4_grpo/run_pipeline.sh    # ~20 min validation
GPU=3 UPDATES=100 bash training_methods/m4_grpo/run_pipeline.sh  # full (~20 h) — use tmux:
tmux new-session -d -s m4 'GPU=3 UPDATES=100 bash training_methods/m4_grpo/run_pipeline.sh 2>&1 | tee training_methods/m4_grpo/runs/pipeline_console.log'
```

## 2. Step by step

### 2.1 Question pool (~5 s)

```bash
.venv_train/bin/python training_methods/m4_grpo/build_dataset.py
# training_methods/m4_grpo/data/grpo_questions.jsonl (2,913 questions, dev excluded)
```

### 2.2 Train

```bash
A=$(ls -dt training_methods/m1_sft/runs/*/adapter | head -1)
# smoke first (~6 min): 2 updates, Q=2, G=2, budget 2
CUDA_VISIBLE_DEVICES=3 .venv_train/bin/python training_methods/m4_grpo/train_grpo.py \
  --init-adapter "$A" --smoke
# full run:
CUDA_VISIBLE_DEVICES=3 .venv_train/bin/python training_methods/m4_grpo/train_grpo.py \
  --init-adapter "$A" --updates 100 --questions-per-update 4 --group-size 4 --budget 4
```

Monitor (updates.jsonl is appended after every update):

```bash
tail -f training_methods/m4_grpo/runs/<ts>_grpo/train.log
# mean_reward should trend up; grad_norm should stay O(1); many "zero variance"
# skips => raise --temperature (e.g. 1.0) or --group-size (6).
```

Adapter checkpoints are written every `--save-every` (20) updates — safe to evaluate
mid-run.

### 2.3 Evaluate (identical protocol to m1/m3)

```bash
# see run_pipeline.sh [3/4]; the pipeline merges the init adapter automatically so the
# eval composition matches training (base + m1 merged + grpo adapter).
```

### 2.4 Golden test

The pipeline runs it; manually it is the standard eval_agent.py call with
`training_methods/common/data/golden100/golden100_{questions,corpus}.jsonl`.

## Tuning notes

- `--reward-fn` accepts a dotted path — m5 reuses this trainer with a PRM reward.
- Filter the pool to hard questions for more group variance:
  `jq -c 'select(.level=="hard")' data/.../questions.jsonl > hard_pool.jsonl` then `--questions hard_pool.jsonl`.
- 4 GPUs: run 4 independent seeds (`--seed 1..4`, one per GPU) and pick the best dev
  checkpoint — more robust than one long run at this scale.
