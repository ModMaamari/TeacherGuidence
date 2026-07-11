# m1_sft — Step-by-step guide

All commands run from the repo root. Python: `.venv_train/bin/python` (torch/TRL stack;
created separately from the inference `.venv`). Every script logs with UTC timestamps
and writes artifacts into timestamped run directories.

## 0. One-time prerequisites (already done on this machine)

```bash
/root/.pyenv/versions/3.11.9/bin/python -m venv .venv_train
.venv_train/bin/pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124
.venv_train/bin/pip install "transformers>=4.57" "trl>=0.14" "peft>=0.14" "accelerate>=1.2" \
    "datasets>=3.0" sentencepiece protobuf scipy scikit-learn matplotlib \
    loguru pydantic python-dotenv pyyaml httpx tenacity json-repair aiohttp
# checkpoint (~7 GB): downloaded to the HF cache on first use
```

## 1. Everything at once

```bash
SMOKE=1 GPU=1 bash training_methods/m1_sft/run_pipeline.sh   # ~15 min validation
GPU=1 bash training_methods/m1_sft/run_pipeline.sh           # full (~7 h, 1 GPU)
# long runs are best inside tmux:
tmux new-session -d -s m1 'GPU=1 bash training_methods/m1_sft/run_pipeline.sh 2>&1 | tee training_methods/m1_sft/runs/pipeline_console.log'
```

## 2. Or step by step

### 2.1 Build the dataset (~1 min, no GPU)

```bash
.venv_train/bin/python training_methods/m1_sft/build_dataset.py
# artifacts: training_methods/m1_sft/data/{train,dev}.jsonl, dev_questions.jsonl, stats.json, build.log
```

Check `stats.json`: expect ≈18k examples, `restored_placeholders` ≈4.4k, and **verify
there are no `[answer hidden]` tokens left** (the builder gates them).

### 2.2 Train (~4.5 h on one A100)

```bash
CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/m1_sft/train.py --epochs 2
# smoke first if anything changed: ... train.py --smoke   (~2 min)
# artifacts: training_methods/m1_sft/runs/<ts>_train/{adapter/,train.log,trainer_state.json,final_metrics.json}
```

Watch `train.log`; eval loss is logged every 200 steps.

### 2.3 Evaluate (base vs adapter; dev + golden-100)

```bash
A=$(ls -dt training_methods/m1_sft/runs/*/adapter | head -1)
Q=training_methods/m1_sft/data/dev_questions.jsonl
C=data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl
G=training_methods/common/data/golden100

CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/common/eval_agent.py \
  --questions $Q --corpus $C --out training_methods/m1_sft/runs/eval_dev --tag base --budget 4
CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/common/eval_agent.py \
  --adapter "$A" --questions $Q --corpus $C --out training_methods/m1_sft/runs/eval_dev --tag m1 --budget 4

# GOLDEN TEST — 100 never-answered questions
CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/common/eval_agent.py \
  --questions $G/golden100_questions.jsonl --corpus $G/golden100_corpus.jsonl \
  --out training_methods/m1_sft/runs/eval_golden100 --tag base --budget 4
CUDA_VISIBLE_DEVICES=1 .venv_train/bin/python training_methods/common/eval_agent.py \
  --adapter "$A" --questions $G/golden100_questions.jsonl --corpus $G/golden100_corpus.jsonl \
  --out training_methods/m1_sft/runs/eval_golden100 --tag m1 --budget 4
```

Evals can be parallelized across the 4 GPUs with `--shard i/4` + different
`CUDA_VISIBLE_DEVICES` (merge by concatenating episodes.jsonl).

### 2.4 Results table

```bash
.venv_train/bin/python training_methods/common/compare_evals.py --title "m1_sft results" \
  --out training_methods/m1_sft/runs/results.md \
  dev_base="$(ls -dt training_methods/m1_sft/runs/eval_dev/*_base | head -1)" \
  dev_m1="$(ls -dt training_methods/m1_sft/runs/eval_dev/*_m1 | head -1)" \
  golden_base="$(ls -dt training_methods/m1_sft/runs/eval_golden100/*_base | head -1)" \
  golden_m1="$(ls -dt training_methods/m1_sft/runs/eval_golden100/*_m1 | head -1)"
```

## What success looks like

- dev: trained ≫ base on EM/F1/cover; `invalid_action_steps` near 0; natural `finish`
  stop-reasons appearing (base rarely finishes voluntarily).
- golden-100: base ≈ 0 correct. Any trained-model hits are the headline number.
