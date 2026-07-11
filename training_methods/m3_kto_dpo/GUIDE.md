# m3_kto_dpo — Step-by-step guide

Prerequisite: the m1 SFT adapter (recommended; see `../m1_sft/GUIDE.md`). All commands
from the repo root, `.venv_train/bin/python`.

## 1. Everything at once

```bash
SMOKE=1 GPU=2 bash training_methods/m3_kto_dpo/run_pipeline.sh   # ~25 min validation
GPU=2 bash training_methods/m3_kto_dpo/run_pipeline.sh           # full (~12-16 h)
# pin a specific SFT adapter:
INIT_ADAPTER=training_methods/m1_sft/runs/<ts>_train/adapter GPU=2 bash training_methods/m3_kto_dpo/run_pipeline.sh
```

## 2. Step by step

### 2.1 Build datasets (~1 min, no GPU)

```bash
.venv_train/bin/python training_methods/m3_kto_dpo/build_dataset.py
# artifacts: training_methods/m3_kto_dpo/data/{kto_train,kto_dev,dpo_train,dpo_dev}.jsonl + stats.json
```

Check `stats.json`: KTO pos/neg roughly balanced (~18k/19k); ~3.6k DPO pairs.

### 2.2 KTO training (~8 h on one A100)

```bash
A=$(ls -dt training_methods/m1_sft/runs/*/adapter | head -1)
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/m3_kto_dpo/kto_train.py \
  --init-adapter "$A"
# validate first: ... kto_train.py --init-adapter "$A" --smoke   (~4 min)
```

Watch `train.log` for `rewards/chosen` − `rewards/rejected` margins growing.

### 2.3 DPO training (~45 min on one A100)

```bash
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/m3_kto_dpo/dpo_train.py \
  --init-adapter "$A"
```

### 2.4 Evaluate

The trained adapters expect the (base + m1) composition. Merge once, then evaluate:

```bash
# the pipeline does this automatically; manually:
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/common/eval_agent.py \
  --model <merged_base_plus_m1_dir> --adapter <kto_or_dpo_adapter> \
  --questions training_methods/m1_sft/data/dev_questions.jsonl \
  --corpus data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl \
  --out training_methods/m3_kto_dpo/runs/eval_dev --tag kto --budget 4

# GOLDEN TEST (100 never-answered questions):
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/common/eval_agent.py \
  --model <merged> --adapter <adapter> \
  --questions training_methods/common/data/golden100/golden100_questions.jsonl \
  --corpus training_methods/common/data/golden100/golden100_corpus.jsonl \
  --out training_methods/m3_kto_dpo/runs/eval_golden100 --tag kto --budget 4
```

### 2.5 Results

```bash
.venv_train/bin/python training_methods/common/compare_evals.py --title "m3 results" \
  --out training_methods/m3_kto_dpo/runs/results.md \
  dev_kto="$(ls -dt training_methods/m3_kto_dpo/runs/eval_dev/*_kto | head -1)" \
  golden_kto="$(ls -dt training_methods/m3_kto_dpo/runs/eval_golden100/*_kto | head -1)"
```

## What success looks like

- KTO: dev EM/F1 above the m1 adapter's numbers; fewer invalid actions and fewer
  wasted steps (the negative labels penalize failed extracts and format errors).
- DPO: better final-answer selection (finish pairs) — EM should move most.
- Golden-100: compare against m1's golden run; every additional hit matters.
