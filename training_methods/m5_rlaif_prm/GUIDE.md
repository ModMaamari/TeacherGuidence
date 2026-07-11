# m5_rlaif_prm — Step-by-step guide

Prerequisites: the m1 SFT adapter (RL starting policy). Uses TWO GPUs during the RL
stage (policy + PRM). All commands from the repo root, `.venv_train/bin/python`.

## 1. Everything at once

```bash
SMOKE=1 GPU=2 PRM_GPU=3 bash training_methods/m5_rlaif_prm/run_pipeline.sh  # ~35 min
GPU=2 PRM_GPU=3 UPDATES=60 bash training_methods/m5_rlaif_prm/run_pipeline.sh  # full (~30 h) — tmux recommended
```

## 2. Step by step

### 2.1 PRM dataset (~2 min, no GPU)

```bash
.venv_train/bin/python training_methods/m5_rlaif_prm/build_dataset.py
# training_methods/m5_rlaif_prm/data/{prm_train,prm_dev}.jsonl + stats.json
```

### 2.2 Train the PRM (~7 h on one A100)

```bash
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/m5_rlaif_prm/train_prm.py
# smoke: ... train_prm.py --smoke   (~3 min)
```

### 2.3 Correlation gate — DO NOT SKIP

```bash
P=$(ls -dt training_methods/m5_rlaif_prm/runs/*prm*/adapter | head -1)
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/m5_rlaif_prm/eval_prm.py \
  --adapter "$P" --limit 500
```

Go/no-go: proceed only if Spearman ≥ ~0.6 and AUC ≥ ~0.85 on the full 500. A weak
PRM will actively mislead the RL stage.

### 2.4 RLAIF (GRPO + PRM reward; 2 GPUs, ~12-15 h at 60 updates)

```bash
A=$(ls -dt training_methods/m1_sft/runs/*/adapter | head -1)
PRM_ADAPTER="$P" PRM_DEVICE=cuda:1 CUDA_VISIBLE_DEVICES=2,3 \
  .venv_train/bin/python training_methods/m4_grpo/train_grpo.py \
  --init-adapter "$A" --updates 60 \
  --reward-fn training_methods.m5_rlaif_prm.prm_reward.episode_reward \
  --out-base training_methods/m5_rlaif_prm/runs --tag rlaif
```

Monitoring: `updates.jsonl` mean_reward WILL inflate as the policy adapts to the PRM
— that is not evidence of improvement. Evaluate the adapter checkpoint on dev
(gold metrics) every ~20 updates instead:

```bash
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/common/eval_agent.py \
  --model <merged base+m1> --adapter training_methods/m5_rlaif_prm/runs/<ts>_rlaif/adapter \
  --questions training_methods/m1_sft/data/dev_questions.jsonl \
  --corpus data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl \
  --out training_methods/m5_rlaif_prm/runs/eval_dev --tag rlaif_mid --budget 4 --limit 30
```

### 2.5 Golden test + results

Same as every method (see run_pipeline.sh steps [5/6]–[6/6]).

## Bonus uses of the trained PRM (no extra training)

- **Rerank**: sample G answers with any policy, keep the PRM-highest — an
  inference-time boost that needs no RL at all.
- **Self-stop**: at each step, score a hypothetical finish action; finish when the
  PRM score clears a threshold (tune on dev).
