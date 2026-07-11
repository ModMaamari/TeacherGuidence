# m2_rft — Step-by-step guide

Prerequisite: the m1 SFT adapter (round-1 rollout policy). All commands from the repo
root, `.venv_train/bin/python`.

## 1. Everything at once (one round)

```bash
SMOKE=1 GPU=2 bash training_methods/m2_rft/run_pipeline.sh    # ~25 min validation
GPU=2 bash training_methods/m2_rft/run_pipeline.sh            # full round (~14 h on 1 GPU)
# round 2, starting from round 1's adapter:
ROUND=2 ADAPTER=$(ls -dt training_methods/m2_rft/runs/*rft_round1*/adapter | head -1) \
  GPU=2 bash training_methods/m2_rft/run_pipeline.sh
```

## 2. Step by step

### 2.1 Fresh questions (~1 min, no GPU; idempotent per pool)

```bash
.venv_train/bin/python training_methods/m2_rft/prepare_fresh_questions.py --limit 2000
# training_methods/m2_rft/data/fresh/{fresh_questions,fresh_corpus}.jsonl
```

### 2.2 Rollouts — shard across the 4 A100s (~2.5 h instead of ~9 h)

```bash
A=$(ls -dt training_methods/m1_sft/runs/*/adapter | head -1)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup .venv_train/bin/python \
    training_methods/m2_rft/generate_rollouts.py --adapter "$A" \
    --tag r1 --shard $i/4 > /tmp/m2_rollout_$i.log 2>&1 &
done; wait
```

Each shard streams to its own timestamped dir; check progress any time via
`tail training_methods/m2_rft/runs/rollouts/*_r1_s*/rollouts.log`.

### 2.3 Filter + merge

```bash
.venv_train/bin/python training_methods/m2_rft/filter_rollouts.py \
  --rollouts training_methods/m2_rft/runs/rollouts/*_r1*/rollouts.jsonl \
  --out training_methods/m2_rft/data/round1 \
  --merge-with training_methods/m1_sft/data/train.jsonl
# check filter_stats.json: accepted %, unique questions kept, merged_total
```

### 2.4 Retrain (reuses the m1 trainer, ~2.5 h)

```bash
CUDA_VISIBLE_DEVICES=2 .venv_train/bin/python training_methods/m1_sft/train.py \
  --train-file training_methods/m2_rft/data/round1/merged_train.jsonl \
  --out-base training_methods/m2_rft/runs --tag rft_round1 --epochs 1
```

### 2.5 Evaluate + report (identical to m1; golden-100 included)

See `run_pipeline.sh` steps [5/6]–[6/6], or the m1 GUIDE — only the adapter path
changes.

## When to stop iterating

Track per round: dev EM/F1, golden-100 hits, acceptance rate in filter_stats.json.
Stop when dev gains < 1 point or acceptance stops rising (the policy has converged
on what it can self-teach).
