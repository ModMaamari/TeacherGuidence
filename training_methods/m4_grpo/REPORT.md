# m4_grpo — Group-Relative Policy Optimization over Agent Episodes

## What it is

Online RL with verifiable rewards (RLVR): the policy explores the real multi-turn
environment (identical prompt renderer + deterministic tool executor as the harness),
gets a gold-based reward per episode, and is updated with **group-relative advantages**
exactly as in GRPO — G rollouts per question, `A_i = (r_i − mean)/std`, no critic, no
reward model. This is the only method here that optimizes *decision-making under
budget* rather than imitating it: the model is rewarded for being right AND cheap.

Implementation notes (train_grpo.py):
- single-update-per-batch on-policy form: each rollout batch is used once, so no
  importance ratio/clipping is needed (they equal 1);
- zero-variance groups (all G rollouts equally good/bad) are skipped — standard GRPO;
- loss = −A_i × mean token-log-prob of every generated turn (plan + steps), which
  length-normalizes and avoids long-episode bias;
- LoRA r=32 on all-linear; the optimizer only touches 1.8% of parameters (62M);
- sequences beyond `--max-length` are skipped and counted (`skipped_long_steps`).

## Reward (rewards.py)

```
r = 1.0·F1 + 0.5·EM + 0.25·cover
  + 0.3·(budget−used)/(budget−1)   only if F1 ≥ 0.5   (never reward fast garbage)
  − 0.1·(invalid action steps)                         (protect the JSON format)
  − 0.2 if the episode never emitted finish            (protect the stop behaviour)
```

Anti-hacking choices: cover-match alone is gameable by verbose answers → F1+EM carry
the signal; the efficiency bonus is gated on correctness; format decay is penalized
directly.

## Data

`build_dataset.py` writes the RL question pool: the 3000 train questions minus the 87
m1-dev questions (kept clean for eval) = **2,913 questions**, seed-shuffled. RL needs
no demonstrations — trajectories are generated on-policy.

## Cost (measured, one A100 80GB)

Rollouts dominate. Smoke config validated the full cycle (rollouts → grouped
advantages → gradient step, grad_norm ≈ 2.4 → adapter save).

| config | per update | 100 updates |
|---|---|---|
| Q=4 questions × G=4 rollouts × B=4 budget | ~8–12 min | **~15–20 h** |

100 updates ≈ 1,600 episodes ≈ touching ~400 distinct questions. This is the most
expensive method per unit of data seen — run it LAST, from the strongest offline
checkpoint (m1, ideally m1+KTO).

## Risks / gotchas

- **Small-model RL fragility**: published results show GRPO often underperforms
  distillation at ~3B scale; expect modest gains over a good SFT policy. Track
  `mean_reward` in updates.jsonl — flat means the policy is not exploring useful
  variation (raise temperature or group size).
- **Reward hacking**: watch for verbose answers creeping up F1 without EM, and for
  budget-burning (efficiency bonus should push the other way).
- **Zero-variance waste**: at temperature 0.8 some groups still collapse (all
  correct/all wrong — observed in smoke); those rollouts are pure cost. Harder
  question pools (e.g. filter to level=hard) raise the variance and the signal.
- **Throughput**: sequential HF generation is the bottleneck. If this method earns a
  full run, batching rollouts or a vLLM rollout worker is the first optimization.

## Evaluation

Same protocol as every method: dev questions + golden-100 through the teacherless
agent loop, compared against base/m1 in the shared results tables.
