# m2_rft — Iterative Rejection-sampling Fine-Tuning (STaR/ReST loop)

## What it is

The classic self-improvement loop, using the assets this project already has (87k
unused HotpotQA train questions, gold answers, the batch environment):

1. **Rollout**: the current policy (m1 SFT adapter for round 1) runs teacherless on
   FRESH train questions, sampling `n` episodes per question at temperature 0.8.
2. **Filter (rejection sampling)**: keep episodes that are verifiably correct
   (cover_match AND F1 ≥ 0.5 against gold); keep only the highest-reward episode per
   question (reward = m4's verifiable reward, so cheap correct beats slow correct).
3. **Retrain**: merge accepted trajectories with the m1 dataset and re-run the m1
   trainer (1 epoch on the mixture).
4. Evaluate, then iterate with `ROUND=2` starting from the new adapter.

Round-≥1 targets have no `teacher_guidance` block (there is no teacher in rollouts);
the m1 mixture keeps that behaviour alive while fresh rounds reinforce *successful
search strategies on unseen questions* — exactly the distribution the policy will
face. This is the cheapest method that learns beyond the fixed 3000-question set.

## Why it fits this project

- The `hard_q_xr` campaign was already one manual rejection-sampling round — this
  automates the loop with the student instead of the teacher-guided ensemble.
- Verifiable filtering needs no teacher calls (gold is available for the train split);
  optionally the FAU teacher could re-judge accepted episodes for extra precision.
- Self-training amplification risk is bounded by the gold-based filter and the fixed
  dev/golden evals.

## Data

- `prepare_fresh_questions.py`: 90,447 train questions loaded from the HF cache,
  minus the 3000 already used → shuffled pool (default 2,000/round, seed 29).
  Verified: exclusion works, corpus rows regenerate in the standard schema.
- Rollout volume (defaults): 2,000 questions × 2 samples × budget 4 ≈ 4,000 episodes.
- Expected acceptance: ~40–60% of questions with at least one correct sample (the m1
  policy's teacherless cover rate bounds this; even the raw base model scored 1/2 on
  the smoke sample).

## Cost (one A100 per round; rollouts shardable across all 4)

| stage | wall time |
|---|---|
| rollouts 2,000 q × 2 samples (1 GPU) | ~9 h — or **~2.5 h sharded on 4 GPUs** |
| filter + merge | minutes |
| retrain 1 epoch on merged (~20k examples) | ~2.5 h |
| evals | ~2.5 h |

## Risks / gotchas

- **Quirk amplification**: the model reinforces its own habits (favorite phrasings,
  tool ruts). The gold filter blocks *wrong* habits, not *inefficient* ones; the
  per-question best-reward dedup pushes toward efficiency.
- **Diminishing rounds**: expect round 1 > round 2 > round 3; stop when dev EM/F1
  plateaus (the golden-100 usually moves last).
- **Filter strictness**: `--min-f1 0.5` + cover is deliberately strict; loosening it
  buys volume at the cost of noisy targets. Never accept on cover alone (verbose
  answers cover-match too easily).
- **No new guidance blocks**: if the internalized `teacher_guidance` voice fades
  after several rounds, re-mix a larger share of the m1 dataset (the merge keeps it
  every round by default).

## Evaluation

Standard protocol: dev questions + golden-100 through the teacherless loop, compared
against the m1 adapter (round 0) in the shared results tables.
