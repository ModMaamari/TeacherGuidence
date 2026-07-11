# m5_rlaif_prm — RLAIF with a Distilled Process Reward Model

## What it is

The classic RLAIF pipeline adapted to this project: instead of calling the (gold-
seeing) teacher at RL time, we first **distill the teacher's ~64k step judgments into
a PRM**, then run **GRPO with the PRM as a dense, gold-free reward** (reusing the m4
trainer via `--reward-fn`). This is the most faithful "internalize the teacher"
method: m1 internalizes the teacher's *voice*, m5 additionally internalizes its
*judgment* as a critic that can score unseen rollouts.

### PRM design (granite has no sequence-classification head)

Generative digit judge: LoRA-SFT the backbone to answer
`state + proposed action → one digit 0-9` (`round(teacher_score × 9)`). At scoring
time one forward pass reads the softmax over the ten digit tokens at the first
generated position and returns the expected value / 9 → a continuous score in [0, 1].
Verified: all ten digits are single tokens; scoring works end-to-end.

**The PRM sees only student-visible state** (guidance blocks stripped, no gold/hidden
fields), so it is legitimately usable at inference (reranking, self-stop) — not just
in training.

## Data

`build_dataset.py`: 64,197 teacher-scored steps across all six runs → capped 60k →
58,218 train / 1,782 dev (split by question id). Label distribution spans the full
0–9 range (0: 7.5k, 2: 11.3k, 4: 10.0k, 8: 9.1k, 9: 10.2k, ...).

## Pipeline

1. Train PRM (1 epoch ≈ 7 h on one A100; smoke-validated).
2. **Correlation gate** (`eval_prm.py`): Pearson/Spearman/AUC against held-out teacher
   scores. Even the 8-step smoke PRM reached AUC 0.77; a full train should land
   well above 0.85. **If the gate is weak, stop here** — a bad PRM only misleads RL.
3. RLAIF: m4's GRPO trainer with `--reward-fn training_methods.m5_rlaif_prm.
   prm_reward.episode_reward`; reward = mean PRM step score − invalid-action /
   no-finish penalties. Policy on one GPU, PRM on a second (2 of the 4 A100s).
4. Standard evals (dev + golden-100).

## Why this over plain GRPO (m4)?

- **Dense reward**: every step gets a score, not just the episode outcome — the main
  stabilizer for small-model RL.
- **Gold-free**: extends beyond questions with known answers (any question corpus
  works for RL, not just annotated ones).
- The PRM doubles as an inference-time tool: rerank G candidate actions, or stop
  early when the PRM scores a finish action highly.

## Risks / gotchas

- **Proxy-error compounding** (the big one): the PRM imitates a teacher who saw the
  gold answer; it is systematically optimistic exactly where the student is blind.
  Under RL pressure the policy will find PRM blind spots (reward hacking a learned
  model is easier than hacking F1). Mitigations: the correlation gate, few updates
  (60 default), monitoring dev EM/F1 (ground truth) rather than reward, and the
  invalid/no-finish penalties staying gold-free but hack-resistant.
- **Two-GPU requirement** during RL (policy + PRM), still within the 4×A100 budget.
- **Score inflation**: PRM scores drift up as the policy adapts; compare rewards only
  within an update (group-relative normalization already does this — an absolute
  reward trend is NOT evidence of improvement, unlike in m4).

## Cost summary

| stage | GPUs | wall time |
|---|---|---|
| PRM dataset build | 0 | ~2 min |
| PRM train (1 epoch, 58k) | 1×A100 | ~7 h |
| PRM correlation eval (500) | 1×A100 | ~25 min |
| RLAIF 60 updates | 2×A100 | ~12–15 h |
| evals | 1×A100 | ~2.5 h |
