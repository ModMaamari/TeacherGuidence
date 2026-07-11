# m3_kto_dpo — Step-level KTO + Same-prompt DPO

## What it is

Offline preference optimization on top of the m1 SFT policy, using the two preference
signals our data gives for free:

1. **KTO (primary)** — every teacher-scored step is a standalone thumbs-up/down example.
   No pairs needed, which matches our data (one trajectory per question per run;
   prompts differ across runs). This is the **only offline method that learns from the
   ~19k rejected steps** the SFT dataset throws away.
2. **DPO (secondary)** — where genuinely identical prompts exist: plan pairs (correct
   vs wrong episode's plan for the same question) and finish pairs (same final state;
   right vs wrong answer as a synthetic contrast).

Completions use the m1 guidance-as-internal-thought format, so both stack cleanly on
the m1 adapter (`--init-adapter`, merged before training; the KTO/DPO reference policy
is then exactly the SFT policy).

## Data (built from all six runs, 18k episodes)

| dataset | size | construction |
|---|---|---|
| KTO | 37,396 (18,358 pos / 19,038 neg → capped 20k/label) | step score ≥0.9 → True, ≤0.2 → False, 26,336 mid-score skipped |
| DPO plan pairs | 1,953 | chosen/rejected plans of correct/wrong episodes, same question |
| DPO finish pairs | 1,686 | correct episode's final state; answer swapped with a wrong episode's (skipped when the "wrong" answer cover-matches gold) |

2,107 of 3,000 questions have both a correct and a wrong episode. Same leakage
handling as m1 (echo-safe restoration, placeholder + gold-leak gates; 465 gated out).
Dev split by question id (3%).

## Training

TRL `KTOTrainer` / `DPOTrainer`, LoRA r=32 on all-linear, bf16, gradient
checkpointing, β=0.1, lr 5e-6, 1 epoch. With `peft_config` the frozen reference
model is the adapter-disabled base — **no second model in memory**; one A100 80GB.

- Measured smoke throughput: KTO ≈ 1.3 ex/s ⇒ ~8 h full; DPO ≈ 1.4 pairs/s ⇒ ~45 min full.
- Smoke runs (6 steps each): KTO loss 0.49, DPO loss 0.60, adapters saved + reloadable.

## Evaluation

Same protocol as m1 (dev questions + golden-100 through the teacherless agent loop).
When `--init-adapter` was used, evaluation runs against a merged (base+SFT) checkpoint
so the composition matches training exactly.

## Risks / gotchas

- **Run after m1.** Preference-optimizing the raw base model wastes the signal on a
  policy that can't produce the format yet.
- **Length/format hacking**: DPO can drift toward longer/shorter outputs instead of
  content preferences; our finish pairs isolate the answer choice by keeping every
  other token identical, which mitigates this. Monitor `rewards/margins` in train.log.
- **KTO label noise**: step scores are bimodal but not perfect; mid-scores (0.2–0.9)
  are deliberately excluded rather than thresholded.
- **Plan-pair approximation**: rejected plans were written under a different budget
  line than the chosen prompt shows (budgets are capped lists; effect is mild and
  documented here for honesty).

## Cost summary

| stage | GPUs | wall time |
|---|---|---|
| dataset build | 0 | ~1 min |
| KTO (1 epoch, 37k) | 1×A100 | ~8 h |
| DPO (1 epoch, 3.6k) | 1×A100 | ~45 min |
| evals | 1×A100 | ~2.5 h |
