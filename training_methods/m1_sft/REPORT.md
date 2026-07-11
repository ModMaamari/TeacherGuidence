# m1_sft — Guidance-as-Internal-Thought SFT

## What it is

Supervised fine-tuning of `granite4.1:3b` (HF: `ibm-granite/granite-4.1-3b`) on its own
teacher-guided trajectories, restructured so the teacher disappears from the input and
reappears as the *student's own generated self-critique*. Each training target is:

```json
{"teacher_guidance": {"score": 0.8, "feedback": "You retrieved the correct pages. Now extract ..."},
 "thought": "...", "decision": {...}, "action": {...}, "new_facts_extracted": [...]}
```

- `teacher_guidance` is generated FIRST (the action is conditioned on the critique).
- Feedback is kept **verbatim in second person** (user decision): the model learns an
  explicit inner-teacher voice rather than a paraphrase.
- The input prompt is the stored student prompt with the `Previous teacher guidance:`
  block **removed** — training inputs exactly match teacherless inference.
- Guidance about step *t* attaches to step *t+1* (causality); step 1 receives the
  plan-review feedback; plan turns train `plan prompt → revised (teacher-approved) plan`.
- The runtime action parser ignores extra JSON keys, so trained models remain drop-in
  compatible with the existing simulation harness.

## Data

Built by `build_dataset.py` from teacher-correct episodes only:

| source | episodes | examples |
|---|---|---|
| ds1 horizontal best (best episode per question across b1–b9) | 2,735 | 10,846 |
| ds4 hard-question recoveries | 118 | 939 |
| b5h + b9h natural `teacher_accept` episodes (stop behaviour) | 1,428 | 6,720 |
| **total** | **4,281** | **18,505** (17,961 train / 544 dev) |

2,853 unique questions; dev split is by question id (3%), never by example.
Tool mix: search 8,579 / finish 4,485 / extract 895 / verify 487 / other 220; 3,839 plan turns.

### Leakage handling (critical)

- `[answer hidden]` placeholders in feedback: **4,373 restored** (echo-safe: the gold
  answer already appeared in the question, the student's own output, or a retrieved
  document), **247 sentences dropped** (true leaks — teacher volunteered an unseen answer).
- 647 examples gated out entirely (unsafe guidance/thought, or the student parroting
  the literal placeholder). **0 placeholder tokens and 0 non-echo-safe gold mentions
  remain in any training target** (verified programmatically at build time).

## Training

TRL `SFTTrainer`, LoRA r=32/α=64 on all linear layers, bf16, gradient checkpointing,
completion-only loss (prompt tokens never trained), max_length 8192 (longest prompt
≈ 4.1k tokens). Batch 2 × grad-accum 8, lr 1e-4 cosine, 2 epochs.

- **Hardware**: one A100 80GB. Measured throughput ≈ 2.2 examples/s ⇒ full run ≈ 4.5 h.
- Smoke run (8 steps): loss 14.2 → 2.1, adapter save + reload in the agent loop verified.

## Evaluation

`common/eval_agent.py` drives the trained policy through the *identical* environment
(same prompt renderer, tool executor, per-question corpus, HotpotQA metrics) with no
teacher, on:
1. **dev questions** (83 held-out train questions) — the internalization gain;
2. **golden-100** — 100 of the 147 questions never answered correctly by ANY teacher-
   guided run (b1–b9 + retry campaign). Baseline is ~0 by construction; every solved
   golden question is evidence the model generalizes beyond its teacher-guided ceiling.

Note: absolute numbers are not directly comparable to the Ollama-based b-run tables
(bf16 vs Q4_K_M quantization, greedy decoding, no teacher) — compare base-vs-trained
within this harness.

## Risks / gotchas

- **Self-praise bias**: most surviving feedback is positive; the model may learn to
  flatter itself. Mitigated by keeping low-score steps whose episodes were still
  teacher-correct (recovery steps are the most valuable examples).
- **Survivorship bias**: ds1 skews toward easy/cheap b2–b3 wins; accepts from b5h/b9h
  partially rebalance stop behaviour.
- **Score generation**: the model learns to emit `teacher_guidance.score`; treat it as
  an uncalibrated self-confidence signal (the original scores were computed by a
  teacher who saw the gold answer).

## Cost summary

| stage | GPUs | wall time |
|---|---|---|
| dataset build | 0 | ~1 min |
| train (2 epochs) | 1×A100 | ~4.5 h |
| dev + golden eval (×2 models) | 1×A100 | ~2.5 h |
