# Evaluation Plan v2 — toward an A\* submission

Status: **proposal, awaiting approval.** Nothing here has been run.
Written against what exists on disk as of 2026-09-01.

---

## 1. Coverage audit — your four goals against what we actually have

### 1.1 Arms

| Arm | Status | Evidence |
|---|---|---|
| **base** (untrained 3B) | ✅ **done** | 8,746 episodes, all 8 test sets |
| **trained students** (4 LODO folds) | ✅ **done** | 10,240 episodes |
| **teacher alone** | ⚠️ **partial** | 1,755 episodes — a ~250-question sample per unseen set, not the full sets |
| **teacher-guided student** | ❌ **NOT RUN** | the arm your goals name first is missing from this study |

The teacher-guided arm is the single largest gap. It is the *external* baseline the whole
distillation claim is measured against: "internalizing guidance is worth as much as having
the teacher present." Without it the paper cannot make that claim. The code exists
(`common/teacher_eval_agent.py`, already used for the HotpotQA-only four-arm study), so
this is a run, not a build.

A fifth arm follows for free and is worth having: **trained student + live teacher**,
which answers whether internalized and external guidance are complementary or redundant.

### 1.2 Accuracy

| Metric | Status |
|---|---|
| EM, F1, cover-match | ✅ done, all arms |
| LLM judge | ✅ done, 20,741 episodes, independent judge |
| **Judge reliability** | ❌ no second judge, no human validation, no agreement statistics |
| **Statistical significance** | ❌ no confidence intervals, no paired tests |
| **Run-to-run variance** | ❌ single greedy run per configuration |

A reviewer will reject a 3-point difference reported without a CI. We have per-episode
outcomes stored, so bootstrap CIs and paired tests are *free* — no new inference.

### 1.3 Efficiency

| Aspect | Status |
|---|---|
| Tokens per episode | ✅ done (prompt + completion, per run) |
| Steps, stop reasons, invalid actions | ✅ done |
| Wall time | ⚠️ captured but **not comparable** — runs shared vLLM servers at varying concurrency |
| **Latency per episode, controlled** | ❌ needs an isolated single-stream measurement |
| **Throughput (episodes/GPU-hour)** | ❌ not measured cleanly |

### 1.4 Cost

| Aspect | Status |
|---|---|
| Training GPU-hours | ⚠️ derivable from logs (4 folds, 182–219 min, 2 co-located per GPU) but never tabulated |
| Teacher API cost | ✅ measured precisely ($0.00040/episode on EdenAI flexai; $0 on FAU) |
| **Student inference cost** | ❌ no $/1k-episode figure for local serving |
| **Amortization** | ❌ the actual argument — "training costs X once, then every episode is Y cheaper than a teacher call" — is not computed |

### 1.5 Generalisation

| Aspect | Status |
|---|---|
| Leave-one-dataset-out across 4 datasets | ✅ **done** — the study's core strength |
| Held-in vs never-trained, paired | ✅ done |
| **Truly unseen benchmark** (never in any fold) | ❌ all four datasets appear in three folds each |
| **Closed-book / contamination control** | ❌ HotpotQA and 2Wiki are almost certainly in pretraining; unquantified |
| **Student-size scaling** | ❌ one student (3B) |
| **Budget sensitivity** | ❌ evaluated only at budget 3 |

---

## 2. What an A\* reviewer will additionally demand

Beyond your four goals, five things are load-bearing for this venue.

1. **An ablation that isolates the mechanism.** The paper claims the *guidance block* is
   what teaches self-guidance. The obvious rival explanation is that any SFT on correct
   trajectories would do the same. Without `SFT-no-guidance` the central claim is
   unfalsified.
2. **Non-trivial baselines.** "Beats the untrained base model" is weak. Few-shot
   prompting and an untrained self-critique loop are the honest comparisons.
3. **Judge validity.** One LLM judge is a single point of failure. Needs a second judge
   and a human-annotated sample with agreement statistics.
4. **Error analysis.** A taxonomy of what still fails, grounded in sampled episodes —
   reviewers ask "why does MuSiQue stay at 0.37?"
5. **Reproducibility surface.** Seeds, model revisions, config hashes, and per-arm cost —
   partly present, never assembled.

---

## 3. Proposed runs

Priorities: **P0** = the paper is not submittable without it · **P1** = materially
strengthens it · **P2** = nice to have.

### P0-A · Teacher-guided student (the missing arm)
Base 3B student with the **live teacher** critiquing the plan and every step at inference
— exactly the collection protocol, on the test sets.

- Questions: the **paired subset** (all 4 held-in sets + the ~250/unseen sample) = 1,755
- Engine: `common/teacher_eval_agent.py` (exists), teacher `edenchat/flexai/DeepSeek-V4-Flash-0731`
- Cost ≈ $0.70 · GPU: 1 · wall ≈ 8–11 h (teacher-API-bound)

### P0-B · Trained student + live teacher
Same as P0-A but with each fold's adapter. Answers whether external guidance still adds
anything once it has been internalized.

- Questions: same 1,755 · Cost ≈ $0.70 · 1 GPU · ≈ 8–11 h

### P0-C · Ablation: SFT without the guidance block
Rebuild the four folds' training data with the `teacher_guidance` object **stripped from
the targets** (thought + action only), retrain, evaluate on the same sets.

- 4 trainings (~3.5 h on 3 GPUs) + 4 unseen evals (~2 h) · **no API cost**
- *This is the experiment that makes or breaks the paper's mechanism claim.*

### P0-D · Statistics over existing data (no new inference)
Bootstrap 95% CIs on every reported figure; paired McNemar tests for
student-vs-base and student-vs-teacher on identical qids; Holm correction across the
comparison family.

- Cost: **zero**, minutes of CPU

### P0-E · Judge validity
- Re-judge all 20,741 episodes with a **second** judge (MiniMax-M3) → Cohen's κ
- **300-episode human-annotated sample**, stratified by arm × dataset × judge verdict →
  judge-vs-human agreement, precision/recall of the judge against human labels
- Cost ≈ $0 (FAU) + annotator time

### P1-F · Truly held-out benchmark
Bamboogle (125 q) and, if convertible, FanOutQA — datasets in **no** fold. Run base,
all 4 students, teacher.

- ~1,000 episodes · needs a converter for each (the framework's converter contract is
  documented and four already exist) · 1 GPU · ~2 h

### P1-G · Contamination control (closed-book)
Every arm answers with **retrieval disabled**. The gap between closed-book and full
retrieval separates parametric recall from genuine multi-hop retrieval — the honest way
to discuss HotpotQA/2Wiki contamination.

- 8 sets × 3 arms, ~250 q each ≈ 6,000 episodes · 1 GPU · ~3 h · no API cost

### P1-H · Budget sensitivity
Re-evaluate base, one student and the teacher at budget ∈ {3, 5, 9} on the paired subset.
Produces the accuracy-vs-cost frontier and tests whether the ranking is budget-dependent.

- ~10,500 student episodes (2 GPU-h) + ~1,500 teacher episodes (~$0.6)

### P1-I · Seed variance
3 seeds at temperature 0.2 for base, one student and the teacher on the paired subset, so
every headline number carries a run-to-run band.

- ~10,500 episodes · 2 GPU-h · teacher ≈ $1.4

### P1-J · Efficiency and cost model
A controlled, single-stream latency run (no server sharing) for each arm, plus a costed
model: training GPU-hours amortized against per-episode inference cost, teacher API cost
per episode, and accuracy-per-1k-tokens / per-dollar / per-second frontiers.

- ~600 episodes timed in isolation · 1 GPU · ~1 h

### P2-K · Student-size scaling
Repeat the LODO training at 0.8B and 2B (both already used in this project) to show the
effect is not specific to 3B.

- 8 trainings + 8 unseen evals · ~12 GPU-h

### P2-L · Prompting baselines
Few-shot (k=4) and an untrained self-critique loop, both on the base model, so the
comparison is not only against a zero-shot base.

- ~4,000 episodes · 1 GPU · ~2 h

### P2-M · Error taxonomy
Sample 400 failures stratified by dataset and arm; label with a fixed scheme
(retrieval miss · composition failure · format · premature finish · answer-form) using the
stored raw outputs.

- Cost ≈ $0 + analyst time

---

## 4. Metrics — the full set to report

| Family | Metrics |
|---|---|
| **Accuracy** | judge-correct (primary) · cover-match · EM · F1 — all with bootstrap 95% CI |
| **Grounding** | supporting-doc recall · supporting-fact recall (sentence-gold datasets) · answer-grounded rate |
| **Behaviour** | steps used · stop-reason distribution · voluntary-finish rate · invalid-action rate · tool-use mix |
| **Efficiency** | prompt/completion tokens per episode · single-stream latency · episodes per GPU-hour · accuracy per 1k tokens |
| **Cost** | training GPU-hours and $ · inference $/1k episodes (local amortized + API) · break-even episode count vs teacher-in-loop |
| **Generalisation** | never-trained vs trained-on (paired) · truly-held-out benchmark · closed-book delta |
| **Reliability** | judge-vs-judge κ · judge-vs-human agreement · seed variance |

## 5. Data splits — unchanged, restated

| Split | Definition | Used for |
|---|---|---|
| `trainable` (90% per dataset) | qid-hash pool, correct episodes only | training the folds |
| `dev` (3% of trainable) | qid-hash, disjoint | loss monitoring only |
| `heldin_<ds>` (10% per dataset) | never trained on, dataset was in training | in-distribution accuracy |
| `unseen_<ds>` (100%) | model never saw this dataset | out-of-distribution accuracy |
| **paired subset** | held-in ∪ ~250/unseen, fixed hash | any comparison involving the teacher |
| **truly held-out** (new) | Bamboogle / FanOutQA | zero-shot generalisation |

Disjointness is enforced by `sha256(qid)` and re-asserted by `audit_folds.py`.

## 6. Evaluation method — unchanged for comparability

Budget 3 (plus the {3,5,9} sweep in P1-H), plan review on, budget hidden, greedy
(temperature 0; 0.2 with 3 seeds in P1-I), vLLM backend, identical prompts, tools and
per-question retrieval for every arm. Arms are never compared across backends.

## 7. Cost and time summary

| Priority | GPU-hours | API cost | Analyst time |
|---|---:|---:|---|
| P0 (A–E) | ~10 | ~$1.40 | 300 human annotations |
| P1 (F–J) | ~10 | ~$2.00 | — |
| P2 (K–M) | ~14 | $0 | 400 labels |
| **Total** | **~34 GPU-h** | **~$3.40** | ~700 labels |

Modest — the corpus and the pipeline are already built; these are runs, not construction.

## 8. Recommended order

1. **P0-D** (statistics) — zero cost, immediately upgrades every number we already have
2. **P0-A + P0-B** — the missing arms; run together, they share the teacher API budget
3. **P0-C** — the mechanism ablation
4. **P0-E** — judge validity
5. P1-F, P1-G, P1-H, P1-I, P1-J as GPUs allow
6. P2 only if time permits

## 9. What I would cut if forced

P2-K (size scaling) and P2-L (prompting baselines) are the least load-bearing. **P0-C is
not cuttable** — without it the paper's mechanism claim rests on assertion.
