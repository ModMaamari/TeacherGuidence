# Reference results

Student `ibm-granite/granite-4.1-3b`; teacher `DeepSeek-V4-Flash` (served through an OpenAI-compatible API); judge `Kimi-K2.6` (with MiniMax-M3 / Mistral-Medium as fallbacks; 99.9 % of verdicts came from Kimi). Budget 3, hidden; greedy student decoding; teacher temperature 0.1. One seed. Numbers are what `scripts/collect_results.py` reports; confidence intervals are paired bootstrap over questions.

> The shipped training split differs from the one these runs used by 26 examples (0.2 %): two hygiene filters — dropping examples that still contain the `[answer hidden]` mask token, and dropping the one training question whose text duplicates a held-out question — were added after the runs. The test set is identical.

## Group B — uniform split: four arms on the 747 held-out questions

| Arm | n | EM | F1 | Cover | **Judge** | Doc recall | Steps | Voluntary finish | Tokens / episode | API $ / episode |
|---|---|---|---|---|---|---|---|---|---|---|
| base student | 747 | 5.9 % | 0.117 | 22.1 % | **29.8 %** | 0.745 | 2.98 | 2.4 % | 4,442 | $0.00000 |
| guided student | 747 | 26.0 % | 0.348 | 49.7 % | **60.5 %** | 0.746 | 2.95 | 4.5 % | 10,972 | $0.00022 |
| trained student | 747 | 34.3 % | 0.430 | 60.9 % | **65.5 %** | 0.787 | 2.89 | 11.0 % | 4,672 | $0.00000 |
| teacher alone | 747 | 24.9 % | 0.340 | 65.7 % | **72.3 %** | 0.844 | 2.76 | 100.0 % | 3,889 | $0.00015 |

Tokens are step-phase tokens for the guided arm (its plan-phase calls were not recorded in that run; the kit's `eval.py` now records them). The trained student's training cost was 4.0 GPU-hours once.

### Judge-correct by dataset

| Dataset | n | base | guided | trained | teacher |
|---|---|---|---|---|---|
| HotpotQA | 189 | 45.5 % | 64.0 % | 66.7 % | 79.4 % |
| 2WikiMultihopQA | 170 | 40.6 % | 62.4 % | 82.3 % | 84.7 % |
| MuSiQue | 203 | 17.2 % | 42.4 % | 42.4 % | 55.7 % |
| StrategyQA | 185 | 17.8 % | 75.1 % | 74.1 % | 71.9 % |

### Paired differences in judge-correct (b − a), pooled

| a → b | Δ pts | 95 % CI | b wins / a wins | McNemar p |
|---|---|---|---|---|
| base student → trained student | +35.6 | [+31.7, +39.4] | 288 / 22 | <1e-6 |
| base student → guided student | +30.7 | [+26.9, +34.4] | 257 / 28 | <1e-6 |
| trained student → guided student | -5.0 | [-8.3, -1.6] | 65 / 102 | 0.0052 |
| trained student → teacher alone | +6.8 | [+3.4, +10.3] | 116 / 65 | 0.00018 |
| guided student → teacher alone | +11.8 | [+8.0, +15.5] | 152 / 64 | <1e-6 |
| base student → teacher alone | +42.4 | [+38.7, +46.2] | 327 / 10 | <1e-6 |

Questions solved by at least one arm: 626 / 747; by all four: 172; by none: 121.

**Reading.** Training on teacher-guided episodes lifts the student by +35.6 points to 91 % of the teacher's accuracy with no teacher at inference, and it beats the same student helped live by the teacher at every step (+5.0 points, CI +1.6…+8.3) while using 0.43× the tokens. The teacher stays ahead on HotpotQA and MuSiQue; on StrategyQA both student arms edge past it. The students' remaining gap is mostly stopping behaviour (2–11 % voluntary finishes vs 100 % for the teacher), not retrieval (doc recall 0.79 vs 0.84).

## Group A — leave-one-dataset-out: transfer to a never-seen dataset

Each fold trains on three datasets and is evaluated on the entire fourth (`full_<ds>`, 2,000 questions) and on the held-out 10 % of its three training datasets. Judge-correct:

| Fold (unseen dataset) | base on unseen | fold on unseen | fold on held-out of its training sets (HotpotQA / 2Wiki / MuSiQue / StrategyQA) |
|---|---|---|---|
| fold_hotpotqa (HotpotQA) | 49.2 % | **70.6 %** | — / 79.4 % / 41.9 % / 71.9 % |
| fold_2wikimultihopqa (2WikiMultihopQA) | 44.4 % | **73.8 %** | 66.1 % / — / 40.4 % / 73.0 % |
| fold_musique (MuSiQue) | 13.8 % | **36.8 %** | 64.0 % / 80.6 % / — / 73.0 % |
| fold_strategyqa (StrategyQA) | 18.9 % | **66.9 %** | 63.5 % / 77.6 % / 36.0 % / — |

For comparison, the uniform student (trained on all four) scores on the held-out sets: HotpotQA 66.7 %, 2WikiMultihopQA 82.3 %, MuSiQue 42.4 %, StrategyQA 74.1 %.

**Reading.** A student that never saw a dataset reaches 71 % (HotpotQA), 74 % (2Wiki), 37 % (MuSiQue) and 67 % (StrategyQA) judge-correct on all of it — far above the base student (49 / 44 / 14 / 19 %). On identical questions, the held-out sets, the folds that never trained on a dataset score within a few points of the student trained on it: HotpotQA 63.5–66.1 % vs 66.7 %, 2Wiki 77.6–80.6 % vs 82.3 %, MuSiQue 36.0–41.9 % vs 42.4 %, StrategyQA 71.9–73.0 % vs 74.1 %. Guidance internalised on other datasets transfers; training on the target dataset itself adds 0.5–6 points.

## Reference teacher-alone numbers on the full sets (samples)

| Dataset | n (seeded sample of `full_<ds>`) | teacher judge | base judge (same questions) | fold judge (same questions) |
|---|---|---|---|---|
| HotpotQA | 254 | 83.5 % | 46.1 % | 68.9 % |
| 2WikiMultihopQA | 252 | 83.7 % | 46.0 % | 71.4 % |
| MuSiQue | 229 | 53.3 % | 10.5 % | 32.0 % |
| StrategyQA | 273 | 66.7 % | 17.9 % | 62.6 % |

## Caveats

* One training seed and one decoding seed; the CIs cover question sampling only.
* A single judge model; EM/F1/cover are reported so the judge's leniency can be inspected (it is systematically kinder than cover-match to verbose but correct answers, for every arm).
* Per-episode API cost is the provider's reported price at run time; GPU cost is not included in the tables.
