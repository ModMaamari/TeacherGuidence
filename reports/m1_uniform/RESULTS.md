# Four arms on one test set (747 held-in questions)

Teacher: `edenchat/flexai/DeepSeek-V4-Flash-0731` · judge: Kimi-K2.6 · budget 3 (+hidden) · GPU $2.00/h A100

## Accuracy (pooled)

| Arm | n | EM | F1 | Cover | Judge | Judge macro | Doc recall |
|---|---|---|---|---|---|---|---|
| Base | 747 | 5.9% | 0.117 | 22.1% | **29.8%** | 30.3% | 0.745 |
| Guided | 747 | 26.0% | 0.348 | 49.7% | **60.5%** | 61.0% | 0.746 |
| Trained (all-4) | 747 | 34.3% | 0.430 | 60.9% | **65.5%** | 66.4% | 0.787 |
| Teacher | 747 | 24.9% | 0.340 | 65.7% | **72.3%** | 72.9% | 0.844 |

## Judge-correct by dataset

| Dataset | Base | Guided | Trained (all-4) | Teacher |
|---|---|---|---|---|
| HotpotQA | 45.5% | 64.0% | 66.7% | 79.4% |
| 2WikiMultihopQA | 40.6% | 62.4% | 82.3% | 84.7% |
| MuSiQue | 17.2% | 42.4% | 42.4% | 55.7% |
| StrategyQA | 17.8% | 75.1% | 74.1% | 71.9% |

## Efficiency (pooled, per episode)

| Arm | Steps | Voluntary finish | Invalid/ep | Student tok | Teacher tok | Total tok | Plan tok | Latency s |
|---|---|---|---|---|---|---|---|---|
| Base | 2.98 | 2.4% | 0.123 | 4,442 | 0 | 4,442 | 601 | 11.4 |
| Guided | 2.95 | 4.5% | 0.005 | 5,389 | 5,582 | 10,972 | — | 21.9 |
| Trained (all-4) | 2.89 | 11.0% | 0.115 | 4,672 | 0 | 4,672 | 669 | 17.8 |
| Teacher | 2.76 | 100.0% | 0.051 | 3,889 | 0 | 3,889 | 557 | 11.1 |

## Cost (pooled)

| Arm | API $ | GPU $ | Training $ (amortized) | $/episode | $/episode excl. training | $/correct | Tokens/correct |
|---|---|---|---|---|---|---|---|
| Base | $0.000 | $0.310 | — | **$0.00042** | $0.00042 | $0.00139 | 14,880 |
| Guided | $0.167 | $1.537 | — | **$0.00228** | $0.00228 | $0.00377 | 18,133 |
| Trained (all-4) | $0.000 | $0.479 | $8.079 | **$0.01146** | $0.00064 | $0.01750 | 7,138 |
| Teacher | $0.114 | $0.000 | — | **$0.00015** | $0.00015 | $0.00021 | 5,380 |

## Paired significance on judge-correct (pooled)

| a → b | Δ pts | 95% CI | b wins / a wins | McNemar p |
|---|---|---|---|---|
| Base → Trained (all-4) | +35.6 | [+31.7, +39.4] | 288 / 22 | 0 |
| Base → Guided | +30.7 | [+26.9, +34.4] | 257 / 28 | 0 |
| Trained (all-4) → Guided | -5.0 | [-8.3, -1.6] | 65 / 102 | 0.0052 |
| Trained (all-4) → Teacher | +6.8 | [+3.4, +10.3] | 116 / 65 | 0.00018 |
| Guided → Teacher | +11.8 | [+8.0, +15.5] | 152 / 64 | 0 |
| Base → Teacher | +42.4 | [+38.7, +46.2] | 327 / 10 | 0 |

## Solved-question overlap

Solved by any arm: 626/747; by all four: 172; by none: 121.


See `report.html` for the full write-up, per-dataset tables and caveats.
