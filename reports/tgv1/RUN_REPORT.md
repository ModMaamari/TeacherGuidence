# tg_v1 Collection — Run Report

| Dataset | Episodes | Expected | Correct | EM | Grounded | Mean F1 | Doc recall | Steps | Teacher calls | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 2000 | 2000 | 1385 (69%) | 489 (24%) | 1629 (81%) | 0.385 | 0.862 | 2.92 | 5.3 | $0.0051 |

**2wikimultihopqa** — stop reasons: `{'budget_forced_finish': 1839, 'teacher_accept': 161}` · provenance: `[('1.0', '73fe4646ace2', '1716bb7efcbd32d0'), ('1.0', 'a647ae9cb016', '1716bb7efcbd32d0')]` · unique qids 2000/2000

| hotpotqa | 2000 | 2000 | 1296 (65%) | 746 (37%) | 1802 (90%) | 0.498 | 0.807 | 2.85 | 5.2 | $0.0109 |

**hotpotqa** — stop reasons: `{'budget_forced_finish': 1719, 'teacher_accept': 281}` · provenance: `[('1.0', '73fe4646ace2', '1716bb7efcbd32d0'), ('1.0', 'a647ae9cb016', '1716bb7efcbd32d0')]` · unique qids 2000/2000

| musique | 2000 | 2000 | 595 (30%) | 270 (14%) | 1491 (75%) | 0.241 | 0.676 | 2.97 | 5.3 | $0.0367 |

**musique** — stop reasons: `{'budget_forced_finish': 1965, 'teacher_accept': 35}` · provenance: `[('1.0', '73fe4646ace2', '1716bb7efcbd32d0'), ('1.0', 'a647ae9cb016', '1716bb7efcbd32d0')]` · unique qids 2000/2000

| strategyqa | 1999 | 1999 | 1111 (56%) | 610 (31%) | 1139 (57%) | 0.332 | 0.817 | 2.96 | 5.7 | $0.5767 |

**strategyqa** — stop reasons: `{'budget_forced_finish': 1925, 'teacher_accept': 74}` · provenance: `[('1.0', '73fe4646ace2', '1716bb7efcbd32d0'), ('1.0', 'a647ae9cb016', '1716bb7efcbd32d0')]` · unique qids 1999/1999

## Totals

- Episodes: **7999**
- Answer-correct: **4387 (54.8%)**
- Exact match: 2115 (26.4%)
- Grounded: 6061 (75.8%)
- Duplicate qids: 0
- **Cost: $0.6294**

## Teachers and serving routes

| Model · route | Episodes |
|---|---:|
| `fau/deepseek-ai/DeepSeek-V4-Flash` | 5475 |
| `fau/deepseek-ai/DeepSeek-V4-Flash-0731` | 2135 |
| `edenchat/flexai/DeepSeek-V4-Flash-0731` | 306 |
| `edenchat/qwen/deepseek-v4-flash-0731` | 83 |

## Leakage

- Guard flags fired (detected **and sanitized**): `{'hidden_title_leaked': 3367, 'gold_answer_leaked': 2159, 'hidden_span_leaked': 10, 'feedback_fallback_used': 27}`
- Gold echoed but already present in the question (comparison items): 314
- Boolean gold matched in a "give a yes/no answer" instruction (format, not content): 24
- **Hidden gold visible to the student: 0** ✅ **PASS**


## Run narrative

Collection ran 2026-08-30 19:34Z → 2026-08-31 14:41Z, interrupted three times, resumed
each time from per-shard checkpoints with no work repeated:

| Phase | Teacher · route | Episodes | Ended by |
|---|---|---:|---|
| 1 | DeepSeek-V4-Flash · FAU | 5,475 | session exit killed the workers' process group |
| 2 | DeepSeek-V4-Flash-0731 · FAU | 2,135 | FAU backend `h11-21` returned 500s |
| 3 | DeepSeek-V4-Flash-0731 · EdenAI/flexai | 306 | switched by request |
| 4 | DeepSeek-V4-Flash-0731 · EdenAI/qwen | 83 | switched back (6× slower, 15× dearer) |
| 5 | DeepSeek-V4-Flash-0731 · EdenAI/flexai | remainder | **completed** |

**Two models, four routes.** `DeepSeek-V4-Flash` (5,475) and its `0731` checkpoint (2,524),
the latter served three ways after both FAU DeepSeek backends failed. Split it by
`teacher_models_used`, **not** by `config_hash` — `provenance.config_hash` deliberately
excludes the teacher id (it treats routing as an operational detail), so all 7,999 episodes
share one hash despite the teacher change. That is a known limitation to fix before the
next collection.

**Metrics were stable across every switch.** Overall answer-correct sat at 54.6–55.1% from
n=3,508 onward, and F1 within 0.357–0.364, while three teacher/route changes landed. This
shows indistinguishable *aggregate* outcomes, not trajectory-level equivalence — every
question was answered once, by whichever teacher was live, so no paired comparison is
possible from this data. A paired run on shared questions would be needed to claim the
routes are interchangeable.

**Failure handling.** 28 episodes were lost to transient gateway failures and re-collected
by `scripts/retry_failed_episodes.py`; the final corpus has **0 errored episodes**. Two
outages also produced 1,395 sample directories carrying a `_SUCCESS` marker and a checkpoint
entry but no episode — which would have been skipped forever on resume, silently holing the
corpus. `scripts/clean_failed_samples.py` removed them and un-marked their checkpoints.

**Cost: $0.63 total** — $0 for phases 1–2 (free FAU gateway) and for every student call;
the EdenAI phases account for all of it, of which $0.577 is StrategyQA, mostly on the
expensive qwen route.
