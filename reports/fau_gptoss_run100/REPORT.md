# FAU gpt-oss-120b teacher run — 100 HotpotQA samples

**Config:** student `ollama/qwen3.5:2b` (local), teacher `fau/gpt-oss-120b` (NHR@FAU gateway); 3 plan-review rounds, running budget = 12 steps; 100 fixed questions; 8 GPUs, one sample per GPU in parallel (round-robin shards).

**Runtime:** ~28.5 min wall (all 8 workers exit 0). The FAU gateway heavily rate-limited the shared key (hundreds of HTTP 429s); the client's backoff absorbed them with zero failed episodes.

## Headline metrics

| Metric | Value |
|---|---|
| Episodes | 100 |
| Answer correct (cover-match) | **71/100 = 71%** |
| Answer grounded in evidence | 94/100 = 94% |
| Exact match | 1/100 = 1% |
| Mean token-F1 | 0.174 |
| Supporting-doc recall | 96.5% |
| Supporting-fact recall (verbatim extract) | 3.5% |
| Steps used (budget 12) | min 2, median 12, mean 11.0, max 12 |
| Mean step-efficiency | 0.75 |
| Total tokens (student+teacher) | 3,393,020 |

![headline](plots/headline.png)

![steps](plots/steps.png)

![by_type](plots/by_type.png)

## Stop reason & finishing behaviour

- Natural finish (`teacher_accept`): **17** episodes, 15 correct (88%).
- Forced finish (`budget_forced_finish`): **83** episodes, 56 correct (67%).

The student rarely satisfied the demanding teacher within 12 steps (only 17/100 natural finishes); most episodes ran to the budget cap. The finish-only forced-finish schema still produced a real, often-correct answer.

## Trace-quality filtering (SFT yield)

| Gate | Accepted | Top rejection reasons |
|---|---|---|
| Strict (natural finish req.) | 8/100 | {'not_natural_finish': 83, 'too_many_wasted_steps': 69, 'gold_answer_leaked': 60, 'incorrect': 29} |
| Relaxed (allow forced, eff≥0.6) | 22/100 | {'gold_answer_leaked': 60, 'low_efficiency': 29, 'too_many_wasted_steps': 29, 'incorrect': 29} |

The dominant filter is **`gold_answer_leaked` (60 episodes)**: the strong teacher frequently states the gold answer in its feedback. The leakage sanitizer redacts it from the student, but the quality gate still excludes those traces so the SFT set only contains episodes the teacher solved *without revealing the answer*. The relaxed gate yields **22 clean traces** which the SFT exporter turned into leakage-free chat examples (healthcheck: clean).

## Takeaways

1. **gpt-oss-120b is a strong teacher.** 71% correct with a 2B student and 94% grounded answers is high for HotpotQA distractor; retrieval (doc recall 96.5%) is near-ceiling.
2. **Verbatim extraction is the student's weak point** (fact recall 3.5%): qwen2b struggles to copy exact supporting sentences, which is also why the teacher keeps rejecting finishes and episodes run to the budget.
3. **The FAU key is the throughput bottleneck**, not the GPUs: per-key rate limiting means teacher calls are globally throttled regardless of GPU parallelism. The backoff kept the run correct and failure-free.
4. **For higher SFT yield, tune the teacher to guide without revealing the answer** — gold-answer leakage in feedback is the single largest reason good traces are filtered out.
