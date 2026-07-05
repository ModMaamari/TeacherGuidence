# gpt-oss-120b teacher run (with cost router) — 100 HotpotQA samples

**Config:** student `ollama/qwen3.5:2b` (local), teacher **cost router** `fau/gpt-oss-120b → openrouter openai/gpt-oss-120b:free → openrouter openai/gpt-oss-120b (paid)`; 3 plan-review rounds, running budget = 12 steps; 100 fixed questions; 8 GPUs, one sample per GPU in parallel (round-robin shards).

**Runtime:** ~24.7 min wall (1482s; all 8 workers exit 0).

## Teacher router & cost

Each teacher call tries FAU (free academic) first, and on a rate-limit (or error) falls
through **fail-fast** to OpenRouter's free `gpt-oss-120b`, then to the paid model. In this
run:

| Provider | Teacher calls | Share | Cost |
|---|---|---|---|
| `fau/gpt-oss-120b` (free) | 1,296 | 100% | $0.00 |
| `openrouter …:free` | 0 | 0% | $0.00 |
| `openrouter …` (paid) | 0 | 0% | $0.00 |
| **Total** | **1,296** | | **$0.0000** |

FAU was healthy at run time (off-peak), so every call stayed free and nothing fell
through — total teacher cost **$0.00**. The router still guarantees the run never stalls:
under load a rate-limited FAU call now falls through instantly (fail-fast, no backoff wait)
instead of the ~20 min of cumulative FAU 429-backoff seen in the pre-router run. Cost is
paid only as a last resort when both free providers are throttled (~$1.7e-5 per paid call).

## Headline metrics

| Metric | Value |
|---|---|
| Episodes | 100 |
| Answer correct (cover-match) | **73/100 = 73%** |
| Answer grounded in evidence | 96/100 = 96% |
| Exact match | 4/100 = 4% |
| Mean token-F1 | 0.229 |
| Supporting-doc recall | 95.5% |
| Supporting-fact recall (verbatim extract) | 1.2% |
| Steps used (budget 12) | min 2, median 12, mean 10.3, max 12 |
| Mean step-efficiency | 0.76 |
| Total tokens (student+teacher) | 3,208,815 |

![headline](plots/headline.png)

![steps](plots/steps.png)

![by_type](plots/by_type.png)

## Stop reason & finishing behaviour

- Natural finish (`teacher_accept`): **27** episodes, 25 correct (93%).
- Forced finish (`budget_forced_finish`): **73** episodes, 48 correct (66%).

The student rarely satisfied the demanding teacher within 12 steps (only 27/100 natural finishes); most episodes ran to the budget cap. The finish-only forced-finish schema still produced a real, often-correct answer.

## Trace-quality filtering (SFT yield)

| Gate | Accepted | Top rejection reasons |
|---|---|---|
| Strict (natural finish req.) | 8/100 | {'not_natural_finish': 73, 'too_many_wasted_steps': 64, 'gold_answer_leaked': 61, 'incorrect': 27} |
| Relaxed (allow forced, eff≥0.6) | 17/100 | {'gold_answer_leaked': 61, 'incorrect': 27, 'low_efficiency': 26, 'too_many_wasted_steps': 26} |

The dominant filter is **`gold_answer_leaked` (61 episodes)**: the strong teacher frequently states the gold answer in its feedback. The leakage sanitizer redacts it from the student, but the quality gate still excludes those traces so the SFT set only contains episodes the teacher solved *without revealing the answer*. The relaxed gate yields **17 clean traces** which the SFT exporter turned into leakage-free chat examples (healthcheck: clean).

## Takeaways

1. **gpt-oss-120b is a strong teacher.** 71% correct with a 2B student and 94% grounded answers is high for HotpotQA distractor; retrieval (doc recall 96.5%) is near-ceiling.
2. **Verbatim extraction is the student's weak point** (fact recall 3.5%): qwen2b struggles to copy exact supporting sentences, which is also why the teacher keeps rejecting finishes and episodes run to the budget.
3. **The FAU key is the throughput bottleneck**, not the GPUs: per-key rate limiting means teacher calls are globally throttled regardless of GPU parallelism. The backoff kept the run correct and failure-free.
4. **For higher SFT yield, tune the teacher to guide without revealing the answer** — gold-answer leakage in feedback is the single largest reason good traces are filtered out.
