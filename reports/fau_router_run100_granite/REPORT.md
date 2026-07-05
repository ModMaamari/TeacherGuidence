# gpt-oss-120b teacher (router) — student ibm-granite/granite-4.1-3b — 100 HotpotQA samples

**Config:** student `hf.co/ibm-granite/granite-4.1-3b-GGUF:Q8_0` (local via Ollama), teacher **cost router** fau→openrouter-free→paid; 3 plan-review rounds, running budget = 12 steps; 100 fixed questions; 8 GPUs, one sample per GPU in parallel (round-robin shards).

**Runtime:** ~24.5 min wall (1469s; all 8 workers exit 0). Teacher router: all 1,392 calls free via FAU → $0.00.

**Student comparison (same config):** granite-4.1-3b **73% correct / 98% grounded** — ties qwen3.5:2b (73%) with the highest grounding and F1 (0.276) of the students tested, well above qwen3.5:0.8b (58%) and MiniCPM5-1B (7%). 100% valid JSON, good tool diversity (search/extract/verify/decompose).

## Headline metrics

| Metric | Value |
|---|---|
| Episodes | 100 |
| Answer correct (cover-match) | **73/100 = 73%** |
| Answer grounded in evidence | 98/100 = 98% |
| Exact match | 7/100 = 7% |
| Mean token-F1 | 0.276 |
| Supporting-doc recall | 88.5% |
| Supporting-fact recall (verbatim extract) | 12.6% |
| Steps used (budget 12) | min 3, median 12, mean 11.5, max 12 |
| Mean step-efficiency | 0.54 |
| Total tokens (student+teacher) | 3,313,632 |

![headline](plots/headline.png)

![steps](plots/steps.png)

![by_type](plots/by_type.png)

## Stop reason & finishing behaviour

- Natural finish (`teacher_accept`): **9** episodes, 9 correct (100%).
- Forced finish (`budget_forced_finish`): **91** episodes, 64 correct (70%).

The student rarely satisfied the demanding teacher within 12 steps (only 9/100 natural finishes); most episodes ran to the budget cap. The finish-only forced-finish schema still produced a real, often-correct answer.

## Trace-quality filtering (SFT yield)

| Gate | Accepted | Top rejection reasons |
|---|---|---|
| Strict (natural finish req.) | 1/100 | {'too_many_wasted_steps': 92, 'not_natural_finish': 91, 'gold_answer_leaked': 62, 'low_efficiency': 33} |
| Relaxed (allow forced, eff≥0.6) | 4/100 | {'low_efficiency': 70, 'too_many_wasted_steps': 70, 'gold_answer_leaked': 62, 'incorrect': 27} |

The dominant filter is **`gold_answer_leaked` (62 episodes)**: the strong teacher frequently states the gold answer in its feedback. The leakage sanitizer redacts it from the student, but the quality gate still excludes those traces so the SFT set only contains episodes the teacher solved *without revealing the answer*. The relaxed gate yields **4 clean traces** which the SFT exporter turned into leakage-free chat examples (healthcheck: clean).

## Takeaways

1. **gpt-oss-120b is a strong teacher.** 71% correct with a 2B student and 94% grounded answers is high for HotpotQA distractor; retrieval (doc recall 96.5%) is near-ceiling.
2. **Verbatim extraction is the student's weak point** (fact recall 3.5%): qwen2b struggles to copy exact supporting sentences, which is also why the teacher keeps rejecting finishes and episodes run to the budget.
3. **The FAU key is the throughput bottleneck**, not the GPUs: per-key rate limiting means teacher calls are globally throttled regardless of GPU parallelism. The backoff kept the run correct and failure-free.
4. **For higher SFT yield, tune the teacher to guide without revealing the answer** — gold-answer leakage in feedback is the single largest reason good traces are filtered out.
