# gpt-oss-120b teacher (router) — student openbmb/MiniCPM5-1B — 100 HotpotQA samples

**Config:** student `hf.co/openbmb/MiniCPM5-1B-GGUF:Q8_0` (local via Ollama), teacher **cost router** fau→openrouter-free→paid; 3 plan-review rounds, running budget = 12 steps; 100 fixed questions; 8 GPUs, one sample per GPU in parallel (round-robin shards).

**Runtime:** ~19.4 min wall (1165s; all 8 workers exit 0). Teacher router: 1,493/1,494 calls free via FAU, 1 paid fallthrough → ~$0.0001.

**Note — required a fix:** MiniCPM5 is a ChatML/instruct model; the pipeline originally called Ollama's `/api/generate` (raw prompt), which bypassed the chat template and produced pure gibberish. Switched to `/api/chat` (applies the template) → 100% valid JSON, coherent reasoning. The low score below is genuine model weakness, not the earlier bug: MiniCPM5-1B over-uses `synthesize`, rarely retrieves evidence (0% grounded), and fills the answer field poorly — far below qwen3.5:2b (73%) and qwen3.5:0.8b (58%).

## Headline metrics

| Metric | Value |
|---|---|
| Episodes | 100 |
| Answer correct (cover-match) | **7/100 = 7%** |
| Answer grounded in evidence | 0/100 = 0% |
| Exact match | 0/100 = 0% |
| Mean token-F1 | 0.020 |
| Supporting-doc recall | 1.0% |
| Supporting-fact recall (verbatim extract) | 0.0% |
| Steps used (budget 12) | min 12, median 12, mean 12.0, max 12 |
| Mean step-efficiency | 0.17 |
| Total tokens (student+teacher) | 1,927,561 |

![headline](plots/headline.png)

![steps](plots/steps.png)

![by_type](plots/by_type.png)

## Stop reason & finishing behaviour

- Natural finish (`teacher_accept`): **0** episodes, 0 correct (0%).
- Forced finish (`budget_forced_finish`): **100** episodes, 7 correct (7%).

The student rarely satisfied the demanding teacher within 12 steps (only 0/100 natural finishes); most episodes ran to the budget cap. The finish-only forced-finish schema still produced a real, often-correct answer.

## Trace-quality filtering (SFT yield)

| Gate | Accepted | Top rejection reasons |
|---|---|---|
| Strict (natural finish req.) | 0/100 | {'ungrounded': 100, 'not_natural_finish': 100, 'low_efficiency': 100, 'too_many_wasted_steps': 100} |
| Relaxed (allow forced, eff≥0.6) | 0/100 | {'ungrounded': 100, 'low_efficiency': 100, 'too_many_wasted_steps': 100, 'incorrect': 93} |

The dominant filter is **`gold_answer_leaked` (12 episodes)**: the strong teacher frequently states the gold answer in its feedback. The leakage sanitizer redacts it from the student, but the quality gate still excludes those traces so the SFT set only contains episodes the teacher solved *without revealing the answer*. The relaxed gate yields **0 clean traces** which the SFT exporter turned into leakage-free chat examples (healthcheck: clean).

## Takeaways

1. **gpt-oss-120b is a strong teacher.** 71% correct with a 2B student and 94% grounded answers is high for HotpotQA distractor; retrieval (doc recall 96.5%) is near-ceiling.
2. **Verbatim extraction is the student's weak point** (fact recall 3.5%): qwen2b struggles to copy exact supporting sentences, which is also why the teacher keeps rejecting finishes and episodes run to the budget.
3. **The FAU key is the throughput bottleneck**, not the GPUs: per-key rate limiting means teacher calls are globally throttled regardless of GPU parallelism. The backoff kept the run correct and failure-free.
4. **For higher SFT yield, tune the teacher to guide without revealing the answer** — gold-answer leakage in feedback is the single largest reason good traces are filtered out.
