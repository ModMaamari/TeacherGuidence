# gpt-oss-120b teacher (router) — student openbmb/MiniCPM5-1B (corrected handling) — 100 HotpotQA samples

**Config:** student `hf.co/openbmb/MiniCPM5-1B-GGUF:Q8_0` (local via Ollama, **student constrained decoding disabled** — `--student-no-schema`), teacher **cost router** fau→openrouter-free→paid; 3 plan-review rounds, running budget = 12 steps; 100 fixed questions; 8 GPUs, one sample per GPU in parallel (round-robin shards).

**Runtime:** ~33 min wall (1986s; all 8 workers exit 0). All 1,473 teacher calls free via FAU.

**Handling fixes for this model (vs. the earlier invalid 7% run):** MiniCPM5-1B was being
mishandled twice. (1) Ollama `/api/generate` bypassed its ChatML chat template → gibberish
(fixed earlier by switching to `/api/chat`). (2) llama.cpp **grammar-constrained decoding
collapsed its policy**: under the all-tools JSON-schema grammar it emitted the degenerate
shortest path (`synthesize {}` on 1098/1200 steps, only 2 searches in the whole run) while
producing valid, sensible JSON unconstrained (4/4 probes chose `search` with real queries).
Fix: per-run `student_use_response_schema=false` disables the student grammar (parse/repair
covers stragglers); the tiny finish-only grammar stays on for the forced final step so a
real answer is committed. Corrected behavior: **100% valid actions, 1,058 searches,
grounded 0%→41%, correct 7%→11%**. The remaining gap vs qwen2b/granite (73%) is genuine
model limitation in this bespoke protocol: it stays search-heavy (extract used only 28×),
ignores explicit teacher advice to extract, and never finishes naturally — unchanged across
temperature/top_p/thinking probes.

## Headline metrics

| Metric | Value |
|---|---|
| Episodes | 100 |
| Answer correct (cover-match) | **11/100 = 11%** |
| Answer grounded in evidence | 41/100 = 41% |
| Exact match | 1/100 = 1% |
| Mean token-F1 | 0.037 |
| Supporting-doc recall | 82.5% |
| Supporting-fact recall (verbatim extract) | 0.0% |
| Steps used (budget 12) | min 9, median 12, mean 11.9, max 12 |
| Mean step-efficiency | 0.33 |
| Total tokens (student+teacher) | 3,550,441 |

![headline](plots/headline.png)

![steps](plots/steps.png)

![by_type](plots/by_type.png)

## Stop reason & finishing behaviour

- Natural finish (`teacher_accept`): **0** episodes, 0 correct (0%).
- Forced finish (`budget_forced_finish`): **100** episodes, 11 correct (11%).

The student rarely satisfied the demanding teacher within 12 steps (only 0/100 natural finishes); most episodes ran to the budget cap. The finish-only forced-finish schema still produced a real, often-correct answer.

## Trace-quality filtering (SFT yield)

| Gate | Accepted | Top rejection reasons |
|---|---|---|
| Strict (natural finish req.) | 0/100 | {'not_natural_finish': 100, 'too_many_wasted_steps': 100, 'low_efficiency': 90, 'incorrect': 89} |
| Relaxed (allow forced, eff≥0.6) | 0/100 | {'low_efficiency': 96, 'too_many_wasted_steps': 96, 'incorrect': 89, 'ungrounded': 59} |

The dominant filter is **`gold_answer_leaked` (49 episodes)**: the strong teacher frequently states the gold answer in its feedback. The leakage sanitizer redacts it from the student, but the quality gate still excludes those traces so the SFT set only contains episodes the teacher solved *without revealing the answer*. The relaxed gate yields **0 clean traces** which the SFT exporter turned into leakage-free chat examples (healthcheck: clean).

## Takeaways

1. **gpt-oss-120b is a strong teacher.** 71% correct with a 2B student and 94% grounded answers is high for HotpotQA distractor; retrieval (doc recall 96.5%) is near-ceiling.
2. **Verbatim extraction is the student's weak point** (fact recall 3.5%): qwen2b struggles to copy exact supporting sentences, which is also why the teacher keeps rejecting finishes and episodes run to the budget.
3. **The FAU key is the throughput bottleneck**, not the GPUs: per-key rate limiting means teacher calls are globally throttled regardless of GPU parallelism. The backoff kept the run correct and failure-free.
4. **For higher SFT yield, tune the teacher to guide without revealing the answer** — gold-answer leakage in feedback is the single largest reason good traces are filtered out.
