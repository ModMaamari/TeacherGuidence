# Teacher-Guidance Corpus — Budget Report

**Date:** 2026-08-30 · **Billing:** EdenAI, measured per call · **EUR→USD:** 1.1607 ([open.er-api.com](https://open.er-api.com/v6/latest/EUR), 2026-08-30)

All figures below are derived from **real episodes billed by EdenAI**, not list prices.

---

## 1. What was measured

| Item | Value |
|---|---|
| Configuration | `g3_plan` — guidance level 3, plan review (3 rounds), step budget 3, budget hidden |
| Student | `fau/ibm-granite/granite-4.1-3b` — free FAU gateway, **$0 of every figure below** |
| Teacher endpoint | `https://api.edenai.run/v3/chat/completions` |
| Episodes run | 26 (5 per teacher on HotpotQA; 3 per dataset with Qwen3.8) |
| Cost basis | EdenAI's per-call `cost` field, summed over every logged LLM call |
| Total spend to produce this report | ≈ $2.40 |

**A teacher-guidance episode costs $0.042–$0.095**, set mainly by the protocol (~4 teacher calls per episode), not by question difficulty.

---

## 2. Measured cost per episode

### 2.1 By teacher — HotpotQA, n=5 each

| Teacher | EdenAI model id | $/episode | sd | Teacher calls | API time | answer_correct |
|---|---|---:|---:|---:|---:|---:|
| **Qwen3.8-2.4T-A95B** | `qwen/qwen3.8-2.4t-a95b` | **$0.0448** | ±0.0134 | 4.2 | 200 s | 4/5 |
| Kimi K3 | `qwen/kimi-k3` | $0.0599 | ±0.0142 | 4.2 | 141 s | 3/5 |
| GLM-5.3 ⚠ | `deepinfra/zai-org/GLM-5.3` | $0.0800 | ±0.0232 | 9.2 | 407 s | 4/5 |

⚠ **GLM-5.3's figure reflects a misconfiguration, not its price.** At `teacher_max_tokens: 2500` it returns empty content with `finish_reason: length` — its inline reasoning consumes the budget before the JSON verdict — so every call is retried (9.2 calls vs 4.2). Raising the cap to 8000 yields valid JSON for $0.0137 vs $0.0123 wasted, but then exceeds the 180 s EdenAI timeout. **Treat GLM-5.3's numbers as an upper bound pending a corrected run.**

### 2.2 By dataset — teacher Qwen3.8-2.4T-A95B

| Dataset | n | $/episode | sd | Hops | Gold | Licence | answer_correct | grounded |
|---|---:|---:|---:|---|---|---|---:|---:|
| StrategyQA | 3 | **$0.0425** | ±0.0107 | 2–5 | paragraph | MIT | 2/3 | 1/3 |
| HotpotQA | 5 | $0.0448 | ±0.0134 | 2 | sentence | CC BY-SA 4.0 | 4/5 | 5/5 |
| MuSiQue | 3 | $0.0470 | ±0.0097 | 2–4 | paragraph | CC BY 4.0 | 1/3 | 3/3 |
| 2WikiMultihopQA | 3 | $0.0530 | ±0.0038 | 2–4 | sentence | Apache-2.0 | 1/3 | 1/3 |

Spread across datasets is only **±12%**. 2Wiki is dearest because the teacher writes longer critiques on 4-hop comparisons (11.3k output tokens vs 9.3k).

---

## 3. Cost per episode — teacher × dataset

Derived: dataset cost × that teacher's HotpotQA ratio (Qwen 1.00, Kimi 1.34, GLM 1.79). The two measured axes are marked ●.

| $/episode | HotpotQA | 2WikiMultihopQA | MuSiQue | StrategyQA |
|---|---:|---:|---:|---:|
| **Qwen3.8-2.4T-A95B** | ●$0.0448 | ●$0.0530 | ●$0.0470 | ●$0.0425 |
| **Kimi K3** | ●$0.0599 | $0.0709 | $0.0629 | $0.0568 |
| **GLM-5.3** ⚠ | ●$0.0800 | $0.0947 | $0.0840 | $0.0759 |

---

## 4. Scale scenarios

Single teacher, dataset-balanced. **Plan** = measured mean +30% contingency, matching the largest observed per-episode standard deviation. **Budget against the Plan column.**

| Teacher | 1,000 | 10,000 | 120,000 |
|---|---:|---:|---:|
| **Qwen3.8-2.4T-A95B** | $46.79 · *plan $60.83* | $467.93 · *plan $608.31* | $5,615 · *plan $7,300* |
| Kimi K3 | $62.60 · *plan $81.38* | $625.98 · *plan $813.77* | $7,512 · *plan $9,765* |
| GLM-5.3 ⚠ | $83.62 · *plan $108.71* | $836.24 · *plan $1,087* | $10,035 · *plan $13,045* |

In EUR — cheapest teacher: **1k ≈ €40** (plan €52) · **10k ≈ €403** (plan €524) · **120k ≈ €4,838** (plan €6,289).

---

## 5. The mixed corpus — 1,000 episodes per teacher × dataset

3 teachers × 4 datasets × 1,000 = **12,000 episodes**.

| USD | HotpotQA | 2Wiki | MuSiQue | StrategyQA | **Row total** |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-2.4T-A95B | $44.75 | $52.98 | $46.99 | $42.45 | **$187.17** |
| Kimi K3 | $59.87 | $70.87 | $62.86 | $56.79 | **$250.39** |
| GLM-5.3 ⚠ | $79.98 | $94.68 | $83.97 | $75.87 | **$334.50** |
| **Column total** | $184.60 | $218.53 | $193.82 | $175.11 | **$772.06** |

| | USD | EUR |
|---|---:|---:|
| Measured mean | **$772.06** | **€665.16** |
| **Planning figure (+30%)** | **$1,003.68** | **€864.70** |

**Budget €865 for the 12,000-episode mixed corpus; expect ≈ €665.** Fixing GLM-5.3's token cap should cut its row materially, taking the total toward €550–600.

---

## 6. Cost levers

1. **Run bulk generation on the free FAU gateway.** `deepseek-ai/DeepSeek-V4-Flash` and `gpt-oss-120b` were both verified working and free on 2026-08-30. Paid teachers are only needed where teacher *identity* is the variable — the teacher-comparison slice. This is the difference between €865 and near-zero for the bulk corpus.
2. **Fix GLM-5.3 before buying any of it.** `teacher_max_tokens: 8000` **and** an EdenAI timeout ≥ 600 s. Note it expands reasoning to fill whatever cap it is given (16k → 5,943 tokens, $0.0243), so 8000 is the sweet spot, not a floor.
3. **Step budget drives cost roughly linearly.** These figures are budget 3. A budget-5 run on the deeper datasets should be planned at ~$0.07–0.08/episode.
4. **The teacher is ~100% of the bill.** The student is free on FAU; a local vLLM student is also $0. Only teacher tokens cost money.

---

## 7. Caveats

- **n is small** — 5 episodes per teacher, 3 per dataset. The ±30% contingency exists for this reason.
- **Nine of twelve matrix cells are derived**, not measured (§3).
- **GLM-5.3 is an upper bound** (§2.1).
- **Student deviation:** granite-**4.1**-3b served by FAU, not local granite-4.2 — every GPU node was draining for a reboot, and 4.2 is not in the FAU catalogue. Student calls are free either way, so this affects trajectory shape, not cost.
- **Wall-clock may bind before budget.** At ~200 s of teacher API time per episode, 12,000 episodes is ~660 hours serialized. EdenAI concurrency limits are unmeasured.
- **Accuracy is not a cost figure** but is reported because it decides how many episodes you need: HotpotQA 4/5 correct vs 1/3 on 2Wiki and MuSiQue.

---

## 8. Resources

**Endpoints**
- EdenAI chat completions — `https://api.edenai.run/v3/chat/completions` · [docs.edenai.co](https://docs.edenai.co/)
- NHR@FAU LLM gateway (free, academic) — `https://hub.nhr.fau.de/api/llmgw/v1` · [hpc.fau.de](https://hpc.fau.de/)

**Teachers** — `qwen/qwen3.8-2.4t-a95b` ([Qwen on HF](https://huggingface.co/Qwen)) · `qwen/kimi-k3` ([moonshotai on HF](https://huggingface.co/moonshotai)) · `deepinfra/zai-org/GLM-5.3` ([zai-org on HF](https://huggingface.co/zai-org))

**Student** — [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b)

**Datasets**
- [HotpotQA](https://hotpotqa.github.io/) — CC BY-SA 4.0
- [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) — Apache-2.0
- [MuSiQue](https://github.com/StonyBrookNLP/musique) — CC BY 4.0
- [StrategyQA](https://allenai.org/data/strategyqa) — MIT · [dataset zip](https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip)

**Reproduce**
```bash
python scripts/prepare_dataset.py --dataset strategyqa --input strategyqa_train.json \
  --paragraphs strategyqa_train_paragraphs.json --split train --out data/datasets/tg_smoke/strategyqa
python scripts/smoke_datasets.py --data-root data/datasets/tg_smoke
python -m agentsim.cli simulate costprobe_ds_musique
python scripts/episode_cost_report.py --runs data/simulation_output/costprobe/* --scale 1000 10000 120000
python scripts/build_budget_report.py
```

**Data behind this report** — `reports/costprobe_teachers.json` · `reports/costprobe_datasets.json` · `reports/budget/budget_data.json`
