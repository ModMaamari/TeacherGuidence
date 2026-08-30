# tg_v1 Collection — Plan and Pre-flight Test Report

**Date:** 2026-08-30 · **Status:** all gates green, **awaiting go/no-go**
**Target:** 8,000 teacher-guidance episodes — 2,000 per dataset × 4 datasets
**Teacher:** `fau/deepseek-ai/DeepSeek-V4-Flash` (NHR@FAU, free) · **Cost: $0**

---

## 1. The run in one table

| | |
|---|---|
| Episodes | **7,999** (2,000 + 2,000 + 2,000 + 1,999) |
| Datasets | HotpotQA · 2WikiMultihopQA · MuSiQue · StrategyQA |
| Teacher | `fau/deepseek-ai/DeepSeek-V4-Flash` — single teacher, **no router fallback** |
| Student | `fau/ibm-granite/granite-4.1-3b` — FAU-served (no GPU available until 2026-09-01) |
| Config | `g3_plan`: guidance level 3, plan review 3 rounds, step budget 3, budget hidden |
| Money cost | **$0** — both models on the free academic gateway |
| Wall-clock | **~50 min** at 24 workers · ~2.5 h at 8 · ~20 h serial |
| Output | `data/simulation_output/tgv1/` |
| Command | `python scripts/run_tgv1_collection.py --all --shards 6` |

---

## 2. Pre-flight tests — results

### 2.1 Dataset preparation and validation ✅

All four converted at scale with provenance manifests (licence, SHA-256, framework commit, seed 13).

| Dataset | Questions | Docs | Gold | Hop distribution | Licence |
|---|---:|---:|---|---|---|
| HotpotQA | 2,000 | 19,900 | sentence | 2-hop ×2000 | CC BY-SA 4.0 |
| 2WikiMultihopQA | 2,000 | 20,000 | sentence | 2-hop ×1574, 4-hop ×426 | Apache-2.0 |
| MuSiQue | 2,000 | 40,000 | paragraph | 2×1460, 3×425, 4×115 | CC BY 4.0 |
| StrategyQA | 1,999 | 24,982 | paragraph | 1×16, 2×556, 3×1054, 4×297, 5×76 | MIT |

- `smoke_datasets.py` offline preflight: **all checks passed** — schema conformance, unique qids, every candidate doc present, and **gold documents reachable by the real retriever**.
- **7,999 unique qids, 0 cross-dataset collisions.**
- StrategyQA: 1 example skipped (no usable evidence paragraphs) — expected and logged.
- Footprint: 136 MB. Disk free on `/shared`: 740 TB.

### 2.2 Model probes ✅

| Role | Model | JSON parse | Median latency | Max |
|---|---|---:|---:|---:|
| Teacher | `fau/deepseek-ai/DeepSeek-V4-Flash` | **5/5** | 0.34 s | 0.43 s |
| Student | `fau/ibm-granite/granite-4.1-3b` | **5/5** | 0.15 s | 0.21 s |

`provider_available` true for both. Teacher prompts were real grading turns, not toy pings.

### 2.3 Gateway concurrency ✅

| Concurrent calls | Wall | Median | p95 | Errors |
|---:|---:|---:|---:|---:|
| 4 | 1.94 s | 1.78 s | 1.78 s | 0 |
| 8 | 1.14 s | 0.54 s | 1.02 s | 0 |
| 16 | 0.86 s | 0.41 s | 0.66 s | 0 |
| 32 | 2.97 s | 0.69 s | 1.49 s | 0 |
| **64** | 1.96 s | 1.09 s | 1.65 s | **0** |

**No throttling to 64 concurrent calls.** This supersedes gap G7 ("FAU throttles above ~24"), which was measured on `gpt-oss-120b`, a different model on the same gateway. The planned 24 workers sit well inside this.

### 2.4 Online smoke — 12 real episodes (3 per dataset) ✅

Run through the **same templates** the production run uses; only `num_samples` differs.

| Dataset | n | Teacher calls | Steps | Teacher API | Correct | F1 | Doc recall | Stop reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HotpotQA | 3 | 5.3 | 2.7 | 16 s | **3/3** | 0.722 | 0.83 | forced ×2, teacher_accept ×1 |
| 2WikiMultihopQA | 3 | 5.3 | 3.0 | 14 s | **3/3** | 0.610 | 1.00 | forced ×3 |
| MuSiQue | 3 | 5.3 | 3.0 | 16 s | 1/3 | 0.360 | 1.00 | forced ×3 |
| StrategyQA | 3 | 6.0 | 3.0 | 17 s | **3/3** | 0.093 | 0.78 | forced ×3 |
| **Total** | **12** | 5.5 | 2.9 | 16 s | **10/12** | — | — | — |

Cost: **$0.00**. Parse failures: 0. Episodes missing metrics: 0.

### 2.5 Leakage audit ✅ **PASS**

The run validator exits 0 but reports flags; flags mean the guard **fired and sanitized**, which is not exposure. Verified at field level:

| Check | Result |
|---|---|
| Episodes / steps audited | 12 / 35 |
| Guard flags raised | 3 (`gold_answer_leaked` ×2, `hidden_title_leaked` ×1) |
| Sanitization applied | `[answer hidden]` substituted in every flagged case |
| **Gold answer visible in `student_visible_guidance`** | **0** |
| Gold echoed but already present in the question | 1 (a comparison question — *"Which was released first, Latin For Lovers or …?"*) |

Worked example — HotpotQA `5ae21393…`, gold `"Oklahoma"`:
`leakage_check.gold_answer_leaked: true` → student actually saw *"The verify tool returned 'supported: false' for the claim about **[answer hidden]**…"*. The teacher tried to name the gold; the renderer removed it before the student read it. **Working as designed.**

### 2.6 Sharded runner ✅

New `scripts/run_tgv1_collection.py`, tested on 12 real MuSiQue episodes across 4 shards:

| Check | Result |
|---|---|
| Shards completed | 4/4, rc=0, 12/12 episodes, 1.5 min |
| Duplicate qids across shards | **0** |
| Dataset label on episodes | `musique` ×12 (correct) |
| Provenance | schema `1.0`, commit `9154aaf130a3`, one `config_hash` for all 12 |
| Resume | re-run skipped all 4 shards ("already 3/3"), launched 0 workers |

Round-robin sharding (not contiguous), so no worker gets a biased slice of the question order.

### 2.7 Throughput

Measured episode wall time: **median 8.4 s**, mean 9.0 s (min 5.0, max 15.3).

| Workers | 8,000 episodes |
|---:|---:|
| 1 | 19.9 h |
| 4 (one per dataset) | 5.0 h |
| 8 | 2.5 h |
| 16 | 1.2 h |
| **24 (6 shards × 4 datasets)** | **~0.8 h** |

---

## 3. Execution plan

```bash
# 0. one-time: clear the runner's 12-episode test output so counts start clean
rm -rf data/simulation_output/tgv1/tgv1_musique_s*

# 1. launch (24 workers: 6 shards x 4 datasets)
python scripts/run_tgv1_collection.py --all --shards 6 --num-samples 2000

# 2. validate every dataset's output
for d in hotpotqa 2wikimultihopqa musique strategyqa; do
  python scripts/validate_teacher_guidance_run.py --run-dir data/simulation_output/tgv1/tgv1_${d}_s0
done

# 3. aggregate + audit
python scripts/episode_cost_report.py --runs data/simulation_output/tgv1/* --scale 8000
python scripts/aggregate_teacher_guidance_run.py --run-dir data/simulation_output/tgv1

# 4. consolidate shards into one run dir per dataset (viewer shows one run)
python scripts/consolidate_run_shards.py --help   # confirm flags for this layout

# 5. build the release views
python scripts/build_release.py --runs data/simulation_output/tgv1/* \
    --out data/release/tg_v1 --version 1.0
```

Resumable throughout: re-running step 1 after any interruption continues from where it stopped.

---

## 4. What to expect from the output

Based on the smoke, per 2,000-episode dataset: ~11,000 teacher calls, ~5,800 steps, and roughly **60–100% answer-correct depending on dataset** (HotpotQA/2Wiki/StrategyQA high, MuSiQue ~33%). MuSiQue's low correctness with perfect doc recall is the expected signature — the student retrieves the right evidence and fails the composition, which is exactly what MuSiQue was designed to expose. **Incorrect episodes are kept and labelled**; they are required for RL, PRM training and error analysis.

---

## 5. Known limitations of this plan

1. **Student is granite-4.1-3b served by FAU, not a local vLLM student.** Every GPU node is draining for a reboot; Slurm's earliest start is 2026-09-01 06:33. The API student is free and behaves identically in-protocol, but it is a different serving path than the eventual training-time student.
2. **One teacher, one config.** This produces a *training corpus*, not the counterfactual ladder — there is no guidance-level variation and no teacher comparison in this run. Those are separate slices.
3. **Budget 3 may under-serve the deep datasets.** 2Wiki 4-hop (426 questions) and StrategyQA 5-hop (76) plausibly need more steps; all smoke episodes on those hit `budget_forced_finish`. Consider a budget-5 variant for the deep subsets after inspecting v1 output.
4. **Smoke n=3 per dataset.** Accuracy figures above are indicative, not measurements.
5. **StrategyQA candidate sets are constructed** (gold + seeded distractors); disclosed per question via `constructed_candidates` and must be stated in the dataset card.

---

## 6. Go/no-go checklist

| Gate | Status |
|---|:--:|
| Four datasets converted, 7,999 questions, manifests + hashes | ✅ |
| Offline preflight incl. retriever gold-reachability | ✅ |
| Teacher + student probed, JSON parses 5/5 | ✅ |
| Gateway sustains 64 concurrent, 0 errors | ✅ |
| 12 real episodes end-to-end, 10/12 correct | ✅ |
| Zero hidden gold reaching the student | ✅ |
| Sharded runner: disjoint, provenance-stamped, resumable | ✅ |
| Templates validate (`--validate-only`) | ✅ |
| Cost | **$0** |
| Disk | 136 MB datasets; 740 TB free |

**Recommendation: ready to run.** Awaiting your confirmation.
