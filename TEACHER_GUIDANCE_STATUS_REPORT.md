# Teacher-Guidance — Full Status Report

**Repo:** `/shared/al-maamari/DeKIS/teacher-guidence` (fork of AgentSim, branch `feature/teacher-guidance`)
**Report date:** 2026-08-30 · **Last code activity:** 2026-08-02 · **Last data move:** 2026-08-09 (migrated to Frigga)
**Author of this report:** repo audit (docs + code + every reports/ and training_methods/ artifact on disk)

---

## 0. Executive summary

Teacher-Guidance is a **teacher–student agentic-trace generation framework** on top of AgentSim.
A small **student** LLM solves a multi-hop QA question with a fixed tool loop while a large
**teacher** LLM privately grades every step and is allowed to reveal only a *code-enforced*
amount of feedback (`guidance_level` 0–4). The traces are then distilled into a small student
that reasons **without a teacher at inference**.

Three things are true right now:

1. **The generation framework is production-grade and proven.** ~470 test functions, resumable
   multi-GPU runners, a 3-tier teacher cost router, a leakage gate enforced in code, an episode
   schema with full per-call telemetry, a trajectory-explorer UI, and a multi-dataset converter
   package (HotpotQA / 2Wiki / MuSiQue / StrategyQA).
2. **The scientific payoff is already demonstrated on HotpotQA.** ~30K+ episodes collected;
   guidance-SFT lifts Qwen3.5-0.8B from **3.0% → 60.2%** judge-correct with *no teacher at
   inference*, and beats the base-model-plus-live-teacher arm (32.6%) at **57% fewer tokens**.
   Cross-student, budget-sweep, budget-disclosure, wiki-memory A/B, teacher-only-vs-guidance,
   and a 4×5 protocol matrix have all been run.
3. **The pivot to the resource paper (the "full multi-dataset corpus") is planned and
   half-built, but mass generation has not started.** The 4-dataset converters, provenance
   stamping, dataset preflight, collection planner, model registry and release builder all
   exist and are tested; what does **not** exist yet is a prepared `tg_v1` dataset at scale,
   a generated collection plan, a working environment on the new cluster, and a single
   non-HotpotQA episode at scale.

**One-line status:** *the machine is built and validated on one dataset; the corpus it was
built to produce has not been generated yet, and the project is currently paused mid-pivot
after a machine migration.*

---

## 1. What the system does

### 1.1 The episode loop

```
[optional plan review]  student drafts plan → teacher critiques → student revises   (≤ N rounds,
                        or: teacher authors the plan, sanitized before the student sees it)
then, for each step up to a step budget B:
  student emits ONE JSON tool action
  → environment executes the tool deterministically (no LLM)
  → teacher privately evaluates the step (it sees the gold answer)
  → guidance renderer exposes only what guidance_level permits
  → leakage checker sanitizes the student-visible text
  → everything is logged (both sides, raw + parsed, tokens, latency, cost, provider)
final step is forced to `finish`; the committed answer may never regress to blank/"unknown"
```

**Tools (9):** `search, extract, verify, decompose, reformulate, synthesize, finish, wiki_read,
wiki_write`. Retrieval is **per-question local** (the distractor paragraph set for that qid,
BM25 with a deterministic lexical fallback) — self-contained, no web, fully reproducible.

### 1.2 The guidance ladder (the scientific core)

| Level | Name | What the student sees |
|---:|---|---|
| `skip_teacher` | control arm | nothing — no teacher call at all |
| 0 | binary reward | `0/1` score |
| 1 | continuous reward | `0.0–1.0` score |
| 2 | outcome feedback | score + short note |
| 3 | diagnostic feedback | score + diagnosis, **no** next action *(the workhorse)* |
| 4 | scaffolded guidance | score + diagnosis + next-step hint |

The separation between **`teacher_private_diagnosis`** (gold-aware) and
**`student_visible_guidance`** (sanitized, leak-gated) is enforced by the renderer + leakage
checker in code, not by prompting. This is the property that makes the corpus auditable and is
the paper's central claim to novelty.

### 1.3 Code map

| Area | Location | Notes |
|---|---|---|
| Core TG package | `agentsim/teacher_guidance/` | 26 modules: schemas, pydantic_schemas, json_utils, prompts, tool_executor, guidance_policy, leakage, metrics, plan_review/plan_execution, episode_exporter, provenance, model_registry, optimality, trace_quality, trace_selection, sft_* , dataset_split, wiki, viewer |
| Dataset converters | `agentsim/teacher_guidance/converters/` | `base` (+`validate_example`), `hotpot`, `twowiki`, `musique`, `strategyqa` |
| Agent components | `agentsim/components/control/` | `teacher_guided_agent_step.py`, `teacher_guided_plan_review.py` |
| Patched upstream | `simulation/modes/standard.py`, `workflow/executor.py`, `clients/llm_client.py`, `config.py`, `cli.py` | mode_config threading, FINISH verdict, provider router, `viewer` subcommand |
| Scripts (43) | `scripts/` | prepare/smoke/plan/collect/consolidate/aggregate/validate/export/report/release |
| Templates | `templates/workflows` (18), `templates/simulations` (77, incl. 32 `hardq_xr*`) | budgets b1…b30, guidance sweep g0–g4, plan-review variants |
| Training | `training_methods/` (48 GB) | m1–m5 methods + `exp_cross_student`, `exp_teacher_only`, `exp_unseen100`, shared `common/` |
| Tests | `tests/` + `tests/teacher_guidance/` | **471 test functions** across 44 files |

### 1.4 Models & routing

- **Teacher = ordered fallback router**, re-evaluated on every call, recording which provider
  actually served. Historic workhorse: `gpt-oss-120b` (FAU free academic gateway → OpenRouter
  free → OpenRouter paid). Hard wall-clock timeout (`FAU_TIMEOUT`, 45 s) + recoverable circuit
  breaker, because FAU can hang open-connection forever (httpx timeouts are inter-byte).
- **Registry (2026-08-02, `model_registry.py`)** — the resource-paper roster:
  **Teachers:** `kimi-k3` (EdenAI ok / OpenRouter no credit), `glm-5.2` (EdenAI HTTP 500 →
  OpenRouter first), `deepseek-v4-flash` (FAU, free, verified).
  **Students:** `g9v3-3b`, `qwen-2b`, `qwen-0.8b`, `granite-3b` — all Apache-2.0, served by
  vLLM under their real HF ids (Qwen3.5 composites need a LoRA merge before vLLM serving).
- **Students historically used:** granite-4.1-3b (best), qwen3.5:2b, qwen3.5:0.8b, qwen3.5:4b,
  ornith-9b; **MiniCPM5-1B abandoned** (7% correct, collapses under grammar constraints).

---

## 2. Data assets on disk (23 GB under `data/`, 48 GB under `training_methods/`)

| Asset | Size / count | State |
|---|---|---|
| `data/simulation_output.tar` | **21 GB, archived** | ALL raw generation runs (traces_g3b_b*_3000, *_run100 families, wiki A/B v1+v2, budget_hidden, hard-question campaign…). Un-indexed; must be untarred or indexed before use |
| `data/datasets/hotpot_teacher_guidance_train3000` | 3,000 questions + corpus | the training question set behind the b-sweep |
| `data/datasets/hotpot_teacher_guidance_exp100 / _exp30 / (base 10)` | 100 / 30 / 10 | **pinned** evaluation sets — regenerating them breaks comparability with prior reports |
| `data/datasets/best_answer_v1/` | ds1 (2,735) · ds2 (2,255) · ds3 (3,000) · ds4 (118) | best-answer distillation sets built from the 6-budget × 3,000 matrix |
| `data/datasets/tg_smoke/{hotpotqa,2wikimultihopqa,musique}` | **50 questions each** | converter smoke output only. MuSiQue = paragraph-gold; 2Wiki = sentence-gold. **No StrategyQA** (needs its paragraph corpus downloaded) |
| `data/corpus/`, `data/seeds/` | 38 MB / 284 KB | legacy AgentSim ATC corpus (MSMARCO/Quasar-T/CausalQA) — not part of teacher-guidance |
| `training_methods/*/runs/` | 48 GB | adapters, eval episodes, judge verdicts, curves for m1–m5 and the three experiments |

**Episode counts generated to date (approx.):** 18,000 (6 budgets × 3,000 HotpotQA train)
+ 3,000 (teacher-only expert) + 6,000 (cross-student, 2×3,000) + ~500 (`*_run100` family)
+ 600 (4×5 matrix) + 300 (wiki A/B v2) + the hard-question escalation campaign + the earlier
pilots → the resource-paper plan cites **31.6K pilot episodes**, all HotpotQA.

---

## 3. Experiments done, and what they showed

### 3.1 Generation-era studies (student behaviour under guidance)

**a) 100-question runs, teacher `gpt-oss-120b`, budget 12, disclosed budget** (`reports/fau_*`)

| Report | Student | Answer-correct | Grounded | Teacher cost |
|---|---|---|---|---|
| `fau_gptoss_run100` | qwen3.5:2b (FAU only) | 71% | 94% | $0 |
| `fau_router_run100` | qwen3.5:2b | 73% | — | $0 (100% FAU) |
| `fau_router_run100_qwen0p8b` | qwen3.5:0.8b | 58% | 97% | $0.0002 |
| `fau_router_run100_granite` | **granite4.1:3b** | **73%** | **98%** | $0 |
| `fau_router_run100_minicpm` | MiniCPM5-1B | 7% | — | abandoned |

**b) Budget-disclosure ablation** (`reports/budget_mode_comparison`, n=10, qwen2b) — *hidden*
budget beat *disclosed*: 3 vs 2 correct, 5 vs 4 grounded, 2 vs 1 natural finishes, 8.0 vs 8.4
mean steps. Hidden budget became the default.

**c) 4 students × 5 settings × 30 questions matrix** (`reports/experiment_matrix_2026-07-03`,
teacher glm-5.2) — **20/20 configs completed** (the HANDOFF's "6/20 paused" is stale).
Headline: setting **C (hidden max-20 budget)** is the best setting for every model
(qwen4b 0.87, ornith9b 0.90, qwen2b 0.77, qwen0.8b 0.23) but costs 2–4× the steps/tokens;
setting **E (no teacher)** is the worst for every model; **D (teacher writes the plan)** helps
EM/F1 but not correctness; small models burn the whole hidden budget (qwen0.8b median 20 steps).

**d) Budget sweep over 3,000 train questions** (`reports/best_answer_v1`, granite4.1:3b,
teacher `gpt-oss-120b`) — six runs, teacher-verdict correctness:

| run | b1 | b2 | b3 | b4 | b5h | b9h |
|---|---|---|---|---|---|---|
| correct rate | 29.1% | 62.7% | 70.2% | 74.3% | 75.2% | **78.4%** |
| doc recall (correct eps) | 0.00 | 0.83 | 0.90 | 0.92 | 0.92 | 0.93 |

265 / 3,000 questions were **never** answered correctly by any run → that pool seeded the
golden-100 hardest-question test and an escalating retry campaign (`run_hard_questions_xr.py`,
budgets 4 → 20 → 30), which recovered 118 of them (`ds4_hard_q_answers_118.jsonl`).

**e) Token/cost profile of a 3,000-episode run** (`reports/token_analysis_g3b_3000`): 85.8M
input + 13.7M visible output tokens; the teacher is **57.6% of billed tokens**, and reasoning
overhead (billed vs visible teacher output) is **2.82×** — the single biggest cost lever.

**f) Agent-wiki A/B v2** (`reports/wiki_ab_v2`, 3 students × 5 paired reps × 10 q × 2 arms):
surgical-edit wiki **fixed qwen2b's v1 collapse** (−21.6 pts → +5.9 pts, gold-in-wiki-missed
23 → 8, `unknown` finals → 0), stayed a **wash for qwen0.8b** (+1.0 pt but cover 5.0 → 2.8;
18/48 episodes had gold in the wiki and still answered wrong), and **regressed granite by
−11.3 pts** in 4 of 5 reps. Verdict: do not ship the wiki by default; also `stats.json`'s
`rewrites`/`keeps` fields are broken (`analyze_wiki_ab_v2.py::ep_stats` matches the wrong op
vocabulary).

### 3.2 Distillation results (the payoff)

**m1 — guidance-as-internal-thought SFT** (`training_methods/m1_sft`, granite-4.1-3b,
LoRA r32, 2 epochs, 4×A100, 17,961 train / 544 dev examples, 0 leaks):

| arm (budget 4, greedy, HF bf16) | dev EM | dev F1 | dev cover | doc recall | golden-100 cover |
|---|---|---|---|---|---|
| base (no teacher) | 2.4% | 0.092 | 18.1% | 0.669 | 2.0% |
| **m1 (internalized)** | **26.5%** | **0.368** | **67.5%** | 0.771 | 7.0% |
| m1 + live teacher | 26.5% | 0.372 | 75.9% | 0.813 | 8.0% |

Invalid-action steps collapse 51 → 2 (dev) and 61 → 0 (golden); the base model ends 75% of
episodes with no `finish` at all, m1 commits an answer in ~99%. The golden-100 (questions no
teacher-guided run ever solved) moved for the first time.

**Four-arm unseen-100 study** (`training_methods/exp_unseen100`, 100 fresh questions, 5 seeds,
budget 4, vLLM, judge = gpt-oss-120b) — judge-correct / tokens per question:

| student | base | base + teacher | **m1 (no teacher)** | m1 + teacher |
|---|---|---|---|---|
| Qwen3.5-0.8B | 3.0% · 8.4k | 32.6% · 17.7k | **60.2% · 7.6k** | 69.6% · 16.1k |
| Qwen3.5-2B | 21.2% · 9.9k | 60.2% · 18.8k | **63.6% · 7.6k** | 74.6% · 16.0k |
| granite-4.1-3B | 30.8% · 8.0k | 63.6% · 17.4k | 48.6% · 8.6k | 59.5% · 20.5k |

Reading: for the two Qwen students, the **internalized student beats the untrained student with
a live 120B teacher in the loop**, using ~57% fewer tokens and no teacher at all. For granite
(the strongest base) internalization keeps 76% of the teacher-in-loop accuracy. All differences
marked significant by paired t + Wilcoxon with Holm correction (granite m1 vs base: cover +0.25,
d=0.68, p≈7e-10).

**Cross-student transfer** (`reports/HTML_reports/cross_student_qwen_vs_granite.html`):
Qwen3.5-0.8B trained on **its own** traces vs on **granite's** traces over the same 3,000
questions, everything else pinned. Judge-correct is a dead heat (52.8% vs 53.7%, within seed
noise; base 7%), but the granite-trained model has **3.6× the exact-match**, +14 F1, finishes
voluntarily 2.4× as often, 0 invalid actions across 3,600 episodes, and evals 22% faster.
**Conclusion: trace quality beats trace ownership; answer-form and efficiency habits transfer
from the trace author.** Test-qid union (386) banned from both training sets; zero overlap
verified.

**Teacher-only vs guidance recipe** (`training_methods/exp_teacher_only`) — 3,000 episodes in
which gpt-oss-120b is *itself* the agent (no gold visible → zero leakage risk), 2,276 judged
correct (75.9%), 7,272 SFT examples. All three students trained. Evaluation is **incomplete**:
granite finished (m1 cover 31.2% vs base 26.2% — far below guidance-m1's 51.2%), qwen-0.8B and
qwen-2B runs failed/partially failed, and `report_compare/` is empty. **The headline comparison
"is it better to distil the teacher's critique of the student's own attempt, or the teacher's
own demonstrations?" is unanswered** — the partial evidence leans toward *guidance wins*.

**m2 (rejection sampling/ReST), m3 (KTO+DPO), m4 (GRPO), m5 (PRM/RLAIF)**: all five pipelines
pass full end-to-end SMOKE runs (dataset → train → dev eval → golden eval → report), datasets
are built and verified (37,396 KTO / 3,639 DPO pairs; 58,218 PRM examples, smoke PRM already
AUC 0.77; 2,913-question GRPO pool). **Their only committed results are the n=3 smoke runs with
throwaway 8-step adapters — no real training run for m2–m5 has ever been executed.**

---

## 4. The multi-dataset corpus pipeline (the resource-paper pivot)

### 4.1 Target — `resource_paper/PLAN.md` (2026-08-02)

Working title *"TeachTrace: Graded Teacher Guidance for Multi-Hop Retrieval Agents — a
counterfactual corpus of 180K interactive trajectories"*, target **NeurIPS Datasets &
Benchmarks**. The claim is the **intersection** nobody else has: same question + same student +
systematically varied supervision, with privileged and visible sides recorded separately, at
two granularities (plan + step), with full cost/provenance telemetry, from **open-weight
teachers over permissively licensed sources** (so the corpus is legally redistributable —
unlike FireAct/AgentInstruct).

Planned slices: **A TG-Core** 72K (counterfactual ladder: 4,000 q × 3 students × 6 conditions),
**B TG-Teachers** 16K, **C TG-Protocol** 24K, **D TG-Scale** 60K (bulk training data),
**E TG-Seeds** 4.5K, **F TG-Ablate** 8K ⇒ ~184.5K episodes / ~26K unique questions,
≈100–170 GPU-hours (teacher-API-bound). Baselines E1–E8, of which E2 (base→SFT lift) and E4
(cross-student) are **already done**; E1 (dose–response), E3 (cross-dataset zero-shot), E5
(teacher-size scaling), E6 (PRM/best-of-n) are the new reviewer-facing ones.

### 4.2 How a new dataset enters the pipeline

The engine is **already generic** — adding a dataset requires only emitting two JSONL shapes,
no engine changes:

```jsonc
// questions.jsonl
{"id","query","answer","type","level",
 "gold":{"answer","supporting_titles","supporting_facts","gold_doc_ids"},
 "retrieval_scope":{"backend":"hotpot_local","qid","candidate_doc_ids"}}

// corpus.jsonl
{"doc_id","qid","title","text","sentences","is_gold_doc","gold_sent_ids","source","split"}
```

End-to-end flow:

```
source (HF or official json)
   └─ scripts/prepare_dataset.py --dataset {hotpotqa|2wikimultihopqa|musique|strategyqa}
        └─ converters/<name>.py  →  base.validate_example() on EVERY row
        └─ writes questions.jsonl + corpus.jsonl + manifest.json
             (conversion stats, license, sha256 of both files, framework_commit, seed)
   └─ scripts/smoke_datasets.py --data-root ...      # OFFLINE preflight: schema, unique qids,
        every candidate doc exists, and the REAL retriever can reach each question's gold docs
        --online                                     # a few real episodes + episode audit
   └─ scripts/probe_models.py                        # is each teacher source alive; does its
                                                       verdict JSON parse (reasoning-model traps)
   └─ scripts/plan_collection.py                     # the collection matrix manifest
   └─ scripts/smoke_matrix.py --plan ...             # a few real episodes per teacher×student cell
   └─ [mass generation: run_tg_vllm.py / run_batched_parallel.py, resumable]
   └─ scripts/validate_teacher_guidance_run.py       # leakage/integrity, non-zero exit on leak
   └─ filter_traces → export_sft_dataset → split_sft_dataset → build_sft_report
   └─ scripts/build_release.py                       # full / train_safe views, cost+credential strip
```

**Dataset-specific handling already solved:** gold granularity is *represented, not faked* —
sentence-level sources (HotpotQA, 2Wiki) keep `supporting_facts`; paragraph-level sources
(MuSiQue, StrategyQA) emit none and report `supporting_fact_recall: null` rather than `0.0`
(which would silently depress every cross-dataset average). StrategyQA ships no distractors, so
the candidate set is **constructed** and disclosed per question via `constructed_candidates`.

**Collection planning** (`plan_collection.py`) partitions rather than crosses: each of the
**12 teacher×student combinations** gets its own question batch (default 5,000) **plus a shared
500-question anchor set** every combination sees — anchors give paired, per-question tests
across teachers and students, the remainder gives breadth at bounded cost. Each assignment then
runs under several named configs (`g3_plan`, `g0_plan`, `g3_noplan`, `g4_plan`, `skip_teacher`),
which is what turns 60K assignments into a ~120K-episode corpus with a counterfactual ladder.

**Release boundary** (`build_release.py`): raw traces keep everything; published records drop
`usage.cost`, `raw_response` bodies and the router chain (keeping `teacher_models_used`), and
ship as **`full`** (with privileged teacher reasoning) and **`train_safe`** (privileged fields
stripped — the default for fine-tuning). Provenance (`schema_version` 1.0, `framework_commit`,
`config_hash`, `generated_at`) is stamped on every episode; pre-2026-08-02 traces are flagged
`unversioned`.

### 4.3 Gap register (from PLAN §2.3), verified against the tree

| # | Gap | Status now |
|---|---|---|
| G1 | `dataset` hardcoded to `"hotpotqa"` | ✅ closed — threaded question-row → mode_config → metadata → exporter |
| G2 | no `schema_version` on episodes | ✅ closed — `provenance.py` |
| G3 | no dataset-level validation | ✅ closed — `validate_example`, `smoke_datasets.py`, release audit |
| G4 | retriever untested off HotpotQA | ✅ closed — retrieval-parity tests on 50 real q each for Hotpot/2Wiki/MuSiQue |
| G5 | no canonical config registry | ⏳ partial — `config_hash` exists; `resource_paper/configs/*.yaml` **not written** |
| G6 | teacher license/redistribution unverified | ⏳ open — Kimi-K3 is "Modified MIT", needs an audit before publishing its outputs |
| G7 | FAU throttles above ~24-way concurrency | ⏳ partial — circuit breaker shipped; **global rate limiter still missing** |
| G8 | 31.6K pilot traces predate several fixes | ⏳ open — regenerate under v1 configs, or ship as documented v0 |

---

## 5. Where the work actually stopped

Everything below is *built and tested* but has **never been run at scale**:

- **No `tg_v1` dataset exists.** Only 50-question smoke sets for 3 of 4 datasets; **StrategyQA
  has never been converted** (its paragraph corpus was never downloaded).
- **No collection plan generated.** `resource_paper/` contains only `PLAN.md` — no `plan/`,
  no `configs/`.
- **`smoke_matrix.py` has never been run against the new teacher roster** — at last probe
  (2026-08-02) only `deepseek-v4-flash` (FAU) and `kimi-k3` (EdenAI, **billed**) were working;
  `glm-5.2` EdenAI returns a deterministic HTTP 500 and every OpenRouter key is out of credit.
  **Teacher capacity is the #1 unresolved risk for a 120–180K-episode run.**
- **Zero non-HotpotQA episodes** have been generated.
- **m2–m5 have no real training runs**; only smoke.
- **teacher-only vs guidance comparison is unfinished** (2 of 3 students failed at eval).
- **Environment on this machine is gone.** No `.venv`, `.venv_train` or `.venv_vllm` in the tree;
  the login node has Python 3.13 (unsupported — TG needs 3.10–3.12) and no `nvidia-smi`.
  Every stored path in run manifests is `/root/DeKIS/teacher-guidence/...` (the old box), and
  all raw simulation output is now a single **21 GB un-indexed tar**.

### Known open bugs / hazards

1. **Voluntary "unknown" finishes the teacher accepts** — episodes stop at 2–7 steps with
   `stop_reason=teacher_accept` on a *voluntary* `finish: "unknown"`. Two candidate fixes
   (extend the answer-extraction fallback to any "unknown" finish; or make the teacher
   `reject_finish` on an empty/unknown answer) — **neither implemented**.
2. **`analyze_wiki_ab_v2.py::ep_stats` reports `rewrites`/`keeps` as always 0** — matches
   `"rewrite"`/`"KEEP"` while the real op vocabulary is `ADD/EDIT/DEL/ANSWER/NEXT/KEEP`. Anyone
   reading `wiki_ab_v2/stats.json` programmatically gets silent zeros.
3. **Reasoning-model parsing** — teachers differ in where chain-of-thought goes
   (`reasoning_content`, an EdenAI array item, or inline `<think>`); a naive parser returns the
   discarded draft. `strip_reasoning_blocks` + `probe_models.py` handle it, but each new teacher
   must be probed before use.
4. **Metric triangulation is unavoidable.** cover-match over-credits verbose answers, EM/F1
   under-credit them, and the teacher-verdict (≥0.40) is lenient; the three disagree materially
   (e.g. wiki A/B qwen0.8b: teacher-rate flat while cover halved). Always report all three.
5. **Contamination** — HotpotQA and 2Wiki are near-certainly in every model's pretraining;
   MuSiQue and the held-out Bamboogle/FanOutQA are the trustworthy generalization signals.
6. **Never regenerate a pinned dataset directory** (`--shuffle` reshuffles) — several published
   reports are pinned to the existing question files.
7. **Never co-author a commit as Claude** — standing rule for this repo.

---

## 6. Reading guide — where the truth lives

| Question | Read |
|---|---|
| How do I set up and run one experiment? | `TEACHER_GUIDANCE.md`, `TEACHER_GUIDANCE_UI.md` |
| Full engineering context up to 2026-07-06 | `HANDOFF.md` (*stale on two points: the 4×5 matrix is 20/20 complete, and the distillation pipeline HAS since been exercised end-to-end via m1*) |
| The corpus/paper design | `resource_paper/PLAN.md` |
| Training methods and their status | `training_methods/README.md` + each `m*/REPORT.md` |
| Teacher-only recipe context and gotchas | `training_methods/exp_teacher_only/PLAN.md` (§11 lessons is the highest-value section in the repo) |
| Measured results | `reports/*/REPORT.md`, `reports/HTML_reports/*.html`, `training_methods/*/runs/results_*.md` |
