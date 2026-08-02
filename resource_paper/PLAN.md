# Resource Paper Plan — a large-scale corpus of interactive teacher–student agentic trajectories

**Working title.** *TeachTrace: Graded Teacher Guidance for Multi-Hop Retrieval Agents — a
counterfactual corpus of 180K interactive trajectories.*

**Status.** Draft plan, 2026-08-02. Written after a full audit of the framework
(`agentsim/teacher_guidance/`, `training_methods/`, `scripts/`) and of the 31.6K pilot
episodes already on disk.

---

## 1. Thesis: what makes this an A\* resource, not just another trace dump

Agent-trajectory datasets already exist (AgentInstruct, FireAct, AgentBank/AgentGym,
Lumos, ToolBench). Step-level supervision datasets exist (PRM800K, Math-Shepherd). What
does **not** exist, and what this framework uniquely produces, is the **intersection**:

> The same question, solved by the same student, under **systematically varied amounts of
> teacher supervision** — with the teacher's *privileged* reasoning and the student's
> *visible* guidance recorded as separate, leakage-audited fields.

Four properties, none of which the related work has together:

| Property | What we have | Why it matters | Who else has it |
|---|---|---|---|
| **Counterfactual supervision ladder** | Same (question, student, seed) run at `skip_teacher` and guidance levels 0–4 | Enables a *dose–response curve*: how much supervision does a 0.8B model actually need? Paired within-question tests (McNemar), not cross-population comparisons | Nobody |
| **Privileged/visible separation** | `teacher_private_diagnosis` (sees gold) vs `student_visible_guidance` (sanitized, leak-gated) | Directly instantiates Learning Using Privileged Information (LUPI) for LLM agents; supports distillation research that needs both sides | Nobody at scale |
| **Two supervision granularities** | Plan-level review rounds (plan → critique → revision) **and** per-step critique with continuous scores | Supports PRM training, plan-level RL, and step-vs-plan ablations from one corpus | Partially (PRM800K = step only, math only) |
| **Full cost/provenance telemetry** | Every LLM call: model actually served, prompt/completion tokens, latency, USD cost, router fallback | Makes efficiency and cost-of-supervision a first-class research object | Rare |

**Plus a licensing advantage that is itself a contribution.** FireAct, AgentInstruct and
most trajectory corpora are distilled from proprietary models (GPT-4), leaving
redistribution rights murky and forbidding competing-model training. We will use
**open-weight teachers only** (gpt-oss-120b Apache-2.0, and peers), over
**permissively licensed source datasets**, giving a corpus that is *fully redistributable
and legally clean for downstream training*. This should be stated explicitly in the
abstract — reviewers value it.

**One-line pitch.** *We release 180K multi-hop agentic trajectories in which an open-weight
teacher supervises small students at five graded levels of information exposure, enabling
the first systematic study of how much guidance a small agent needs — and providing
train-ready data that lifts a 0.8B agent from 7% to 53% end-to-end accuracy.*

---

## 2. Framework audit — what exists, what is missing

### 2.1 What is already production-grade

- **Episode schema is deep and already research-ready.** Per episode: 23 top-level fields;
  per step: 24 fields (full student prompt, raw output, parsed action, tool observation,
  full teacher prompt/raw, private diagnosis, rendered guidance, leakage check, per-step
  metrics, timings); per plan review: 25 fields including every revision round.
- **Per-call LLM logging** (`llm_call_log.timed_completion`): model *actually served*,
  usage, cost, latency, attempt index — survives router fallback.
- **Quality machinery already written and tested**: `leakage.py` (sanitize + detect),
  `trace_quality.py` (accept/reject gate), `trace_selection.py` (canonical trace per qid),
  `sft_diversity.py` (near-dup capping, tool-signature diversity), `optimality.py`
  (wasted steps, repeats, efficiency), `metrics.py` (SQuAD-normalized EM/F1, doc/fact recall,
  groundedness).
- **Guidance levels 0–4 are enforced in code**, not by prompt — the renderer constructs the
  student-visible object and the leakage checker sanitizes it. This is auditable, which is
  exactly what a reviewer will probe.
- **`skip_teacher`** gives the true no-guidance control arm.
- **Nine tools**: `search, extract, verify, decompose, reformulate, synthesize, finish,
  wiki_read, wiki_write`.
- **Scale infrastructure**: resumable vLLM runner (`scripts/run_tg_vllm.py`) with
  answered/unanswered tracking, one server per GPU, 18–31 episodes/min at 24 workers;
  post-hoc LLM judge; GPU samplers; **396 passing tests**.
- **Validated pilot results**: SFT lifts Qwen3.5-0.8B from 7% → 53% judge-correct;
  cross-student transfer study; teacher-only-vs-guidance study.

### 2.2 The dataset contract is already generic

Adding a new QA dataset requires **only** emitting two JSONL shapes — no engine changes:

```jsonc
// questions.jsonl
{"id", "query", "answer", "type", "level",
 "gold": {"answer", "supporting_titles", "supporting_facts", "gold_doc_ids"},
 "retrieval_scope": {"backend": "hotpot_local", "qid", "candidate_doc_ids"}}

// corpus.jsonl
{"doc_id", "qid", "title", "text", "sentences", "is_gold_doc", "gold_sent_ids",
 "source", "split"}
```

`HotpotLocalRetriever` groups by `qid` and ranks within a question's candidates (BM25 with
a deterministic lexical fallback). **Any distractor-style multi-hop dataset fits directly.**

### 2.3 Gaps that must be closed *before* mass generation

| # | Gap | Severity | Fix |
|---|---|---|---|
| G1 | `dataset` is **hardcoded** to `"hotpotqa"` in 3 places (`schemas.py:265`, `hotpot_converter.py:108`, `episode_exporter.py:164`) | **Blocking** — every trace would be mislabeled | Thread a `dataset` field through `mode_config` → metadata → exporter |
| G2 | No `schema_version` on episodes | **Blocking** — cannot version a public release | Add `schema_version: "1.0"` + `framework_commit` to every episode |
| G3 | No dataset-level validation CLI | High | `validate_release.py`: schema conformance, leakage audit, dedup, contamination, split disjointness |
| G4 | Retriever untested on non-Hotpot context shapes (MuSiQue paragraph lists, 2Wiki evidence triples, StrategyQA) | High | Converter + retrieval parity tests per dataset |
| G5 | No canonical config registry — configs live in ad-hoc YAML/shell | Medium | `resource_paper/configs/*.yaml`, one per matrix cell, hashed into the episode record |
| G6 | Teacher license/redistribution not verified (MiniMax-M3) | Medium | License audit before including any teacher's outputs |
| G7 | FAU gateway throttles above ~24-way concurrency (caused the circuit-breaker incident) | Medium | Global rate limiter + already-fixed recoverable breaker; stagger run-groups |
| G8 | Pilot traces (31.6K) predate several fixes (student-name bug, budget-as-ceiling prompt) | Medium | Treat as **v0 pilot**, regenerate or exclude from v1 release |

---

## 3. Source datasets

**Selection criteria:** permissive license, genuine multi-hop, distractor contexts
(so the local retriever applies), and *diversity of reasoning and answer type*.

### Tier 1 — generation set (traces produced)

| Dataset | License | Size (train) | Hops | Reasoning types | Answer type | Why included |
|---|---|---|---|---|---|---|
| **HotpotQA** (distractor) | CC BY-SA 4.0 | 90K | 2 | bridge, comparison | span | Baseline; already integrated; comparable to prior work |
| **2WikiMultihopQA** | Apache 2.0 | 167K | 2–4 | compositional, inference, comparison, bridge-comparison | span | Deeper hops; structured evidence triples give a *second* gold signal |
| **MuSiQue-Ans** | CC BY 4.0 | 20K | 2–4 | composed single-hops, explicitly shortcut-resistant | span | Hardest; disallows the "answer by shortcut" failure mode HotpotQA permits |
| **StrategyQA** | MIT | 2.7K | implicit | implicit decomposition, strategy | **boolean** | Different answer form; tests whether guidance transfers past span extraction |

### Tier 2 — held-out zero-shot evaluation only (no traces generated)

| Dataset | Purpose |
|---|---|
| **Bamboogle** (125 q) | Compositional, deliberately out-of-distribution; clean generalization test |
| **FanOutQA** | Breadth ("fan-out") multi-hop rather than depth |

Holding these out entirely lets us claim *zero-shot cross-dataset generalization*, which is
much stronger than in-distribution dev performance.

**Contamination caveat to state openly:** HotpotQA and 2Wiki are almost certainly in the
pretraining data of every model used. MuSiQue and Bamboogle are the more trustworthy
generalization signals. We report per-dataset results separately and never average across
them without saying so.

---

## 4. The generation matrix

Full factorial is impossible (4 datasets × 4 teachers × 5 students × 6 guidance × 3 budgets
× 2 disclose ≈ 2,880 cells). The design below is a **fractional design with one dense
counterfactual core**, chosen so each research question has a slice that answers it
*cleanly* while the bulk of the data remains usable as training corpus.

### Slice A — **TG-Core** (the scientific backbone) · ~72,000 episodes

The centerpiece. Paired counterfactuals: identical question, student and seed; **only the
supervision level varies**.

- Questions: **1,000 per dataset × 4 datasets = 4,000**
- Students: 3 (Qwen3.5-0.8B, Qwen3.5-2B, Granite-4.1-3B)
- Teacher: fixed (gpt-oss-120b) — held constant so supervision level is the sole variable
- Conditions: **6** — `skip_teacher`, g0, g1, g2, g3, g4
- Budget 5, plan review on (1 round), budget hidden
- `4,000 × 3 × 6 = 72,000 episodes`

**Enables:** dose–response curves; per-question paired significance tests; "what is the
minimum information the teacher must reveal?"; value-of-supervision by question difficulty.

### Slice B — **TG-Teachers** · ~16,000 episodes

- Questions: 500/dataset × 4 = 2,000 · Students: 2 (0.8B, 3B) · Guidance: g3 fixed
- Teachers: **4**, spanning size and family (see §5)
- `2,000 × 2 × 4 = 16,000`

**Enables:** does teacher identity matter? Is a mid-size open teacher enough? Teacher-size
scaling of guidance quality.

### Slice C — **TG-Protocol** · ~24,000 episodes

- Questions: 500/dataset × 4 = 2,000 · Students: 2 · Guidance g3
- Budget ∈ {3, 5, 9} × disclose_budget ∈ {true, false}
- `2,000 × 2 × 3 × 2 = 24,000`

**Enables:** interaction of step budget with guidance; does telling the agent its budget
change behaviour (we already see budget-as-quota effects); efficiency/accuracy frontier.

### Slice D — **TG-Scale** (bulk training corpus) · ~60,000 episodes

- Questions: **5,000 per dataset × 4 = 20,000** (disjoint from A/B/C where possible)
- Students: 3 · Teacher: 1 · Config: the workhorse (g3, b5, plan review on, hidden budget)
- `20,000 × 3 = 60,000`

**Enables:** the actual SFT/RL training data at scale; per-dataset and pooled training.

### Slice E — **TG-Seeds** · ~4,500 episodes

- 500 questions × 3 students × 3 seeds, g3 — for run-to-run variance estimates.

### Slice F — **TG-Ablate** (optional, ~8,000)

Plan-review off / teacher-as-planner / agent-wiki on — protocol variants we already support.

### Totals

| | Episodes | Unique questions |
|---|---|---|
| A · Core (counterfactual) | 72,000 | 4,000 |
| B · Teachers | 16,000 | 2,000 |
| C · Protocol | 24,000 | 2,000 |
| D · Scale (training) | 60,000 | 20,000 |
| E · Seeds | 4,500 | 500 |
| F · Ablations (optional) | 8,000 | 1,000 |
| **Total** | **~184,500** | **~26,000** |

At the measured 18–31 episodes/min this is **100–170 GPU-hours of generation** — roughly
**5–9 days** of wall-clock on 4 GPUs, or 3–5 days running two staggered groups on 8 GPUs.
Generation is teacher-API-bound, not GPU-bound.

---

## 5. Teachers and students

**Teachers — open-weight only** (redistribution rights; verify each before use):

| Role | Model | License | Serving |
|---|---|---|---|
| Primary | gpt-oss-120b | Apache 2.0 | FAU (free) |
| Secondary | MiniMax-M3 | *verify* | FAU / EdenAI |
| Large peer | Qwen3-235B-class or DeepSeek-V3-class | Apache/MIT | FAU if catalogued |
| **Small teacher** | ~8B open model | Apache 2.0 | local vLLM |

The **small teacher is scientifically important**: it tests whether guidance requires a
frontier teacher or whether a cheap local model suffices — a result practitioners care about
and a cost argument for the whole method.

**Students — 0.8B to ~8B, ≥3 families:** Qwen3.5-0.8B, Qwen3.5-2B, Granite-4.1-3B,
plus one ~8B model and optionally MiniCPM for family diversity. All served by local vLLM
under their **real HF ids** (the `vllm/student` placeholder bug is already fixed).

---

## 6. Quality control — the "super clean" requirement

This is where resource papers are won or lost. Every item below is either already
implemented or is a listed engineering task.

1. **Schema validation.** Every episode validated against a published JSON Schema;
   `schema_version` + `framework_commit` + `config_hash` on every record. Release blocked on
   100% conformance.
2. **Leakage audit.** `leakage.py` already records a per-step `leakage_check`. We publish
   the aggregate leak rate, and additionally run an **independent post-hoc auditor** (string
   + fuzzy + LLM-based) over all student-visible text. Target: report the true rate with a
   CI, not a claim of zero.
3. **Train-safe vs full views.** `teacher_private_diagnosis` contains gold-derived content.
   The release ships two views: `full` (research; includes privileged fields) and
   `train_safe` (privileged fields stripped; safe to fine-tune on naively). **This
   distinction must be loud in the datasheet** — it is a foreseeable misuse otherwise.
4. **Deduplication.** Exact + near-duplicate (`sft_diversity` Jaccard shingles) within and
   across slices; tool-signature capping available as a documented filter, not applied
   destructively.
5. **Contamination / split hygiene.** Question-level disjointness between train/dev/test
   enforced by qid hash; the union of all evaluation qids banned from all training views
   (the discipline we already used in the cross-student experiment); held-out datasets never
   generated from.
6. **Answer-correctness signals, triangulated.** Every episode carries (a) deterministic
   EM/F1/cover-match, (b) groundedness in retrieved evidence, (c) an LLM judge verdict.
   Disagreement rates between them are published — a feature, not an embarrassment.
7. **Path-quality signals.** `optimality.py` fields on every episode (wasted steps, repeats,
   failed tools, efficiency) so downstream users can filter for clean paths.
8. **Failure traces retained.** Incorrect episodes are **kept and labelled**, not discarded —
   they are essential for RL, PRM training, and error analysis. (Our own cross-student test
   sets were built from 100 correct + 100 wrong precisely because of this.)
9. **Reproducibility.** Pinned configs, seeds, model revisions/digests, and the exact router
   chain per episode. Self-hosted FAU serving is more reproducible than a commercial API and
   should be stated as such.

---

## 7. Human validation protocol (required for A\*)

- **Sample:** 1,200 episodes, stratified by dataset × guidance level × student × correctness.
- **Annotators:** 3, with **300 overlapping** items for agreement.
- **Labels per item:**
  1. Guidance *helpfulness* (1–5)
  2. Guidance *factual correctness* (correct / partially / wrong)
  3. **Leakage present?** (binary) — validates the automated gate
  4. Final answer correct? — validates the **LLM judge**
  5. Free-text failure category (for the error taxonomy)
- **Reported:** Krippendorff's α (or Cohen's κ), judge-vs-human agreement and κ, gate
  precision/recall for leakage with CIs, per-level helpfulness distribution.
- **Payment/ethics:** documented rates, no PII, public-corpus text only.

Deliverable: the annotation guidelines are released with the dataset.

---

## 8. Baseline experiments (the "utility" section)

A resource paper must prove the resource is useful. We already have two of these results.

| # | Experiment | Status | Claim it supports |
|---|---|---|---|
| E1 | **SFT dose–response**: train the same student on TG-Core data from each guidance level; plot end-to-end accuracy vs level | New | *How much supervision is worth collecting?* — the paper's headline curve |
| E2 | **Base → guidance-SFT lift** | ✅ done (7% → 53%) | The corpus is train-ready |
| E3 | **Cross-dataset zero-shot** (train Hotpot+2Wiki → test MuSiQue, Bamboogle, FanOutQA) | New | Generalization, not memorization |
| E4 | **Cross-student transfer** (self vs other-student traces) | ✅ done | Trace provenance matters less than trace quality; efficiency/precision transfer |
| E5 | **Teacher-size scaling** (TG-Teachers) | New | Is a cheap teacher enough? |
| E6 | **PRM from step scores** → best-of-n reranking | New | The *step-level* labels have standalone value |
| E7 | **Teacher-in-the-loop at inference vs internalized** | Partially done | Externalized vs internalized supervision |
| E8 | **Budget/efficiency frontier** (TG-Protocol) | New | Cost-aware agent design |

E1, E3, E5, E6 are the new ones and are the strongest reviewer-facing results.

---

## 9. Release artifacts

1. **HuggingFace dataset** — one config per slice (`core`, `teachers`, `protocol`, `scale`,
   `seeds`), each with `full` and `train_safe` views; parquet + JSONL.
2. **Croissant metadata** (increasingly expected by NeurIPS D&B).
3. **Datasheet for Datasets** (Gebru et al.) — motivation, composition, collection process,
   preprocessing, uses, distribution, maintenance, **and the misuse warning on privileged
   fields**.
4. **Framework code** — permissive license, the 396-test suite, one-command reproduction per
   matrix cell.
5. **Trained baselines** — the SFT checkpoints + recipes.
6. **Evaluation harness** — the held-out benchmark (Bamboogle/FanOutQA/MuSiQue-test) with a
   fixed protocol so numbers are comparable.
7. **Interactive trace explorer** — we already have a viewer (`teacher_guidance/viewer/`);
   polishing it into a public demo materially helps reviewers *see* the data.

---

## 10. Engineering work plan

### Phase 0 — Hardening (blocking; ~3–4 days)
- [ ] **G1**: parameterize `dataset` end-to-end (mode_config → metadata → exporter)
- [ ] **G2**: add `schema_version`, `framework_commit`, `config_hash` to every episode
- [ ] **G5**: config registry `resource_paper/configs/`, one YAML per matrix cell
- [ ] **G7**: global teacher rate limiter; verify the recoverable circuit breaker under load
- [ ] Publish the JSON Schema; add a schema-conformance test to CI

### Phase 1 — Dataset converters (~1.5 weeks)
- [ ] `converters/twowiki.py`, `converters/musique.py`, `converters/strategyqa.py`
- [ ] Per-dataset retrieval parity tests (gold docs must be reachable; BM25 sane)
- [ ] Golden-file unit tests per converter (dependency-free, like `hotpot_converter`)
- [ ] Per-dataset difficulty/hop-count profiling report

### Phase 2 — Model onboarding (~4 days)
- [ ] Teacher license audit; FAU catalogue check for the large peer
- [ ] Local vLLM serving for the small teacher and the ~8B student
- [ ] Capacity test: concurrency ceiling per teacher, measured not guessed

### Phase 3 — Pilot (~4 days)
- [ ] 500 episodes per **new** dataset × 2 students × {g0, g3, skip}
- [ ] Validate: leak rate, judge agreement, parse-failure rate, tool-error rate, cost/episode
- [ ] **Go/no-go gate** per dataset before mass generation

### Phase 4 — Full generation (~2 weeks wall-clock, mostly unattended)
- [ ] Slices D → A → C → B → E (D first: training data is the long pole for E1–E6)
- [ ] Nightly: status tracker, cost tracker, failure triage
- [ ] Resumability already proven — re-running a slice continues where it stopped

### Phase 5 — Curation & packaging (~1 week)
- [ ] `validate_release.py` full pass; dedup; contamination checks
- [ ] Build `full` / `train_safe` views; splits; HF upload; croissant; datasheet

### Phase 6 — Baselines (~2 weeks, overlaps Phase 5)
- [ ] E1, E3, E5, E6 (E2/E4 already done, re-run on v1 data for consistency)

### Phase 7 — Human validation (~1.5 weeks, parallel)
- [ ] Guidelines, annotator onboarding, 1,200-item study, agreement analysis

### Phase 8 — Writing (~3 weeks, overlapping)
- [ ] Paper, appendix, datasheet, demo, artifact submission

**Critical path:** Phase 0 → 1 → 3 → 4 → 6 → 8. **Total ≈ 12–14 weeks** with overlap.

---

## 11. Paper outline

1. **Introduction** — small agents fail at multi-hop; supervision helps; nobody has measured
   *how much* supervision is needed because no corpus varies it.
2. **Related work** — agent trajectory corpora; process supervision; knowledge distillation;
   LUPI. Position on the intersection.
3. **The guidance protocol** — teacher/student loop, plan review, the 0–4 exposure ladder,
   code-enforced rendering, leakage gating. (Figure: one annotated episode.)
4. **Corpus construction** — datasets, models, the matrix, quality pipeline, human validation.
5. **Corpus analysis** — scale, hop/type distributions, guidance-level effects on *behaviour*
   (steps, tool mix, stop reason), leak rates, judge/human agreement, cost telemetry.
6. **Experiments** — E1–E8.
7. **Limitations** — contamination, English-only, retrieval is local-distractor not open web,
   teacher monoculture risk, LLM-judge reliability.
8. **Ethics & licensing** — open-weight provenance, privileged-field misuse, annotator terms.

**Target venue.** **NeurIPS Datasets & Benchmarks** (primary — best fit for a corpus +
benchmark + baselines). Alternates: **COLM** (strong LLM-resource fit), **ACL/EMNLP** via ARR.
LREC-COLING as a floor. *Deadline drives the timeline — this is the first decision needed.*

---

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| "Just another distillation dataset" reviewer framing | Med | **High** | Lead with the *counterfactual ladder* and paired design, not scale; E1's dose–response curve is the identity of the paper |
| Contamination undermines results | High | Med | MuSiQue/Bamboogle as the trusted signals; report per-dataset; never average silently |
| LLM judge unreliable | Med | High | Human validation quantifies it; triangulate with deterministic metrics |
| Teacher API drift/throttling | Med | Med | Self-hosted FAU first; pin revisions; rate limiter; resumable runners (proven) |
| Privileged fields misused for training | Med | High | `train_safe` default view; loud datasheet warning; leakage audit published |
| MiniMax-M3 (or peer) license forbids redistribution | Med | Med | License audit in Phase 2; drop the teacher rather than compromise the "fully open" claim |
| Generation slips | Med | Med | Slice D first (training data unblocks all baselines); slices are independently publishable |
| StrategyQA/MuSiQue don't fit the retriever cleanly | Med | Low | Phase 3 go/no-go per dataset; 3 datasets is still a strong paper |

---

## 13. Decisions needed before Phase 0

1. **Target venue + deadline** — sets the whole schedule. Recommendation: NeurIPS D&B.
2. **Scale target** — 184K episodes as planned, or trim (drop Slice F, halve Slice D) for speed?
3. **Dataset set** — confirm HotpotQA + 2Wiki + MuSiQue + StrategyQA; StrategyQA is the
   riskiest to convert (boolean answers, different evidence structure).
4. **Teacher roster** — which 4? Depends on the FAU catalogue and license audit.
5. **Budget ceiling** — FAU-first keeps cost near zero; EdenAI overflow at ~$0.003/episode
   implies **~$550 if 100% paid**, realistically **$150–400**. Confirm an acceptable cap.
6. **Human annotation resources** — who annotates, and is there budget/IRB-equivalent?
7. **v0 pilot traces** — regenerate the 31.6K existing HotpotQA episodes under v1 configs
   (clean, consistent) or ship them as a separate documented v0? Recommendation: regenerate
   the useful configs; they predate several fixes.
