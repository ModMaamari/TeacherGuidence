# m1 from teacher-only traces — plan, context, and lessons

**Goal.** Produce 3,000 *expert* agent trajectories in which **gpt-oss-120b is the agent
itself** (it plans and answers, with a hidden 3-planning / 3-running-step budget), keep
the correct ones, train the three student models on them **with the same m1 method** (LoRA
SFT — the teacher's own thought+action becomes the student's internal thought), evaluate
with the existing four-arm protocol, and compare this **teacher-only data recipe** against
the **guidance data recipe** we have already run.

Same method (m1), same eval arm (m1), same four-arm protocol — **only the training-trace
source changes.** This document is written so someone who did **not** run the earlier
experiments can execute this one end-to-end and avoid every trap we hit. Read "Lessons &
gotchas" (§11) before you touch anything.

---

## 0. What is being compared — two data recipes for the *same* m1 model

"m1" is our SFT method: the student is fine-tuned so that **teacher-derived reasoning
becomes its own internal thought**, and at eval the trained student is the **`m1` arm** of
the four-arm protocol (`base`, `base + teacher`, `m1`, `m1 + teacher`). What changes
between the two experiments is only **which traces train that m1 model**:

| | **m1 from GUIDANCE traces (DONE)** | **m1 from TEACHER-ONLY traces (THIS PLAN)** |
|---|---|---|
| Who acts in the trace | weak student proposes every action | **gpt-oss-120b is the agent** — it plans and acts |
| Teacher in the loop | yes — reviews the plan, critiques every step | none (the 120b *is* the expert) |
| The student's internal thought comes from | the teacher's second-person **guidance** on the student's own attempt | the teacher's **own thought+action** on each step |
| Trace source | `traces_g3b_b*_3000/` (6 budget runs, 18k episodes) | `traces_oss120b_teacheronly_3000/` (1 run, ≤3k correct) |
| Target format | `{teacher_guidance{…}, thought, decision, action, …}` (guidance first) | `{thought, decision, action, new_facts_extracted}` (the teacher's own) |
| Gold-leakage risk | **high** — teacher sees gold; needed `[answer hidden]` redaction | **none** — the 120b-as-student never sees gold (§6) |
| SFT trainer | `training_methods/m1_sft/train.py` | **same** `training_methods/m1_sft/train.py` |
| Eval | four-arm unseen-100, arm = `m1` | **same** four-arm unseen-100, arm = `m1` |

**The scientific question:** to teach a small agent, is it better to distil *the teacher's
critiques of the student's own attempts* (guidance recipe) or *the teacher's own expert
demonstrations* (teacher-only recipe)? Both feed the identical m1 SFT method and are
measured with the identical four-arm eval, so the comparison is clean **except** for the
confounds in §10 (data volume, budget, no error-recovery demonstrations) — state those in
the report. Within a single student model, `m1(guidance)` vs `m1(teacher-only)` is a clean
paired test; the arm name stays `m1` in both.

---

## 1. What this project is (orientation)

A "teacher-guidance" agent solves multi-hop HotpotQA questions with a small tool loop:

```
initial plan  →  [step: pick a tool (search / extract / verify / synthesize / finish)]×budget  →  answer
```

Tools run against a **per-question local corpus** (the HotpotQA distractor set for that
question — retrieval is self-contained and reproducible, no web). Every question row carries
`retrieval_scope = {backend: "hotpot_local", qid, candidate_doc_ids}`.

The harness (`agentsim/teacher_guidance/`) renders the student-visible prompt, executes the
chosen tool deterministically, and scores each episode with `exact_match`, `f1_score`,
`cover_match`, `supporting_doc_recall`. A **teacher** (gpt-oss-120b via an academic gateway)
can optionally review the plan and grade each step — that is the guidance we distilled in
the first m1 experiment. This plan removes the teacher and lets the 120b be the agent.

**Key components you will reuse:**

| path | what it is |
|---|---|
| `agentsim/teacher_guidance/prompts.py` | `build_initial_plan_prompt`, `build_student_prompt`, `build_student_visible_state` — the exact prompt renderers. Reuse verbatim so traces match eval. |
| `agentsim/teacher_guidance/tool_executor.py` | `execute_student_tool`, `derive_final_answer` — deterministic tool execution. |
| `agentsim/teacher_guidance/json_utils.py` | `parse_student_action`, `parse_student_plan`. |
| `agentsim/teacher_guidance/metrics.py` | EM / F1 / cover / doc-recall. |
| `agentsim/teacher_guidance/sft_export.py` | `DEFAULT_SYSTEM` (system prompt in every training example). |
| `agentsim/components/control/teacher_guided_plan_review.py`, `…/teacher_guided_agent_step.py` | the plan + step components; honor `skip_teacher`. |
| `training_methods/common/teacher_eval_agent.py` | already drives those components through a client — **the template for the trace runner** (§4.3). |
| `scripts/gen_fau_smoke_template.py` | `build_fau_smoke_template` + `TEACHER_ROUTER` (FAU→OpenRouter fallback). |
| `training_methods/m1_sft/train.py` | the m1 LoRA SFT trainer (model-agnostic; used for all three students). |
| `training_methods/m2_rft/filter_rollouts.py` | `episode_to_examples` — trajectory→m1-format rows **without a guidance block** (the exact builder you need). |
| `training_methods/common/` | the eval + vLLM + judge + analysis + report stack (§8–9). |

---

## 2. Data assets (do not regenerate these)

- **The 3,000 questions** (use exactly these — same as every prior run):
  `data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl`
  + corpus `…/hotpot_distractor_train_corpus.jsonl`. Prior runs used **the first 3,000
  lines in file order** (`head -n 3000`). Match that.
- **Unseen-100 eval set** (never overlaps the 3,000):
  `training_methods/exp_unseen100/data/unseen100/` (built with
  `m2_rft/prepare_fresh_questions.py --limit 100 --seed 101`). Reuse as-is.
- **Existing guidance traces** (the other recipe, reference only):
  `data/simulation_output/traces_g3b_b5_3000/` and siblings.
- **`.env`** holds `FAU_LLM_API_KEY` / `CUSTOM_LLM_API_KEY`. **Never commit it.** Confirm
  keys are live (`config.provider_available(...)`) before a long run.

---

## 3. Models & environments

| | model id | served how |
|---|---|---|
| Teacher/expert (this run) | `fau/gpt-oss-120b` (router: FAU → `custom/openai/gpt-oss-120b:free` → `custom/openai/gpt-oss-120b`) | API only, no local GPU |
| Student — granite | `ibm-granite/granite-4.1-3b` | train: HF; eval: vLLM (LoRA ok) |
| Student — Qwen 0.8B | `Qwen/Qwen3.5-0.8B` | train: HF; eval: vLLM (**merged weights**, G-5) |
| Student — Qwen 2B | `Qwen/Qwen3.5-2B` | train: HF; eval: vLLM (**merged weights**, G-5) |

- **`.venv_train`** — training + HF eval. **Qwen students need `flash-linear-attention` +
  `causal-conv1d`** or training is 5× slower (G-7).
- **`.venv_vllm`** — isolated vLLM 0.25 for fast eval serving. Never mix its torch with
  `.venv_train`.

---

## 4. PART A — Generate the 3,000 teacher-only traces

### 4.1 What "teacher-only" means operationally

The 120b plays the **student role** (plans and acts) with **no teacher in the loop** —
`skip_teacher = True`, `student_model = fau/gpt-oss-120b`:

- **Plan turn:** the 120b writes one plan (skip_teacher → **0** review/revision rounds).
- **Each step:** the 120b picks a tool; no teacher grade; the loop stops on the 120b's own
  `finish` or when the running budget is exhausted (then `derive_final_answer`).
- The trace records the 120b's plan + its per-step `thought/decision/action` — **no
  `teacher_guidance` block, no gold, no leakage** (§6). That per-step `thought` is exactly
  what m1 trains the student to produce as its internal thought.

### 4.2 Budget: "3 planning, 3 running, hidden"

```
budget            = 3       # running / tool-use steps
max_plan_steps    = 3       # the plan may contain up to 3 planned steps
disclose_budget   = False   # HIDDEN — the agent is not told the step count
planning_steps    = 3       # (see decision note)
```

> **DECISION NOTE.** `planning_steps` is the number of *teacher-review → revise* rounds.
> Under `skip_teacher=True` there is no reviewer, so those rounds do not happen and
> `planning_steps` is effectively ignored — the 120b produces **one** plan. Two ways to
> honor "3 planning steps":
>
> 1. **(Recommended) Solo expert, one plan.** `skip_teacher=True`, `max_plan_steps=3`,
>    `budget=3`, hidden. The 120b plans once (≤3 planned steps) and executes ≤3 tool steps.
>    Cleanest expert trajectory, matches "teacher only" literally, fewest API calls.
> 2. **Self-refined plan (variant).** Set `student_model` **and** `teacher_model` to
>    `fau/gpt-oss-120b`, `skip_teacher=False`, `planning_steps=3`, `planner="student"` —
>    the 120b-as-teacher reviews the 120b-as-student's plan for 3 rounds. Genuine 3-round
>    refinement, but doubles plan-stage cost **and** exposes the reviewer to **gold** →
>    the revised plan can leak gold reasoning; you must re-apply the §6/G-4 leakage gate to
>    the plan text. Only use it if self-refinement is itself under study.
>
> Everything downstream is identical for both variants. **Use Variant 1 unless you have a
> specific reason not to.**

### 4.3 The trace runner — the change you must make

`scripts/run_batched_parallel.py` (the guidance-run collector) assumes a **local Ollama
student**: it starts `ollama serve` per GPU and `ollama pull`s the student tag. Your
student is an **API model (gpt-oss-120b)** — no Ollama, no GPU student. Do **not** use it
as-is.

**Write a small async API runner** `scripts/run_teacher_only_traces.py` — much simpler than
the Ollama path because it is pure API. `training_methods/common/teacher_eval_agent.py` is a
near-exact template: it already drives `TeacherGuidedPlanReview` + `TeacherGuidedAgentStep`.
For teacher-only, use an **all-API client** (student calls also go to the router) and
`skip_teacher=True`:

```python
# per question (bounded by an asyncio.Semaphore — see G-9 for the safe value):
metadata = dict(
    student_model="fau/gpt-oss-120b", teacher_model="fau/gpt-oss-120b",
    teacher_router=TEACHER_ROUTER, skip_teacher=True,
    budget=3, disclose_budget=False,
    plan_review_config=dict(enabled=True, planner="student", planning_steps=3,
                            max_initial_plan_steps=3, max_revised_plan_steps=3),
    corpus_path=<3000 corpus>, retrieval_backend="hotpot_local",
    gold=row.get("gold"), gold_answer=row["answer"], retrieval_scope=row["retrieval_scope"],
    teacher_max_tokens=2500, teacher_max_tokens_retry=4000,
    student_temperature=0.2,   # sampled expert; keep a seed per rep if you want variance
)
ctx = WorkflowContext(task_id=qid, query=row["query"], metadata=metadata)
await TeacherGuidedPlanReview(config={}, llm_client=client).execute(ctx)
for t in range(1, 4):
    if ctx.metadata.get("done"): break
    await TeacherGuidedAgentStep(config={"step_index": t, "budget": 3, "force_finish": t == 3},
                                 llm_client=client).execute(ctx)
# export: qid, query, gold_answer, final_answer, plan(_review), steps=ctx.metadata["teacher_guided_steps"],
#         stop_reason, used_steps, final_metrics{em,f1,cover,doc_recall}
```

- **`skip_teacher` is hardcoded `False`** in `gen_fau_smoke_template.py` (~line 74) and not a
  parameter. The async runner builds `mode_config`/metadata directly, so it sidesteps the
  template — preferred. (If you instead patch the template, add a `skip_teacher` param.)
- Stream one JSONL line per finished episode into
  `data/simulation_output/traces_oss120b_teacheronly_3000/`, matching the export shape
  (`qid, query, gold_answer, final_answer, plan, steps[], stop_reason, used_steps,
  final_metrics`); each `step` has `student_prompt, student_raw, student_action,
  action_valid, tool_observation`.

### 4.4 Cost, time, monitoring

- **No GPUs** — pure API. ~4–6 sequential 120b reasoning calls per episode (1 plan + ≤3
  steps, 10–40 s each) at a safe concurrency of ~8–16 → **~2–4 h** for 3,000, plus retries.
- **Write the sample-id manifest before launch** and a **live results/stats file** appended
  per episode + a `missing_ids.txt` diff, so you can re-run only the gaps (copy
  `scripts/watch_trace_collection.py`). Long API runs will have transient failures — G-11.
- **Hard per-call timeouts** (`FAU_TIMEOUT`, `CUSTOM_TIMEOUT`) so a hung call raises — but
  mind the circuit-breaker trap in **G-9**.

---

## 5. PART B — Select the "correct" samples

Train only on episodes where the expert got it right (a wrong expert trajectory is a bad
demonstration). Two signals — use both, prefer the judge:

1. **Lexical (free, offline):** `final_metrics.cover_match == True`. Fast first pass.
2. **Semantic (recommended):** `training_methods/common/judge_final_answers.py` — one
   gpt-oss-120b verdict per final answer → `{correct, score}`; keep `verdict.correct == 1`.
   This matches the exact metric the eval uses, so the training filter and eval metric agree.

Expect a large-majority yield (the 120b is strong). If yield is low, look for **tool-format
failures** (the 120b over-explains and never emits a clean `finish` — G-8), not reasoning
failures. Write kept qids + episodes to `…/selected_correct.jsonl` and log the yield.

---

## 6. Leakage — the good news

In the guidance recipe we fought the `[answer hidden]` placeholder because the **teacher**
sees gold and could echo it into the student's target. **In teacher-only Variant 1 there is
no teacher and the 120b-as-student never sees gold** — it only ever receives student-visible
state. So:

- **No `[answer hidden]` handling needed** for Variant 1.
- **Still run a belt-and-suspenders gate** on every training target: reject any example
  whose target text contains the gold answer verbatim **unless** the gold already appears in
  the question or the retrieved observations (`leak_gate_ok` in
  `training_methods/common/episode_lib.py`). Catches parametric-knowledge guesses you do not
  want to clone.
- **Variant 2 (self-review) reintroduces leakage** via the plan reviewer — gate the plan.

---

## 7. PART C — Build the m1 training data from teacher-only traces

The m1 dataset is `{"prompt": [system, user], "completion": [assistant JSON]}` with the
student's internal thought+action as the completion. For teacher-only traces the completion
is **the teacher's own thought+action** (no separate `teacher_guidance` block — the whole
trajectory is the teacher's reasoning). So **do NOT use `m1_sft/build_dataset.py`** (it
assembles a guidance block); use the trajectory builder in
`training_methods/m2_rft/filter_rollouts.py :: episode_to_examples`, which already emits the
m1 format from parsed plan/action turns:

- **Plan example:** `prompt=[DEFAULT_SYSTEM, plan_prompt]`, `completion=the 120b's plan JSON`.
- **Per step:** `prompt=[DEFAULT_SYSTEM, student_prompt]`, `completion=the 120b's action JSON`.
- Keep only structurally valid turns (`action_valid` + `parse_student_action` re-validates).

Create `training_methods/exp_teacher_only/build_dataset.py` (copy that function; add the §6
leakage gate and a qid-hash dev split mirroring `episode_lib.qid_split`, 3% dev). Output
`train.jsonl` / `dev.jsonl` / `dev_questions.jsonl` / `stats.json` — the same files the m1
trainer consumes.

**Gotchas specific to this build:**

- **`decision.category == ""`** — the harness stores an omitted category as `""`, but the
  action schema **rejects** `""` (accepts a *missing* category). Cloning the 120b's action
  dict verbatim teaches an invalid value that **zeroes every action at eval**
  (`invalid_category`) — this exact bug cost us a whole run (G-2). **Drop the key when its
  value is `""`.**
- **Budget-in-prompt mismatch.** These prompts say "step *t* of **3**"; eval runs at budget
  **4**. Either (a) re-render every training prompt at the eval budget with
  `build_student_prompt` (the state is stored, so re-rendering is exact — recommended), (b)
  eval at budget 3 to match, or (c) accept and document. Pick one and note it.
- **Data volume vs the guidance recipe.** ≤3k episodes here vs 18k for guidance-m1 — a real
  confound (§10), not a bug. Optionally also train a ~3k-episode matched slice of the
  guidance data to control for volume.

---

## 8. PART D — Train the three students with the m1 method

Reuse `training_methods/m1_sft/train.py` unchanged. Per model:

```bash
CUDA_VISIBLE_DEVICES=<4 free gpus> .venv_train/bin/torchrun --nproc_per_node=4 \
  --master_port=297xx training_methods/m1_sft/train.py \
  --model <ibm-granite/granite-4.1-3b | Qwen/Qwen3.5-0.8B | Qwen/Qwen3.5-2B> \
  --train-file training_methods/exp_teacher_only/data/train.jsonl \
  --dev-file   training_methods/exp_teacher_only/data/dev.jsonl \
  --epochs 2 --batch-size 4 --grad-accum 1 \
  --run-dir training_methods/exp_teacher_only/runs/<ts>_<model>
# then: training_methods/common/plot_training_curves.py --run-dir <that dir>
```

**Must-dos (each bit us):**

- **Qwen fast kernels first** (G-7) — verify the log does **not** say "fast path is not
  available," or Qwen trains 5× slower.
- **DDP artifact races are fixed** in `train.py` (rank-0 guards on the pre-training
  `train_config.json` **and** the post-training writes). If you fork the trainer, keep every
  shared-file write behind `trainer.is_world_process_zero()` (G-1).
- **Sanity:** eval_loss ~0.34–0.45, token accuracy ~86–89% was the band for the three
  guidance-m1 students; flag anything wildly off.

---

## 9. PART E — Evaluate (four arms) + per-model report

Reuse the `exp_unseen100` stack unchanged — only point it at the new adapters. Arms are the
standard four; the trained model is the **`m1`** arm (now m1-from-teacher-only):

1. **Merge weights for the Qwen students** — vLLM cannot LoRA Qwen3.5 (loads the adapter and
   silently ignores it — G-5). Merge into the base and splice into the full composite
   checkpoint. Granite serves its LoRA fine. Verify with an identical-outputs probe.
2. **Run the experiment:**
   ```bash
   .venv_train/bin/python training_methods/exp_unseen100/run_experiment.py \
     --backend vllm --gpus <5 free> --clients-per-server 4 \
     --model <base id> [--merged-model <merged dir>] [--adapter <granite adapter>] \
     --exp-tag exp_teacheronly_<model>
   ```
   20 jobs (4 arms × 5 seeds 11/23/37/53/71), auto-judges all 2,000 answers, writes
   `analysis/`. Servers self-stagger (G-6). **If any `*_teacher_s*` run has < 100 episodes**
   (G-9), refill just that seed with a dedicated single server at low `--teacher-concurrency`
   and re-judge — never report a partial-count seed.
3. **Per-model report:** `training_methods/exp_unseen100/analyze_experiment.py --exp-dir
   <run>` → `analysis/report.html` (box plots, paired t + Wilcoxon + Welch, Holm, cost).

---

## 10. PART F — The comparison: m1(teacher-only) vs m1(guidance)

The headline result is **the same m1 model trained two ways**, per student. Two ways to
produce it:

- **Quickest:** feed both experiments' dirs to
  `training_methods/exp_unseen100/compare_models.py` with labels like
  `"Qwen3.5-2B · m1(guidance)"=<guidance dir>` and
  `"Qwen3.5-2B · m1(teacher-only)"=<teacher-only dir>`. The accuracy-vs-cost frontier and the
  per-model significance table then show which recipe wins.
- **Cleanest:** a per-student paired comparison of the two `m1` arms on teacher-verdict
  correctness (the primary metric), plus the `m1 + teacher` variants.

**State these confounds explicitly** — they separate a credible conclusion from a misleading
one:

1. **Data volume:** guidance-m1 ≈ 18k episodes (6 budgets × 3k); teacher-only-m1 ≤ 3k.
2. **Budget:** guidance traces spanned budgets 1–9; teacher-only traces are budget-3; eval is
   budget-4. Note how you handled the prompt-budget wording (§7).
3. **Error-recovery:** guidance data shows the student *its own mistakes and the fix*;
   teacher-only data shows only clean expert successes — the student never learns to recover
   from a bad step. This is the central qualitative difference and likely the most
   interesting result.
4. **Model family:** granite-3B is a different family than the Qwen students; size and
   architecture are confounded *across* models but **not** in the within-model
   m1(guidance)↔m1(teacher-only) pairing, which is the clean test.

---

## 11. Lessons & gotchas (read before you start)

- **G-1 — DDP artifact-write races.** Under `torchrun` every rank runs the script; any
  `write_json` with an atomic `.tmp`→final rename **races across ranks** and crashes the job.
  Guard **every** shared-file write with `trainer.is_world_process_zero()`. We hit this three
  times (final metrics, trainer state, pre-training `train_config.json`). Fixed in `train.py`.
- **G-2 — `decision.category == ""` zeroes eval.** The schema accepts a *missing* category
  but rejects `""`. Targets that copy the stored dict verbatim make **every action fail
  validation at eval** (`invalid_category`) — silent and total. Strip the empty category on
  the write side (dataset build) and the read side (`_safe_parse_action` already does).
- **G-3 — Inspect stored fields before cloning them.** (Guidance-specific instance: a
  default `score: 0.0` on positive plan feedback trained a "score 0 + praise" pattern.)
  General lesson for this run too: understand what each stored field means before it becomes
  a target.
- **G-4 — Leakage.** Guidance data needed heavy `[answer hidden]` redaction. Teacher-only
  Variant 1 is leak-free by construction, but **still run `leak_gate_ok`** (§6) to catch
  parametric-knowledge guesses. Variant 2 reintroduces leakage via the plan reviewer.
- **G-5 — vLLM can't LoRA Qwen3.5.** Qwen3.5 loads as `Qwen3_5ForConditionalGeneration`; a
  PEFT adapter's module names don't match, so vLLM **loads the adapter and silently ignores
  it** (base and adapter outputs identical). Detect with an identical-outputs probe. **Fix:
  merge into the base and serve the merged model** — splice the trained language-model
  tensors into the **full composite checkpoint** (keep the mtp head + original
  `config.json`) or vLLM rejects the config type. Granite's LoRA serves fine.
- **G-6 — Two vLLM servers per GPU race at startup.** The merged-weights eval starts a base
  server + a merged-twin on the same GPU; two engines profiling free memory at once collide
  ("connection refused"). Fixed with a 20 s stagger in `run_experiment.py`. Stagger any
  hand-launched pair.
- **G-7 — Qwen training fast path.** Without `flash-linear-attention` **and** `causal-conv1d`,
  Qwen3.5 falls back to a torch impl (~5× slower). Install `fla` (pip), then **build
  `causal-conv1d` locally**: `pip install wheel setuptools` first (the `--no-build-isolation`
  build needs `wheel`), then `CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install
  causal-conv1d --no-build-isolation --no-cache-dir` (the prebuilt wheel links a newer glibc
  than the box has — force an nvcc build). Put the venv's `ninja` on PATH for JIT kernels.
- **G-8 — The 120b over-explains and won't `finish`.** As a reasoning model it sometimes
  narrates instead of emitting a clean `finish`, burning budget → "unknown" or a verbose
  non-EM answer. Give it a generous completion budget (`teacher_max_tokens≈2500`, retry
  4000) and rely on the forced-finish fallback. Measure "correct" with the semantic judge,
  not EM (verbose-but-right passes cover/judge, fails EM).
- **G-9 — Gateway circuit breaker permanently disables a worker.** The router trips a
  provider after a hard timeout and **does not un-trip until the process restarts**. Under
  high concurrency a burst of timeouts can trip **all** providers for one worker; every
  remaining episode then fails instantly with "no configured provider" — we lost 71/100
  episodes in one seed this way. Mitigate: (a) keep teacher concurrency modest (≤ ~5 per
  process); (b) after a run, **check every teacher-arm run for a short episode count and
  refill short seeds** with a fresh low-concurrency process; (c) keep all three router
  providers configured.
- **G-10 — Local Ollama students (only if you use one).** granite4.1:3b's default context is
  huge; parallel slots blow past 80 GB KV cache and **Ollama silently degrades to CPU**
  (every call times out). Set `OLLAMA_CONTEXT_LENGTH=16384`. Not relevant to the API-only run.
- **G-11 — Long API runs need a manifest + live watcher.** Write expected qids before launch,
  append one line per finished episode, diff to `missing_ids.txt`, re-run only the gaps.
- **G-12 — transformers 5.x / TRL 1.8 quirks.** `apply_chat_template(..., return_tensors=
  "pt", return_dict=True)` then `model.generate(**enc)`. `KTOConfig`/`DPOConfig` have no
  `max_prompt_length`. `StudentActionModel` uses `extra="ignore"`.
- **G-13 — Batched/served generation isn't bit-identical to sequential HF.** vLLM and batched
  HF give cosmetically different greedy outputs. Compare arms **within one backend**; report
  **tokens** (hardware-independent) as the primary cost metric; GPU peak-mem isn't captured
  per-arm in vLLM serving mode.
- **G-14 — Never attribute commits to an assistant.** No "Co-authored-by" / "Generated by"
  lines in commits or PRs (standing project rule).

---

## 12. End-to-end checklist

```
[ ] Confirm .env has live FAU/CUSTOM keys (provider_available()).
[ ] Write scripts/run_teacher_only_traces.py (async all-API runner; copy teacher_eval_agent.py;
    skip_teacher=True, budget=3, max_plan_steps=3, hidden, planner=student). Manifest + watcher;
    hard timeouts; concurrency ≤ ~8–16 (G-9).
[ ] Collect 3,000 → data/simulation_output/traces_oss120b_teacheronly_3000/.
[ ] Filter to correct: cover_match, then judge_final_answers.py (keep verdict.correct==1).
[ ] build_dataset.py (m1-format via m2 episode_to_examples): strip decision.category=="" (G-2);
    leak gate (§6/G-4); re-render prompts at eval budget or note mismatch (§7); qid-hash dev split.
[ ] Qwen fast kernels installed & verified (G-7).
[ ] Train granite-3B / Qwen-0.8B / Qwen-2B with m1_sft/train.py (rank-0 guards intact G-1); plot.
[ ] Merge Qwen adapters into full checkpoints; identical-outputs probe (G-5). Granite: LoRA ok.
[ ] Four-arm eval per model (run_experiment.py, vLLM, staggered G-6); refill short teacher seed
    (G-9); 2000/2000 judged.
[ ] analyze_experiment.py per model → report.html.
[ ] compare_models.py: pair m1(teacher-only) vs m1(guidance) per student; state confounds (§10).
[ ] Publish reports; atomic commits; push. NO co-author lines (G-14).
```

---

## 13. Where the m1(guidance) results live (what you compare against)

- Guidance-m1 unseen-100 experiments: `training_methods/exp_unseen100/runs/…_exp` (granite),
  `…_exp_qwen05b`, `…_exp_qwen2b`; per-model reports in `report_qwen05b/`, `report_qwen2b/`;
  cross-model efficiency report in `report_compare/`.
- Guidance-m1 training runs & curves: `training_methods/m1_sft/runs/…_train4gpu` (granite),
  `…_qwen05b`, `…_qwen2b`.
- Headline guidance-m1 numbers (teacher-verdict correct, unseen-100, 5 seeds) —
  `base / base+teacher / m1 / m1+teacher`:
  granite-3B 30.8 / 63.6 / 48.6 / 59.5; Qwen-0.8B 3.0 / 32.6 / 60.2 / 69.6;
  Qwen-2B 21.2 / 60.2 / 63.6 / 74.6. **Your m1(teacher-only) arm is measured against the
  `m1` column here, per student.**
```
