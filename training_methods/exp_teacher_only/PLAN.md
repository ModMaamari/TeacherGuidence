# Teacher-only expert traces — plan, context, and lessons

**Goal.** Produce 3,000 *expert* agent trajectories in which **gpt-oss-120b is the agent
itself** (it plans and answers, with a hidden 3-planning / 3-running-step budget), keep
the correct ones, train the three student models on them, evaluate with the existing
four-arm protocol, and compare this **trajectory distillation** ("teacher steps only")
against the **guidance distillation** we have already done ("student steps guided by the
teacher").

This document is written so that someone who did **not** run the earlier experiments can
execute this one end-to-end and avoid every trap we hit. Read the "Lessons & gotchas"
section (bottom) before you touch anything — most of it is non-obvious and cost us hours.

---

## 0. The two approaches being compared

| | **Guidance distillation (DONE — "m1")** | **Trajectory distillation (THIS PLAN — "e1")** |
|---|---|---|
| Who acts | weak student (granite/Qwen) proposes every action | **gpt-oss-120b is the agent** — it plans and acts |
| Teacher role | reviews the student's plan and critiques every step | none in the loop (the 120b *is* the expert) |
| What the trace contains | student action **+ teacher's second-person guidance** | the 120b's own plan + actions (no guidance block) |
| Training signal | student learns to internalize the teacher's critiques of **its own** mistakes | student learns to **imitate the expert's** trajectory (behavioral cloning) |
| Data volume collected | 6 budget runs × 3,000 = 18k episodes | 1 run × 3,000 = ≤3k episodes (correct-only after filtering) |
| Gold leakage risk | **high** — teacher sees gold; we had to redact `[answer hidden]` | **none** — the 120b-as-student never sees gold (see §6) |

**The scientific question:** for a small agent, is it better to learn from *the teacher's
critiques of your own attempts* or from *the teacher's own expert demonstrations*? Both are
distilled from the same gpt-oss-120b teacher, evaluated on the same unseen questions with
the identical four-arm protocol, so the comparison is clean **except** for the confounds
listed in §10 (data volume, budget, no error-recovery demonstrations) — call them out in
the final report.

---

## 1. What this project is (orientation)

A "teacher-guidance" agent solves multi-hop HotpotQA questions with a small tool-using loop:

```
initial plan  →  [step: pick a tool (search / extract / verify / synthesize / finish)]×budget  →  answer
```

Tools run against a **per-question local corpus** (the HotpotQA distractor set for that
question, so retrieval is self-contained and reproducible — no web). Every question row
carries `retrieval_scope = {backend: "hotpot_local", qid, candidate_doc_ids}`.

The harness (in `agentsim/teacher_guidance/`) renders the student-visible prompt, executes
the chosen tool deterministically, and scores each episode with `exact_match`, `f1_score`,
`cover_match`, and `supporting_doc_recall`. A **teacher** (gpt-oss-120b via an academic
gateway) can optionally review the plan and grade each step, producing the guidance we
distilled in the "m1" line of work.

**Key components you will reuse:**

| path | what it is |
|---|---|
| `agentsim/teacher_guidance/prompts.py` | `build_initial_plan_prompt`, `build_student_prompt`, `build_student_visible_state` — the exact prompt renderers. **Reuse verbatim** so traces match eval. |
| `agentsim/teacher_guidance/tool_executor.py` | `execute_student_tool`, `derive_final_answer` — deterministic tool execution. |
| `agentsim/teacher_guidance/json_utils.py` | `parse_student_action`, `parse_student_plan`. |
| `agentsim/teacher_guidance/metrics.py` | EM / F1 / cover / doc-recall. |
| `agentsim/teacher_guidance/sft_export.py` | `DEFAULT_SYSTEM` (the system prompt used in every training example). |
| `scripts/run_batched_parallel.py` | the trace-collection runner (built for a **local Ollama student** — see §5 for the change needed). |
| `scripts/gen_fau_smoke_template.py` | `build_fau_smoke_template` + `TEACHER_ROUTER` (the FAU→OpenRouter fallback chain). |
| `training_methods/m1_sft/train.py` | the LoRA SFT trainer (model-agnostic; reused for all three students). |
| `training_methods/m2_rft/filter_rollouts.py` | `episode_to_examples` — **trajectory→training-rows without a guidance block** (this is the pattern you want, not the m1 guidance builder). |
| `training_methods/common/` | the whole eval + vLLM + judge + analysis + report stack (see §8–9). |

---

## 2. Data assets (do not regenerate these)

- **The 3,000 questions** (use exactly these — same as every prior run):
  `data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_questions.jsonl`
  and its corpus `…/hotpot_distractor_train_corpus.jsonl`.
  The prior runs used **the first 3,000 lines in file order** (`head -n 3000`). Match that.
- **Unseen-100 eval set** (never overlaps the 3,000): built by
  `training_methods/m2_rft/prepare_fresh_questions.py --limit 100 --seed 101`
  → `training_methods/exp_unseen100/data/unseen100/`. Reuse as-is.
- **Existing guidance traces** (the "m1" side of the comparison), for reference only:
  `data/simulation_output/traces_g3b_b5_3000/` (budget-5 hidden) and siblings b1–b4/b9.
- **`.env`** holds `FAU_LLM_API_KEY` / `CUSTOM_LLM_API_KEY` (the 120b providers). **Never
  commit `.env`.** Confirm keys are live before a long run (`config.provider_available(...)`).

---

## 3. Models & environments

| | model id | served how |
|---|---|---|
| Teacher/expert (this run) | `fau/gpt-oss-120b` (router: FAU → `custom/openai/gpt-oss-120b:free` → `custom/openai/gpt-oss-120b`) | API only, no local GPU |
| Student — granite | `ibm-granite/granite-4.1-3b` (HF) / `granite4.1:3b` (Ollama) | train: HF; eval: vLLM |
| Student — Qwen 0.8B | `Qwen/Qwen3.5-0.8B` | " |
| Student — Qwen 2B | `Qwen/Qwen3.5-2B` | " |

- **`.venv_train`** — training + HF eval. torch 2.6 cu124, transformers 5.x, trl 1.8, peft
  0.19. **For Qwen students you must have `flash-linear-attention` + `causal-conv1d`
  installed** or training runs 5× slower (see gotcha G-7).
- **`.venv_vllm`** — isolated vLLM 0.25 for fast serving during eval. Never mix with
  `.venv_train`'s torch.

---

## 4. PART A — Generate the 3,000 teacher-only traces

### 4.1 What "teacher-only" means operationally

The 120b plays the **student role** (it plans and acts) with **no teacher in the loop**.
In harness terms that is `skip_teacher = True` with `student_model = fau/gpt-oss-120b`:

- **Plan turn:** the 120b writes one plan (skip_teacher → **0** review/revision rounds).
- **Each step:** the 120b picks a tool; there is no teacher grade; the loop stops when the
  120b calls `finish` or the running budget is exhausted (then `derive_final_answer`).
- The trace records the 120b's plan + its per-step `thought/decision/action` — **no
  `teacher_guidance` block, no gold, no leakage** (§6).

### 4.2 Budget: "3 planning, 3 running, hidden"

Map it to the harness config as:

```
budget            = 3       # running / tool-use steps
max_plan_steps    = 3       # the plan may contain up to 3 planned steps
disclose_budget   = False   # HIDDEN — the agent is not told the step count
planning_steps    = 3       # (see decision note below)
```

> **DECISION NOTE — read this.** `planning_steps` is the number of *teacher-review →
> revise* rounds. Under `skip_teacher=True` there is **no reviewer**, so review rounds do
> not happen and `planning_steps` is effectively ignored: the 120b produces **one** plan.
> Two ways to honor "3 planning steps":
>
> 1. **(Recommended) Solo expert, one plan.** `skip_teacher=True`, `max_plan_steps=3`,
>    `budget=3`, hidden. The 120b plans once (≤3 planned steps) and executes ≤3 tool
>    steps. This is the cleanest *expert trajectory*, matches "teacher only" literally,
>    and costs the fewest API calls. **Use this unless you have a specific reason not to.**
> 2. **Self-refined plan (variant).** Set both `student_model` and `teacher_model` to
>    `fau/gpt-oss-120b`, `skip_teacher=False`, `planning_steps=3`, `planner="student"`.
>    Now the 120b-as-teacher reviews the 120b-as-student's plan for 3 rounds. This gives
>    genuine 3-round plan refinement but (a) doubles plan-stage API cost, (b) is
>    philosophically "the model talking to itself," and (c) the review prompts expose the
>    teacher to **gold** — so the *revised plan* could leak gold reasoning into the trace.
>    If you use this variant you MUST re-apply the leakage gates from §6/G-4 to the plan
>    text before training. **Only use it if the reviewer's self-refinement is itself an
>    object of study.**

Everything downstream (selection, training, eval, report) is identical for both variants.

### 4.3 The runner — and the change you must make

`scripts/run_batched_parallel.py` was built for a **local Ollama student**: it starts one
`ollama serve` per GPU and `ollama pull`s the student tag. **Your student is an API model
(gpt-oss-120b), so there is no Ollama and no GPU student.** Two options:

- **(Recommended) Write a small async API runner** (`scripts/run_teacher_only_traces.py`).
  It is much simpler than the Ollama path because it is pure API: no GPU management, no
  server startup. Skeleton:

  ```python
  # For each question: run the SAME agent loop the harness uses, with the 120b as the
  # student and no teacher. Reuse the harness components directly (as teacher_eval_agent.py
  # already does) but with the LLMClient routing student calls to fau/gpt-oss-120b.
  #
  #   ctx = WorkflowContext(task_id=qid, query=..., metadata=build_metadata(...))
  #   await TeacherGuidedPlanReview(config={}, llm_client=client).execute(ctx)   # skip_teacher → 1 plan
  #   for t in 1..budget:
  #       await TeacherGuidedAgentStep(config={"step_index": t, "budget": 3,
  #                                             "force_finish": t==3}, llm_client=client).execute(ctx)
  #   # export ctx.metadata["teacher_guided_steps"] + plan_review + final metrics
  #
  # metadata: student_model="fau/gpt-oss-120b", teacher_model=<same or unused>,
  #           teacher_router=TEACHER_ROUTER, skip_teacher=True, disclose_budget=False,
  #           budget=3, plan_review_config={enabled:True, planner:"student",
  #           planning_steps:3, max_initial_plan_steps:3, max_revised_plan_steps:3},
  #           corpus_path=<3000 corpus>, retrieval_backend="hotpot_local".
  # Concurrency: an asyncio.Semaphore (see G-9 for the safe value). Stream one JSONL line
  # per finished episode; write a sample-id manifest up front (see G-11).
  ```

  `training_methods/common/teacher_eval_agent.py` is a **near-exact template** — it already
  drives `TeacherGuidedPlanReview` + `TeacherGuidedAgentStep` through a hybrid client. For
  teacher-only you want an *all-API* client (student calls also go to the router, not a
  local model) and `skip_teacher=True`. Copy it and strip the local-policy half.

- **(Alternative) Patch `run_batched_parallel.py`** to skip Ollama when `--student` has an
  `fau/` or `custom/` prefix. More invasive; only worth it if you want its worker/manifest
  machinery. The async runner above is the clean path.

**You must also expose `skip_teacher` in the template builder.**
`scripts/gen_fau_smoke_template.py` hardcodes `"skip_teacher": False` (line ~74) and does
not accept it as a parameter. Add a `skip_teacher: bool = False` argument and thread it
into `mode_config`. Then set it `True` for this run. *(The async runner sidesteps the
template entirely by building `mode_config` directly — preferred.)*

### 4.4 Cost, time, monitoring

- **No GPUs needed** — this is pure API. Concurrency is bounded by the FAU/OpenRouter
  gateway, not hardware (see G-9). At ~4–6 sequential 120b calls per episode (1 plan + up
  to 3 steps, each a reasoning-model call of 10–40 s) and a safe concurrency of ~8–16,
  3,000 episodes take **~2–4 h**. Budget for retries.
- **Write the sample-id manifest before launch** and a **live results/stats file** you
  append to per finished episode (copy the pattern in `scripts/run_trace_collection_g3b_b5.sh`
  + `scripts/watch_trace_collection.py`). Long API runs *will* have transient failures;
  you want to see coverage in real time and re-run only the missing ids.
- **Hard per-call timeouts** (`FAU_TIMEOUT`, `CUSTOM_TIMEOUT`) so a hung provider call
  raises instead of blocking a worker — but note the circuit-breaker trap in **G-9**.

### 4.5 Output

Land traces in `data/simulation_output/traces_oss120b_teacheronly_3000/` with the same
episode shape the exporters produce (`qid, query, gold_answer, final_answer, plan, steps[],
stop_reason, used_steps, final_metrics{em,f1,cover,doc_recall}`). Each `step` has
`student_prompt`, `student_raw`, `student_action`, `action_valid`, `tool_observation`.

---

## 5. PART B — Select the "correct" samples

We train only on episodes where the expert actually **got the answer right** (a wrong
expert trajectory is a bad demonstration). Two correctness signals — use both, prefer the
judge:

1. **Lexical (free, offline):** `cover_match == True` (the gold answer is covered by the
   final answer). Computable directly from `final_metrics`. Fast first pass.
2. **Semantic (recommended, ~$ / API):** run `training_methods/common/judge_final_answers.py`
   over the traces — one gpt-oss-120b verdict per final answer → `{correct, score}`.
   Keep `verdict.correct == 1`. This matches the **exact metric the eval uses**, so the
   training filter and the eval metric are consistent.

Expect the 120b to be correct on a **large majority** of the 3,000 (it is the strong
model). If yield is low, inspect for tool-format failures rather than reasoning failures
(the 120b sometimes over-explains and never emits a clean `finish` — see G-8).

Write the kept qids + episodes to `…/selected_correct.jsonl` and log the yield.

---

## 6. Leakage — the good news

In the guidance line we fought the `[answer hidden]` placeholder for days because the
**teacher** sees gold and its guidance text could echo it into the student's training
target. **In teacher-only mode there is no teacher and the 120b-as-student never sees
gold** — it only ever receives the student-visible state (question + retrieved docs). So:

- **No `[answer hidden]` handling is needed** for the recommended Variant 1.
- **Still run a belt-and-suspenders leakage gate** on every training target: reject any
  example whose target text contains the gold answer verbatim **unless** the gold already
  appears in the question or the retrieved observations (the same `leak_gate_ok` logic in
  `training_methods/common/episode_lib.py`). This catches the rare case where the model
  guessed the gold from parametric knowledge before retrieving it — you do not want to
  train "hallucinate the answer" behavior.
- **If you use Variant 2 (self-review), leakage is back on the table** for the revised
  plan (the reviewer saw gold). Gate the plan text too.

---

## 7. PART C — Build the training data (trajectory cloning)

**Do NOT use the m1 guidance builder** (`m1_sft/build_dataset.py` / `episode_lib.build_step_example`)
— it assembles a `teacher_guidance` block that these traces do not have. Use the
**trajectory-cloning pattern** from `training_methods/m2_rft/filter_rollouts.py :: episode_to_examples`:

- **Plan example:** `prompt = [DEFAULT_SYSTEM, plan_prompt]`, `completion = the 120b's plan JSON`.
- **Per step:** `prompt = [DEFAULT_SYSTEM, student_prompt]`,
  `completion = the 120b's action JSON` (`thought/decision/action/new_facts_extracted`).
- Keep only structurally valid turns (`action_valid`, `parse_student_action` re-validates).

Create `training_methods/exp_teacher_only/build_dataset.py` (copy m2's function; add the
§6 leakage gate and a train/dev split by qid hash, mirroring `episode_lib.qid_split`, 3%
dev). Output `train.jsonl` / `dev.jsonl` / `dev_questions.jsonl` / `stats.json`.

**Gotchas specific to this build:**

- **`decision.category == ""`** — the harness stores an omitted category as the empty
  string, but the action schema **rejects** `""` (it accepts a *missing* category). If you
  copy the 120b's action dict verbatim you will teach the student an invalid value that
  **zeroes every action at eval** (this exact bug cost us a whole granite run — G-2). Drop
  the key when its value is `""`. `training_methods/common/hf_agent_loop._safe_parse_action`
  already normalizes this on the read side; do the same on the write side.
- **Budget-in-prompt mismatch.** These prompts say "You are at step *t* of **3**"; eval
  runs at budget **4** ("of 4"). The student can overfit to the "of 3" wording. Options:
  (a) re-render every training prompt at the eval budget with `build_student_prompt` (most
  correct — the state is stored, so you can re-render); (b) evaluate at budget 3 to match;
  (c) accept the mismatch and note it. **Recommend (a)** for a clean comparison, or run
  eval at budget 4 as we did for m1 and just document that trajectories were budget-3.
- **Data volume vs m1.** You will get ≤3k episodes here vs 18k for m1. That is a real
  confound in the comparison, not a bug — surface it. If you want to control for it, you
  can *also* train an "m1 subset" on a matched ~3k-episode slice, but that is optional.

---

## 8. PART D — Train the three students

Reuse `training_methods/m1_sft/train.py` unchanged (it is model-agnostic). For each model:

```bash
# 4-GPU DDP, 2 epochs, LoRA r=32 all-linear, bf16. Point --train-file at the new dataset.
CUDA_VISIBLE_DEVICES=<4 free gpus> .venv_train/bin/torchrun --nproc_per_node=4 \
  --master_port=297xx training_methods/m1_sft/train.py \
  --model <ibm-granite/granite-4.1-3b | Qwen/Qwen3.5-0.8B | Qwen/Qwen3.5-2B> \
  --train-file training_methods/exp_teacher_only/data/train.jsonl \
  --dev-file   training_methods/exp_teacher_only/data/dev.jsonl \
  --epochs 2 --batch-size 4 --grad-accum 1 \
  --run-dir training_methods/exp_teacher_only/runs/<ts>_<model>
# then: plot_training_curves.py --run-dir <that dir>
```

**Must-dos (each bit us — see gotchas):**

- **Qwen fast kernels first** (G-7): `pip install flash-linear-attention` and build
  `causal-conv1d` locally, or Qwen training is 5× slower. Verify the log does **not** say
  "The fast path is not available."
- **DDP artifact races are fixed** in `train.py` (rank-0 guards on *both* the pre-training
  `train_config.json` and the post-training writes) — but if you fork the trainer, keep
  every `write_json`/rename behind `trainer.is_world_process_zero()` (G-1).
- **Sanity:** eval_loss should land ~0.34–0.45 and token accuracy ~86–89% (that was the
  band for all three m1 students; expert-clone may differ but flag anything wildly off).

---

## 9. PART E — Evaluate (four arms) + report

Reuse the entire `exp_unseen100` stack **without changes to the eval logic** — only point
it at the new adapters. The four arms for *this* experiment (the trained model is now the
expert-clone "e1"):

| arm | meaning |
|---|---|
| `base` | untrained student, no teacher |
| `base_teacher` | untrained student + live gpt-oss-120b teacher |
| `m1` (relabel `e1`) | **expert-clone student, no teacher** ← the arm under test |
| `m1_teacher` | expert-clone student + live teacher |

Steps (all already built):

1. **Merge weights for the Qwen students** — vLLM cannot apply a LoRA adapter to Qwen3.5's
   `Qwen3_5ForConditionalGeneration` (the module names don't match; it loads but silently
   does nothing — G-5). Merge the adapter into the base and splice into the full composite
   checkpoint (the snippet is in this session's history / `report_qwen*` commits). Granite
   serves its adapter fine as a normal LoRA module.
2. **Run the experiment:**
   ```bash
   .venv_train/bin/python training_methods/exp_unseen100/run_experiment.py \
     --backend vllm --gpus <5 free> --clients-per-server 4 \
     --model <base id> [--merged-model <merged dir>] \
     --exp-tag exp_teacheronly_<model> [--adapter <granite adapter>]
   ```
   20 jobs (4 arms × 5 seeds 11/23/37/53/71), auto-judges all 2,000 answers, writes
   `analysis/`. Servers stagger themselves (G-6). **Watch for a short teacher-arm seed**
   (G-9): if any `*_teacher_s*` run has < 100 episodes, refill just that seed with a
   dedicated single server at low `--teacher-concurrency` and re-judge, exactly as we did
   (see this session's s23 repair) — do **not** report a seed with a partial count.
3. **Per-model report:** `analyze_experiment.py --exp-dir <run>` → `analysis/report.html`
   (box plots, paired t + Wilcoxon + Welch, Holm correction, tokens/time/GPU).

---

## 10. PART F — The comparison report

Regenerate the cross-model efficiency report and, more importantly, put **guidance-m1 vs
trajectory-e1 head to head** for each student. Two ways:

- **Quickest:** feed both experiments' dirs to `training_methods/exp_unseen100/compare_models.py`
  with labels like `"Qwen3.5-2B (guidance-m1)"=<m1 dir>` and
  `"Qwen3.5-2B (expert-e1)"=<e1 dir>`. The accuracy-vs-cost frontier and the per-model
  significance table then show which distillation wins per model.
- **Cleanest:** add a small comparison that pairs, **per student**, the `m1`/`e1` arms and
  reports the paired difference on teacher-verdict correctness (the primary metric), plus
  the `+teacher` variants.

**State these confounds explicitly in the report** (they are the difference between a
credible and a misleading conclusion):

1. **Data volume:** guidance-m1 saw ~18k episodes (6 budgets × 3k); expert-e1 saw ≤3k.
2. **Budget:** guidance traces spanned budgets 1–9; expert traces are budget-3; eval is
   budget-4. Note how you handled the prompt-budget wording (§7).
3. **Error-recovery:** guidance data shows the student *its own mistakes and the fix*;
   expert data shows only clean successes — the student never sees how to recover from a
   bad step. This is the central qualitative difference and likely the most interesting
   result.
4. **Model family:** granite-3B is a different family/generation than the Qwen students;
   size and architecture are confounded across models (they are not across the m1↔e1
   comparison *within* a model, which is why the within-model pairing is the clean test).

---

## 11. Lessons & gotchas (read before you start)

These are the concrete traps from the guidance line of work. Most are non-obvious.

- **G-1 — DDP artifact-write races.** Under `torchrun`, every rank runs the script; any
  `write_json` that does an atomic `.tmp`→final rename **races across ranks** and crashes
  the job (one rank renames the tmp away before another can). Guard **every** shared-file
  write with `trainer.is_world_process_zero()`. We hit this three times (final metrics,
  trainer state, and the *pre-training* `train_config.json`). Already fixed in `train.py`.
- **G-2 — `decision.category == ""` zeroes eval.** The harness stores an omitted category
  as `""`; the pydantic action schema accepts a *missing* category but **rejects `""`**.
  Training targets that copy the stored dict verbatim teach the model to emit `""`, and
  then **every action fails validation at eval** with `invalid_category` — silent, total,
  looks like the model "can't act." Strip the empty category on both the write side
  (dataset build) and the read side (`_safe_parse_action` already does).
- **G-3 — Miscalibrated pseudo-scores.** Plan-review feedback is stored with a default
  `score: 0.0` even when positive; if you carry it into targets you train a "score 0.0 +
  praise" pattern. Not applicable to teacher-only (no scores), but the general lesson:
  **inspect what stored fields actually mean before cloning them into targets.**
- **G-4 — Leakage.** The guidance data needed heavy `[answer hidden]` redaction because the
  teacher sees gold. Teacher-only Variant 1 is leak-free by construction, **but still run
  the `leak_gate_ok` check** (§6) to catch parametric-knowledge guesses. Variant 2
  reintroduces leakage via the plan reviewer.
- **G-5 — vLLM can't LoRA Qwen3.5.** Qwen3.5 loads in vLLM as
  `Qwen3_5ForConditionalGeneration`; a PEFT adapter trained against the `…ForCausalLM`
  view has non-matching module names, so vLLM **loads the adapter and silently ignores
  it** (base and "adapter" outputs identical). Detect with an *identical-outputs probe*
  (base vs adapter on one prompt). **Fix: merge the adapter into the base weights and
  serve the merged model.** When you merge, splice the 320 trained language-model tensors
  into the model's **full composite checkpoint** (keep the mtp head etc. and the original
  `config.json`), or vLLM rejects the config type. Granite serves its LoRA fine.
- **G-6 — Two vLLM servers per GPU race at startup.** The merged-weights eval starts a base
  server and a merged-twin on the *same* GPU; two engines profiling free memory at the
  same instant can collide and one fails with "connection refused." Fixed with a 20 s
  stagger in `run_experiment.py`. If you launch servers by hand, stagger them.
- **G-7 — Qwen training fast path.** Without `flash-linear-attention` **and**
  `causal-conv1d`, Qwen3.5 falls back to a torch implementation and trains ~5× slower
  (5 s/step vs 1 s/step). Install `fla` (pip) and **build `causal-conv1d` locally**:
  `pip install wheel setuptools` first (the `--no-build-isolation` build needs `wheel`),
  then `CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install causal-conv1d
  --no-build-isolation --no-cache-dir` (the prebuilt wheel links a newer glibc than the
  box has — force a local nvcc build). `serve_vllm.sh` puts the venv's `ninja` on PATH for
  vLLM's JIT kernels; do the same for any tool that JIT-compiles.
- **G-8 — The 120b over-explains and won't `finish`.** As a reasoning model it sometimes
  narrates instead of emitting a clean `finish` action, burning the budget → answer
  "unknown" or a verbose non-EM answer. Give it a **generous completion budget**
  (`teacher_max_tokens ≈ 2500`, retry 4000) and rely on the harness's forced-finish
  fallback. When measuring "correct," prefer the semantic judge over EM (verbose-but-right
  answers fail EM but pass cover/judge).
- **G-9 — The gateway circuit breaker permanently disables a worker.** The router trips a
  provider after a hard timeout and **does not un-trip until the process restarts**. Under
  high concurrency (we ran up to ~50 concurrent teacher calls) a burst of timeouts can
  trip **all** providers for one worker, after which every remaining episode fails
  instantly with "no configured provider" — we lost 71/100 episodes in one seed this way.
  Mitigations: (a) keep per-run teacher concurrency modest (≤ ~5 per process; total across
  processes matters); (b) after a run, **check every teacher-arm run for a short episode
  count and refill short seeds** with a fresh low-concurrency process; (c) the FAU→free→
  paid fallback exists precisely for this — make sure all three providers are configured.
- **G-10 — Local Ollama students (only if you ever use one).** granite4.1:3b's default
  context is huge; 8 parallel slots blow past 80 GB of KV cache, the allocation fails, and
  **Ollama silently degrades to CPU** (every call times out). Set
  `OLLAMA_CONTEXT_LENGTH=16384`. Not relevant to the API-only teacher-only run.
- **G-11 — Long API runs need a manifest + live watcher.** Write all expected qids before
  launch, append one line per finished episode, and diff to a `missing_ids.txt` so you can
  re-run only the gaps. Transient API failures are normal; you want coverage visible, not a
  black box.
- **G-12 — transformers 5.x / TRL 1.8 quirks.** `apply_chat_template(..., return_tensors=
  "pt", return_dict=True)` then `model.generate(**enc)` (it returns a `BatchEncoding`, not
  a tensor). `KTOConfig`/`DPOConfig` have **no** `max_prompt_length` (only `max_length`).
  `StudentActionModel` has `extra="ignore"` so extra keys are tolerated at parse time.
- **G-13 — Batched/served generation isn't bit-identical to sequential HF.** vLLM and
  batched HF give cosmetically different greedy outputs (kernels/padding). **Compare arms
  within one backend**, and report **tokens** (hardware-independent) as the primary cost
  metric; wall time is infra-dependent and GPU peak-mem isn't captured per-arm in vLLM
  serving mode (the eval process doesn't hold the CUDA context).
- **G-14 — Never attribute commits to an assistant.** No "Co-authored-by" / "Generated by"
  lines anywhere in commits or PRs (standing project rule).

---

## 12. End-to-end checklist

```
[ ] Confirm .env has live FAU_LLM_API_KEY / CUSTOM_LLM_API_KEY (provider_available()).
[ ] Add skip_teacher param to gen_fau_smoke_template.py OR build mode_config directly.
[ ] Write scripts/run_teacher_only_traces.py (async API runner; copy teacher_eval_agent.py,
    all-API client, skip_teacher=True, budget=3, max_plan_steps=3, hidden, planner=student).
[ ] Manifest + live watcher; hard per-call timeouts; concurrency ≤ ~8–16 (G-9).
[ ] Collect 3,000 → data/simulation_output/traces_oss120b_teacheronly_3000/.
[ ] Filter to correct: cover_match, then judge_final_answers.py (keep verdict.correct==1).
[ ] build_dataset.py (trajectory cloning via m2 episode_to_examples pattern):
      strip decision.category=="" (G-2); leak gate (G-4/§6); re-render prompts at eval
      budget or note the mismatch (§7); qid-hash dev split.
[ ] Qwen fast kernels installed & verified (G-7).
[ ] Train granite-3B / Qwen-0.8B / Qwen-2B (4-GPU DDP, rank-0 guards intact G-1); plot curves.
[ ] Merge Qwen adapters into full checkpoints; identical-outputs probe (G-5). Granite: LoRA ok.
[ ] Four-arm eval per model (run_experiment.py, vLLM, staggered servers G-6); refill any
    short teacher seed (G-9); 2000/2000 judged.
[ ] analyze_experiment.py per model → report.html.
[ ] compare_models.py: pair guidance-m1 vs expert-e1 per student; state confounds (§10).
[ ] Publish reports; atomic commits; push. NO co-author lines (G-14).
```

---

## 13. Where the earlier results live (for the m1 side of the comparison)

- Guidance-m1 unseen-100 experiments: `training_methods/exp_unseen100/runs/…_exp` (granite),
  `…_exp_qwen05b`, `…_exp_qwen2b`; per-model reports in `report_qwen05b/`, `report_qwen2b/`,
  and the cross-model efficiency report in `report_compare/`.
- Guidance-m1 training runs & curves: `training_methods/m1_sft/runs/…_train4gpu` (granite),
  `…_qwen05b`, `…_qwen2b`.
- Headline guidance-m1 numbers (teacher-verdict correct, unseen-100, 5 seeds): granite-3B
  base 30.8 / base+teacher 63.6 / m1 48.6 / m1+teacher 59.5; Qwen-0.8B 3.0 / 32.6 / 60.2 /
  69.6; Qwen-2B 21.2 / 60.2 / 63.6 / 74.6. **These are the numbers your expert-e1 arms are
  measured against.**
```
