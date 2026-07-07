# Teacher Guidance — Session Handoff

> Paste-this-into-a-new-session context for continuing development on another device.
> Repo: <https://github.com/ModMaamari/TeacherGuidence> (a fork of AgentSim).
> Active branch: **`feature/teacher-guidance`** (everything below is on it, pushed).
> Last updated: 2026-07-06 · 145 commits on the branch · 343 tests passing.

---

## 0. Quick start from scratch (new machine) — runbook for an AI agent

Follow these steps top to bottom. They bring the project up from nothing. Read §3–§5
for the *why* behind each choice.

### 0.1 Clone and check out the branch
```bash
git clone https://github.com/ModMaamari/TeacherGuidence
cd TeacherGuidence
git checkout feature/teacher-guidance
git pull
```

### 0.2 Python env (use 3.10–3.12, NOT 3.13/3.14)
```bash
python3.12 -m venv .venv
source .venv/bin/activate                # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .                          # agentsim + deps: torch (CPU), pydantic, json-repair
pip install datasets pytest
python -m pytest tests/ -q                # expect: all passing (343 at handoff time)
```
If pytest's temp dir errors on Windows, add `--basetemp=<writable dir>`.

### 0.3 Local student model via Ollama (GPU) — used by every `*_local` template
```bash
# winget install Ollama.Ollama  (Windows) / https://ollama.com/download (macOS/Linux)
ollama pull qwen3.5:2b            # good default; also used: qwen3.5:0.8b, :4b
ollama pull granite4.1:3b         # ibm-granite/granite-4.1-3b — strongest student tried so far
```
Skip this if you will only run the all-cloud templates (`*_openrouter`).

### 0.4 Create `.env` at the repo root (git-ignored; supply your own keys)
```ini
# --- Teacher: cost router, tried in this order (see §4) ---
# 1) FAU academic gateway (free, but occasionally goes down for a specific model —
#    see the FAU hang gotcha in §11). Endpoint already defaults to the FAU URL below.
FAU_LLM_ENDPOINT=https://hub.nhr.fau.de/api/llmgw/v1
FAU_LLM_API_KEY=sk-REPLACE_ME
# 2) / 3) OpenRouter free tier, then paid, as fallback (endpoint must NOT include /v1).
CUSTOM_LLM_ENDPOINT=https://openrouter.ai/api
CUSTOM_LLM_API_KEY=sk-or-v1-REPLACE_ME

# --- Student: local Ollama, separate endpoint so it never collides with the teacher ---
OLLAMA_ENABLED=true
OLLAMA_ENDPOINT=http://127.0.0.1:11434

SIMILARITY_METRIC=token_overlap
LLM_TIMEOUT=300
LLM_MAX_RETRIES=3
LLM_MAX_TOKENS=1500
FAU_TIMEOUT=45          # hard wall-clock cap on a single FAU call (see §4)
```
Any subset of the three teacher providers works — `config.provider_available()` skips
whichever aren't configured. Get an OpenRouter key at <https://openrouter.ai/keys>.

### 0.5 Build the dataset
```bash
python scripts/prepare_hotpot_teacher_guidance.py \
  --subset distractor --split validation --limit 100 --shuffle \
  --out_dir data/datasets/hotpot_teacher_guidance_exp100
```
`data/` is git-ignored, so datasets and run outputs do NOT come with the clone — you
regenerate them here. **Do not re-run this if you want to compare against prior 100-q
runs** — several reports/results below are pinned to the existing
`hotpot_teacher_guidance_exp100` question set.

### 0.6 Run, view, score
```bash
agentsim simulate hotpot_teacher_guided_b9_plan_review --validate-only
agentsim simulate hotpot_teacher_guided_b9_plan_review
agentsim viewer --port 8000
python scripts/aggregate_teacher_guidance_run.py --run-dir data/simulation_output/<run_dir>
python scripts/validate_teacher_guidance_run.py  --run-dir data/simulation_output/<run_dir>
```
For a 100-question run across multiple GPUs, use the batched runner instead of a single
`agentsim simulate` call — see §5.

---

## 1. What this project is

A **Teacher Guidance** pipeline added to AgentSim that generates step-level
retrieval-agent trajectories on QA datasets (HotpotQA first). Per question:

```
[optional plan review]  student plans → teacher reviews → student revises
then for each step (up to a step budget B):
  student emits ONE JSON tool action  → environment executes the tool
  → teacher privately evaluates the step → guidance renderer shows the student only the
    feedback allowed by guidance_level (0–4) → leakage checker sanitizes → everything logged
```

The end goal is **distillation**: generate many trajectories with a large teacher-guided
setup, keep only the good ones (correct, grounded, efficient, no leakage), and turn them
into SFT data to fine-tune a small student that reasons well *without* a teacher at
inference time. The generation side (above) is mature and has produced several 100-sample
runs (§8). The distillation side (trace-quality gate → canonical selection → SFT export,
§7) is built and unit-tested but has **not yet been run end-to-end into a trained model**.

Central design rules (kept stable):
- **Student never sees the gold answer or teacher-private diagnosis.** Visibility is
  enforced in **code** (the guidance renderer + leakage checker), not by prompting.
- The **teacher** sees gold metadata; how much it can tell the student is gated by
  `guidance_level`.
- Tool execution is **deterministic** (no LLM); the environment owns the corpus and
  validates extracted spans against retrieved text.
- The final step is forced to `finish` when the budget is exhausted, and the answer
  committed there is never allowed to regress to a blank/"unknown" if a real answer was
  ever produced earlier in the episode (see §11 for the one remaining gap in this area).

---

## 2. Where the code lives

```
agentsim/teacher_guidance/
  schemas.py            dataclasses: StudentAction, TeacherEvaluation, GuidanceConfig,
                         PlanReviewConfig, records, etc.
  pydantic_schemas.py    Pydantic models mirrored from schemas.py's enums (single source of
                         truth) used two ways: (a) grammar-constrained decoding — their
                         .model_json_schema() is handed to Ollama's `format` field so the
                         model is physically unable to emit invalid JSON/tool/category;
                         (b) post-hoc validation in json_utils.py. Includes a finish-only
                         schema (StudentFinishActionGenerationModel) always used on the
                         forced final step regardless of the per-run schema opt-out.
  json_utils.py          tolerant JSON extract/parse; repair chain: json.loads -> regex
                         fix (curly quotes, trailing commas, invalid \-escapes) ->
                         json_repair library fallback. validate_* functions delegate to
                         the Pydantic models above, converting ValidationError into the
                         (bool, List[str]) shape callers use.
  hotpot_converter.py    HotpotQA -> question rows + corpus rows (HF columnar or raw shape).
  local_retrieval.py     HotpotLocalRetriever (BM25 if rank_bm25 present, else lexical).
  metrics.py             normalize_answer, exact_match, cover_match (robust bidirectional +
                         prefix-token correctness, tolerant of dropped words/fillers), f1,
                         doc/fact recall, compute_step_metrics, compute_final_metrics.
  leakage.py             word-boundary detection + strict-policy sanitization (redacts only
                         the gold answer itself, never titles/docs/spans).
  guidance_policy.py     render_student_guidance (levels 0-4) + derive_plan_review_guidance.
  prompts.py             build_student_prompt / build_teacher_prompt + plan prompts.
                         disclose_budget flag controls whether the step counter/plan prompt
                         states the budget (see hidden-budget mode, §4/§8).
  llm_call_log.py        timed_completion: shared per-call logging helper (timestamps, raw
                         response, token usage/cost, response_schema passthrough) used by
                         both the agent-step and plan-review components.
  tool_executor.py       execute_student_tool (decompose/reformulate/search/extract/verify/
                         synthesize/finish). derive_final_answer: priority fallback chain
                         (params answer -> previously-committed answer -> draft_answer ->
                         extracted facts -> "unknown", never empty). clean_forced_answer
                         strips markdown/JSON wrapper noise from a forced free-text answer.
  plan_review.py         compute_plan_review_metrics.
  plan_execution.py      validate_formal_plan + PlanTracker (deterministic adherence).
  episode_exporter.py    TeacherGuidanceEpisodeExporter -> clean per-episode files, incl.
                         final_metrics.teacher_answer_correct/teacher_answer_score (the
                         teacher's own verdict on the final answer -- see §6).
  optimality.py           deterministic path-efficiency signals (redundant steps, wasted
                         calls, step-count vs a lower bound) for the distillation gate.
  trace_quality.py        acceptance gate: is this episode good training data? (correct +
                         grounded + clean path + self-terminated + no leakage).
  trace_selection.py      canonical trace selection: keep one optimal trace per qid across
                         the whole generation matrix (shortest/cleanest/most-efficient).
  sft_diversity.py        diversity metrics + tool-sequence-signature capping so a handful
                         of trajectory shapes don't dominate the training set.
  sft_export.py           accepted episodes -> chat-format SFT examples (plan + per-step).
  sft_internalize.py      strips the teacher-guidance block from the input side of each
                         training pair so the student learns to reason as if it had
                         already internalized the guidance, with no teacher at inference.
  dataset_split.py        qid-level stratified train/val/test split (hash(seed, qid) ->
                         deterministic, no question leaks across splits).
  sft_report.py           SFT dataset stats + hard gold-answer-leakage healthcheck.
  viewer/                 trajectory explorer web UI (see §6).

agentsim/components/control/
  teacher_guided_agent_step.py    one student tool-use step; student repair retry; formal-
                                  plan adherence; force-finish (_resolve_forced_finish: if
                                  the student's own finish answer is "unknown", makes one
                                  extra free-text answer-extraction call before giving up).
  teacher_guided_plan_review.py   preflight: student multistep loop OR teacher-planner;
                                  skips revision when teacher accepts.

Patched upstream files (additive, low risk):
  agentsim/simulation/modes/standard.py  injects mode_config (gold, models, guidance,
                                         plan_review_config, corpus_path, budget,
                                         disclose_budget) into context.metadata.
  agentsim/workflow/executor.py          stops the workflow when a component returns
                                         verdict == FINISH.
  agentsim/simulation/loader.py          allows teacher_guided_agent_step as terminal comp.
  agentsim/cli.py                        `viewer` subcommand; episode export hook.
  agentsim/clients/llm_client.py         get_completion(response_schema=...) — Ollama
                                         "format", OpenRouter response_format=json_object;
                                         _fau_completion (OpenAI-compatible FAU gateway,
                                         hard asyncio.wait_for(FAU_TIMEOUT) wall-clock cap);
                                         get_completion_with_fallback — the provider router
                                         (see §4); per-call max_retries; Ollama /api/chat
                                         (chat template applied, not /api/generate).
  agentsim/config.py                     provider_available(model_id) classmethod; FAU_*,
                                         OLLAMA_THINK, FAU_TIMEOUT options.

scripts/
  prepare_hotpot_teacher_guidance.py     build the dataset (HF or local raw json).
  validate_teacher_guidance_run.py       leakage/integrity validator (exit!=0 on leak).
  aggregate_teacher_guidance_run.py      mean EM/correct/F1/recalls, steps, stop reasons.
  run_trajectory_viewer.py               launch the UI.
  gen_fau_smoke_template.py              build a teacher-guided template with the FAU-first
                                         cost router (TEACHER_ROUTER); base for both runners
                                         below.
  run_fau_smoke_parallel.py              one episode per GPU, N GPUs, round-robin shards
                                         (used for the *_run100 family in §8).
  run_batched_parallel.py                several episodes IN FLIGHT PER GPU (one `ollama
                                         serve --num-parallel=K` per GPU + K simulate
                                         workers) — the right tool for small students on
                                         big (80 GB) GPUs. --teacher-model/--teacher-router
                                         overrides let you pin the teacher when FAU is down.
  consolidate_run_shards.py              merge per-worker shard output dirs into one run
                                         dir so the explorer shows a single run.
  run_experiment_matrix.py               orchestrator for the N-model x M-setting matrix
                                         (GPU/time monitoring, restart-safe).
  gen_experiment_matrix_templates.py      generate the matrix's per-config templates.
  build_experiment_matrix_report.py       analysis + report/plots for the matrix.
  filter_traces.py / export_sft_dataset.py / split_sft_dataset.py / build_sft_report.py
                                         CLI front-ends for the trace_quality -> selection
                                         -> sft_export -> dataset_split -> sft_report
                                         distillation pipeline (§7).
  recompute_answer_correct.py            re-derive final_metrics.answer_correct on old run
                                         files after a matching-rule change (retroactive).

templates/workflows/    hotpot_teacher_guided_b3|b5|b7|b9|b10|b12[,_plan_review],
                        bmax20_plan_review (hidden-budget-mode ablation).
templates/simulations/  45 templates: guidance-level sweep (b5_g0..g4), plan-review
                        variants, FAU smoke templates, the 4x5 experiment-matrix configs,
                        and per-student local/openrouter templates.

tests/ (+ tests/teacher_guidance/)  343 tests, all green.
```

Planning/scaffolding files that are intentionally **untracked** (never committed):
`TG-*.md`, `commitN_message.txt`, `.env`, `reports/SESSION_SUMMARY.md` (a separate,
denser session log kept locally — not required reading, this file supersedes it).

User docs (committed): `TEACHER_GUIDANCE.md` (setup/run/planning options),
`TEACHER_GUIDANCE_UI.md` (the viewer).

---

## 3. Environment & setup

- **Use Python 3.10–3.12** (NOT 3.13/3.14 — `torch`/`sentence-transformers` have no
  wheels there, and a full `simulate` imports them).
- `pip install -e .` pulls in everything, including `pydantic` and `json-repair` (added
  for the JSON-schema-enforcement work — no separate install step needed).
- **Run tests:** `python -m pytest tests/ -q` (343 passing at handoff time).
- Windows console is cp1252 and chokes on the `✓/✗` glyphs the CLI prints — set
  `PYTHONUTF8=1` first.

---

## 4. Models & providers (how the LLM routing works)

**Teacher = a 3-tier cost router**, tried in this order and re-checked on *every call* so
a mid-run credential/outage change is picked up immediately:

1. **`fau/gpt-oss-120b`** — NHR@FAU's free academic "LLMs as a Service" gateway (an
   OpenAI-compatible `fau` provider). Free, but see the gotcha below — this specific model
   occasionally goes fully unresponsive on FAU's side for extended periods.
2. **`custom/openai/gpt-oss-120b:free`** — OpenRouter's free tier of the same model
   (rate-limited, 429s common under load).
3. **`custom/openai/gpt-oss-120b`** — OpenRouter paid, same model, as the reliable last
   resort (~$0.000004/call — negligible for a 100-question run).

Router order is defined in `scripts/gen_fau_smoke_template.py::TEACHER_ROUTER` and
consumed by `LLMClient.get_completion_with_fallback` in `agentsim/clients/llm_client.py`.
Robustness properties (all covered by tests in `tests/test_llm_client.py`):
- **Skips unconfigured providers up front** via `config.provider_available(model_id)` —
  no guaranteed-failing round-trip, no crash from a provider's missing-key `ValueError`.
- **Falls through on ANY exception**, not just HTTP errors (malformed response, timeout,
  connection error, etc.) — a single bad provider never takes the whole run down.
- **Non-last providers are called fail-fast** (`max_retries=0`) so falling through is
  quick; the last configured provider keeps normal retry/backoff as the true last resort.
- **Hard wall-clock timeout on FAU** (`FAU_TIMEOUT`, default 45s, via `asyncio.wait_for`).
  This exists because `httpx`'s own timeout is *inter-byte*, not wall-clock — a FAU
  backend that holds the connection open trickling keepalive bytes without ever finishing
  the response defeats it and hangs forever. `asyncio.wait_for` forces a `TimeoutError`
  that the router treats like any other failure.
- **Circuit breaker**: once a provider hard-times-out, `LLMClient._tripped_providers`
  remembers it for the rest of the process, so every subsequent call skips straight past
  it instead of paying the full 45s again. Cleared only by restarting the process (a fresh
  `agentsim simulate` worker starts untripped, so a later run gets to retry FAU).

**Student = local Ollama** (GPU), always a separate endpoint from the teacher so the two
never collide. Models tried, best first:
- `granite4.1:3b` (`hf.co/ibm-granite/granite-4.1-3b-GGUF:Q8_0`) — **best so far**: 73%
  correct / 98% grounded at budget=12 (disclosed), 72% teacher-verdict at budget=9
  (hidden). Ties or beats every other student tested.
- `qwen3.5:2b` — 73–45% correct depending on config (see §8 for why the same model spans
  that range).
- `qwen3.5:0.8b` — 58–39% correct; smallest, fastest, still usable.
- `qwen3.5:4b` — used in the earliest (10-question, glm-5.2-teacher) experiments.
- `MiniCPM5-1B` — **abandoned**. Needed two fixes to stop producing gibberish (Ollama
  `/api/chat` for its ChatML template, `student_use_response_schema=False` because
  llama.cpp's grammar collapsed its policy to `synthesize {}`), and even corrected only
  hit 7% — the user has since said not to spend further effort on it.

**Grammar-constrained decoding** (`student_use_response_schema`, default **`True`**):
when true, the student's Ollama call gets a JSON-schema `format` (from
`pydantic_schemas.py`) that makes invalid JSON/tool/category physically impossible. This
flag has flipped twice — worth knowing the history if you see an old commit reference the
other default:
- Originally `True`. MiniCPM5-1B collapsed under the all-tools grammar (shortest-path
  degenerate output), so commit `a7a3f6a` flipped the *global default* to `False`
  (unconstrained + a strengthened prompt) to fix it universally.
- The user later decided to stop supporting MiniCPM and asked to **revert to the old
  schemas** (commit `f422af3`): default is `True` again; the terse pre-`a7a3f6a` prompt
  schema text is back. A **per-run opt-out** (`--student-no-schema` in the runner scripts)
  still exists for any future model that collapses under the grammar the way MiniCPM did.
- The forced *last* step always uses the finish-only schema regardless of this flag, so a
  real answer is always committed even when the general grammar is off.

`.env` keys (all in `agentsim/config.py`): `FAU_LLM_ENDPOINT` (defaults to the gateway
URL), `FAU_LLM_API_KEY` (or `LLMAPI_KEY`), `CUSTOM_LLM_ENDPOINT`/`CUSTOM_LLM_API_KEY`
(OpenRouter, endpoint **without** `/v1`), `OLLAMA_ENABLED`/`OLLAMA_ENDPOINT`,
`OLLAMA_THINK` (off by default — Qwen3 "thinking" wastes budget on JSON-only tasks),
`FAU_TIMEOUT` (default 45), `LLM_TIMEOUT`/`LLM_MAX_RETRIES`/`LLM_MAX_TOKENS`.

---

## 5. How to run an experiment

**Single template, small scale:**
```bash
python scripts/prepare_hotpot_teacher_guidance.py --subset distractor --split validation \
  --limit 100 --shuffle --out_dir data/datasets/hotpot_teacher_guidance_exp100
agentsim simulate hotpot_teacher_guided_b9_plan_review --validate-only
agentsim simulate hotpot_teacher_guided_b9_plan_review
agentsim viewer --port 8000
```

**100+ samples, multiple GPUs, one episode at a time per GPU** (the pattern used for the
`*_run100` reports in §8):
```bash
python scripts/run_fau_smoke_parallel.py --num-samples 100 --num-gpus 8 \
  --student ollama/qwen3.5:2b --budget 12 --planning-steps 3 \
  --out-root data/simulation_output/fau_run100
```

**100+ samples, several episodes IN FLIGHT per GPU** (better GPU utilization for small
students on big cards — the pattern used for the granite/qwen b9 runs this session):
```bash
python scripts/run_batched_parallel.py --num-samples 100 --gpu-ids 1,2,3,4,5,6,7 \
  --workers-per-gpu 4 --student ollama/granite4.1:3b \
  --budget 9 --planning-steps 3 --max-plan-steps 9 \
  --workflow hotpot_teacher_guided_b9_plan_review --hidden-budget \
  --out-root data/simulation_output/granite3b_b9_run100
```
`--hidden-budget` enables the efficiency-lever mode from §8 (student never sees the step
count until forced to finish). `--teacher-model`/`--teacher-router` let you override the
default FAU-first router for one run — useful when FAU is confirmed down and you don't
want to burn the 45s timeout on every single step (pin straight to
`custom/openai/gpt-oss-120b` instead).

Outputs land in `data/simulation_output/<sim_id>/<run_uuid>/hotpot_questions/sample_XXX/`
with `teacher_guidance_episodes.jsonl`, `student_sft.jsonl`, `teacher_sft.jsonl`,
`student_visible_guidance.jsonl`, `plan_review_rows.jsonl`,
`teacher_guidance_metrics.json`. The batched/parallel runners auto-consolidate shards into
one `run/` dir (`scripts/consolidate_run_shards.py`) so the explorer shows a single run.
**All of `data/` is git-ignored** — regenerate on a new device.

---

## 6. The trajectory explorer UI (`agentsim viewer`)

Dependency-free (Python stdlib server + vanilla HTML/CSS/JS in
`agentsim/teacher_guidance/viewer/`).

- **Main page is a runs table**, newest first: episode count, accuracy, F1, doc recall,
  student/teacher model, and a **teacher-source breakdown** (% of teacher calls served by
  FAU vs OpenRouter-free vs OpenRouter-paid — the fastest way to see whether a run
  actually used the free tier). Click a row to open that run's trajectory list.
- `/api/runs` is **cached by a stat-signature** (path/mtime/size of every episode file),
  so repeat loads don't re-parse everything (~100ms vs ~1000ms before).
- Per-episode view: 3-phase plan review, step timeline, raw model I/O (with per-call
  timestamps and token usage), and a **"Show teacher-private"** toggle (now **on by
  default**) revealing the gold answer + teacher diagnosis.
- **Correct/incorrect signal (important, changed 2026-07-06):** an episode counts as
  correct when the **teacher's own verdict** (`final_metrics.teacher_answer_score`,
  which the teacher assigns 0.0–1.0 while seeing the gold answer) is **>= 0.40**. This
  supersedes the older deterministic cover-match signal, which remains only as a fallback
  for episodes with no teacher verdict (e.g. `skip_teacher` ablation runs, or a teacher
  call that didn't return a score). Implemented in
  `agentsim/teacher_guidance/viewer/data_access.py::_answer_correct` /
  `TEACHER_CORRECT_THRESHOLD`. This is computed **on the fly**, so it applies
  retroactively to old runs without re-running anything — but a long-running `agentsim
  viewer` process needs a **restart** to pick up code changes like this one (it's plain
  Python, not hot-reloaded).

---

## 7. The distillation / SFT pipeline (built, not yet exercised end-to-end)

Turns accepted generation-phase episodes into a fine-tuning dataset for a student that
runs **without** a teacher at inference time. Stages, in order:

1. **`trace_quality.py`** — acceptance gate: is one episode good enough to train on?
   (correct, grounded in evidence it actually retrieved, clean path — no invalid JSON or
   redundant steps —, self-terminated rather than force-finished, no leakage.)
2. **`optimality.py`** — deterministic path-efficiency signals (redundant steps, wasted
   calls, step count vs. a lower bound) feeding into both the gate and selection.
3. **`trace_selection.py`** — canonical selection: keep the single most-optimal accepted
   trace **per qid** across the whole generation matrix, so the SFT set isn't dominated by
   whichever model/setting happened to solve the easy questions the most times.
4. **`sft_diversity.py`** — tool-sequence-signature capping + Jaccard near-duplicate
   check, so a handful of trajectory shapes don't dominate training.
5. **`sft_export.py`** — accepted traces -> chat-format examples (the plan step, teacher-
   free by construction, plus one example per tool-use step).
6. **`sft_internalize.py`** — strips the teacher-guidance block from each example's input
   side, so the student learns to reason *as if it had already internalized the
   guidance*, matching the no-teacher inference-time setup.
7. **`dataset_split.py`** — qid-level stratified train/val/test split
   (`hash(seed, qid)`-based, deterministic, no question crosses splits).
8. **`sft_report.py`** — dataset stats + a **hard leakage healthcheck**: verifies no
   accepted step is flagged `gold_answer_leaked` and no gold answer string appears
   verbatim in any training input.

CLI front-ends: `scripts/filter_traces.py`, `scripts/export_sft_dataset.py`,
`scripts/split_sft_dataset.py`, `scripts/build_sft_report.py`. All unit-tested. **Not yet
run against a real, large generation corpus, and no student has actually been fine-tuned
yet** — see §9 open items.

---

## 8. Experiments run so far

### Early era (10 fixed HotpotQA questions, teacher = OpenRouter `glm-5.2`)
| Run | Student | B | Answer-correct | EM | F1 | Notes |
|---|---|---|---|---|---|---|
| openrouter B5 | qwen3.7-plus (cloud) | 5 | 0.60 | 0.60 | 0.72 | first session |
| local B5 (after robustness fixes) | qwen3.5:4b (GPU) | 5 | 0.60 | 0.10 | 0.14 | matches cloud; 0 empty answers |

**Biggest early finding:** a local 4B's apparent weakness was a pipeline/tool-call bug
(quoted extract spans -> `span_not_found` -> empty answers), not a capability gap — fixed,
reaching parity with the cloud model.

### 100-question era (teacher = `gpt-oss-120b`, FAU-first cost router, `run_fau_smoke_parallel.py`, budget=12, 3 plan-review rounds, disclosed budget, 1 episode/GPU)
| Report | Student | Answer-correct | Grounded | Teacher cost |
|---|---|---|---|---|
| `reports/fau_gptoss_run100` | qwen3.5:2b (FAU only, no router) | 71% | 94% | $0 (heavy 429s absorbed by backoff) |
| `reports/fau_router_run100` | qwen3.5:2b | 73% | — | $0 (100% FAU) |
| `reports/fau_router_run100_qwen0p8b` | qwen3.5:0.8b | 58% | 97% | $0.0002 |
| `reports/fau_router_run100_granite` | **granite4.1:3b** | **73%** | **98%** | $0 (100% FAU) |
| `reports/fau_router_run100_minicpm` | MiniCPM5-1B (corrected, schema off) | 7% | — | $0 — abandoned per user decision |

### Latest era (this session; `run_batched_parallel.py`, budget=9, 3 plan-review rounds, **hidden budget**, batched multi-episode/GPU)
No `REPORT.md` written yet for these — raw data only, listed here so it isn't lost:
| Run dir | Student | Answer-correct (cover-match) | Answer-correct (teacher-verdict >=0.40) | Unknown |
|---|---|---|---|---|
| `data/simulation_output/qwen0p8b_b9_run100/run` | qwen3.5:0.8b | 39% | not recomputed | 0 |
| `data/simulation_output/qwen2b_b9_run100/run` | qwen3.5:2b | 45% | not recomputed | 4 |
| `data/simulation_output/granite3b_b9_run100/run` | **granite4.1:3b** | 69% | **72%** | 0 |

Granite 4.1 3B is the clear best student across both eras. The qwen2b b9 run's 4 "unknown"
answers are a distinct, still-open bug — see §9.

### Budget-disclosure ablation (`reports/budget_mode_comparison`, n=10, qwen3.5:2b, plan=3/run=9)
Hidden budget beat disclosed budget: 3 vs 2 correct, 5 vs 4 grounded, 2 vs 1 natural
finishes, fewer mean steps (8.0 vs 8.4). This is why hidden-budget became the default for
the b9-era 100-sample runs above.

### 4-model x 5-setting experiment matrix (`reports/experiment_matrix_2026-07-03`)
30 fixed questions, teacher `glm-5.2`, students qwen 0.8b/2b/4b + ornith9b, settings A-E
(planning/running budget variants + teacher-authored-plan + no-teacher ablation). **Only
6/20 configs completed** — paused because invalid-JSON rate was high enough on small
models to make the comparison unfair before the JSON-schema-enforcement work (now done,
§4/§9) landed. Restarting this matrix cleanly is an open item.

---

## 9. Commit history on the branch (newest first, 145 commits total, base `159a21c`)

Highlights since the last handoff refresh (140 commits, base `3fd4c26`):
- **Forced-finish "unknown" fix**: when budget exhausts, if the student's own finish
  answer is "unknown", one extra free-text answer-extraction call is made before
  committing (`d2d4d5f`); `tool_executor.derive_final_answer` also reuses the last
  committed finish answer before ever falling back to "unknown" (`6b19790`).
- **Full JSON-schema-enforcement plan implemented**: Pydantic models as the single source
  of truth for grammar + validation (`c690527`, `671eecf`), `response_schema` threaded
  through the LLM client and call-logger (`6e58afb`, `53221a7`), schemas enforced on every
  agent-step/plan-review call (`978b8e1`), invalid-JSON-escape repair + `json_repair`
  fallback tier (`6b55535`).
- **Grammar-constrained-decoding flip and revert**: unconstrained-by-default to fix
  MiniCPM5 (`a7a3f6a`), then reverted to constrained-by-default per explicit user request
  once MiniCPM was abandoned (`f422af3`), while keeping everything built in between that
  doesn't touch schemas/tool/JSON formats (hidden-budget mode, forced-finish fix, viewer
  work, router hardening).
- **Hidden-budget mode** (`disclose_budget`, `c23b410`) — the student is told only "this
  is step N" until the forced final step; validated to reduce steps and improve
  correctness (`2124e01`).
- **Teacher-source-aware cost router** built up over many commits: FAU provider added
  (`2084bbd`, `150bfdd`), 429/503-aware backoff (`d6aa527`), full 3-tier fallback router
  (`145be4b`, `7036690`, `0465889`, `6724b98`), then hardened to skip unconfigured
  providers and fall through on *any* exception (`3436125`), and finally given a hard
  wall-clock timeout + circuit breaker for FAU's silent hangs (`8270cff`).
- **Distillation pipeline built**: path-optimality signals, trace-quality gate, canonical
  trace selection, SFT export/internalization, dataset split, leakage healthcheck
  (`5914259` … `112a75d`).
- **Teacher final-answer verdict** captured end-to-end: teacher prompt asks for a 0-1
  score vs gold (`16789eb`), captured (`8956bcd`), exported (`0489cc8`), shown in the
  viewer (`50186a1`), and finally promoted to be the explorer's actual correct/incorrect
  signal at a >=0.40 threshold (`5862e2e`, this session).
- **Viewer overhaul**: parallel-run consolidation (`0893563`), runs-table main page +
  teacher-source stats + response caching + private-by-default (`1c972b6`).
- **Batched multi-GPU runner** (`scripts/run_batched_parallel.py`, uncommitted script
  turned into a committed one alongside the router hardening in `8270cff`) — packs
  several episodes per GPU instead of one, for small students on 80 GB cards.

Run `git log --oneline 3fd4c26..HEAD` for the full 140-commit list, or
`git log --oneline` for the complete 145-commit branch history.

---

## 10. Open items / good next steps

1. **Voluntary "unknown" finishes the teacher accepts** — distinct from the forced-finish
   "unknown" bug (fixed, see §9). In the qwen2b b9 run, 4 episodes had
   `stop_reason=teacher_accept` at 2-7 steps (well under budget) with a *voluntary*
   student finish of "unknown" that the teacher then accepted. Two candidate fixes, not
   yet implemented: (a) extend the answer-extraction fallback to any finish resolving to
   "unknown", not just forced-finish ones; (b) teacher policy change to `reject_finish` on
   an empty/unknown answer so the student is pushed to keep working.
2. **FAU `gpt-oss-120b` backend instability** — the hard-timeout + circuit-breaker fix
   (§4/§9) stops a FAU hang from freezing a whole run, but the model itself does go fully
   unresponsive for extended periods on FAU's side (confirmed twice this session; other
   FAU-hosted models responded fine in under 1s during the same outage). No FAU-side fix
   is possible from here — just make sure the router's fallback keeps working.
3. **Restart the 4-model x 5-setting experiment matrix** (§8) now that JSON-schema
   enforcement + the router hardening are both done — the 6/20 configs completed
   pre-dated both fixes and aren't a fair comparison to a clean re-run.
4. **Exercise the distillation pipeline (§7) end-to-end** on a real generation corpus and
   actually fine-tune a student — it's built and unit-tested but has never processed a
   full run's worth of episodes or produced a trained model.
5. **Recompute/backfill `teacher_answer_score`-based correctness** for the b9-era runs
   that predate the >=0.40 signal becoming the explorer default (`qwen0p8b_b9_run100`,
   `qwen2b_b9_run100`) — the viewer computes it on the fly so no code change is needed,
   just re-check the numbers (granite's was already re-checked: 69% cover-match -> 72%
   teacher-verdict).
6. Guidance-level sweep (G=0..4) — templates exist (`hotpot_teacher_guidance_b5_g0..g4`),
   never run at 100-question scale.
7. Stronger local students beyond granite-4.1-3b, now that the pipeline reliably
   discriminates between them.

---

## 11. Gotchas to remember

- **Never attribute a commit to Claude/co-author it as Claude** — permanent rule from the
  user, applies to every commit in this repo, no exceptions.
- Don't regenerate a dataset directory if you want to compare against prior runs on it —
  `--shuffle` reshuffles; the existing file IS the fixed question set other results are
  pinned to.
- `.env` and `data/` are git-ignored — set up `.env` and regenerate data on a new box.
- OpenRouter `custom` endpoint must be `https://openrouter.ai/api` (no `/v1`); FAU's
  endpoint already includes `/v1` (client appends only `/chat/completions`) — don't add
  another `/v1` to `FAU_LLM_ENDPOINT`.
- **A hung FAU call looks like nothing is happening** (no error, no log line) for up to
  `FAU_TIMEOUT` seconds (45s default) per worker — this is expected now (it used to hang
  forever); if a whole run seems stuck at step 1 across every worker with zero step-2
  progress for several minutes, check `grep -i "hard timeout" <sim log>` before assuming
  something is actually broken.
- The explorer's "correct" column/badge is the **teacher verdict** (score >= 0.40), not
  cover-match, as of 2026-07-06 — a long-running `agentsim viewer` process must be
  restarted to see this (and any other data_access.py change) take effect.
- Leakage flags in episodes mean the guard **fired and sanitized** — verify with "is the
  gold actually visible to the student" (should be 0); detection != exposure.
- Nothing dataset- or gold-specific may go into any student-visible prompt (leakage rule).
- Set `PYTHONUTF8=1` on Windows for the CLI's `✓/✗` glyphs.
