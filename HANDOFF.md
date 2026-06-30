# Teacher Guidance — Session Handoff

> Paste-this-into-a-new-session context for continuing development on another device.
> Repo: <https://github.com/ModMaamari/TeacherGuidence> (a fork of AgentSim).
> Active branch: **`feature/teacher-guidance`** (everything below is on it, pushed).

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

Central design rules (kept stable):
- **Student never sees the gold answer or teacher-private diagnosis.** Visibility is
  enforced in **code** (the guidance renderer + leakage checker), not by prompting.
- The **teacher** sees gold metadata; how much it can tell the student is gated by
  `guidance_level`.
- Tool execution is **deterministic** (no LLM); the environment owns the corpus and
  validates extracted spans against retrieved text.
- The final step is forced to `finish` when the budget is exhausted.

---

## 2. Where the code lives

```
agentsim/teacher_guidance/
  schemas.py          dataclasses: StudentAction, TeacherEvaluation, GuidanceConfig,
                      PlanReviewConfig (fields: enabled, planner, planning_steps,
                      formal_plan, review_guidance_level, ...), records, etc.
  json_utils.py       tolerant JSON extract/parse (+ repair: curly quotes, trailing
                      commas) and validate_student_action / validate_teacher_evaluation
                      (the latter flags a finish with no answer).
  hotpot_converter.py HotpotQA -> question rows + corpus rows (HF columnar or raw shape).
  local_retrieval.py  HotpotLocalRetriever (BM25 if rank_bm25 present, else lexical),
                      scoped to a question's candidate docs.
  metrics.py          normalize_answer ("&"->"and"), exact_match, cover_match (robust
                      bidirectional + prefix-token correctness), f1, doc/fact recall,
                      compute_step_metrics, compute_final_metrics (exact_match +
                      answer_correct + f1 + recalls).
  leakage.py          word-boundary detection + strict-policy sanitization.
  guidance_policy.py  render_student_guidance (levels 0–4) + derive_plan_review_guidance.
  prompts.py          build_student_prompt / build_teacher_prompt + plan prompts
                      (initial / review / revised / teacher-authored). Budget-aware.
                      Includes a neutral gold-free format example.
  tool_executor.py    execute_student_tool (decompose/reformulate/search/extract/verify/
                      synthesize/finish). clean_span + span_in_text (quote/whitespace/
                      case tolerant). derive_final_answer (never empty).
  plan_review.py      compute_plan_review_metrics.
  plan_execution.py   validate_formal_plan + PlanTracker (deterministic adherence).
  episode_exporter.py TeacherGuidanceEpisodeExporter -> clean per-episode files.
  viewer/             trajectory explorer web UI (see §6).

agentsim/components/control/
  teacher_guided_agent_step.py    one student tool-use step; student repair retry;
                                  formal-plan adherence; force-finish.
  teacher_guided_plan_review.py   preflight: student multistep loop OR teacher-planner;
                                  skips revision when teacher accepts.

Patched upstream files (additive, low risk):
  agentsim/simulation/modes/standard.py  injects mode_config (gold, models, guidance,
                                         plan_review_config, corpus_path, budget) into
                                         context.metadata when mode_config present.
  agentsim/workflow/executor.py          stops the workflow when a component returns
                                         verdict == FINISH.
  agentsim/simulation/loader.py          allows teacher_guided_agent_step as terminal comp.
  agentsim/cli.py                        `viewer` subcommand; episode export hook; creates
                                         output_dir before writing the checkpoint.
  agentsim/clients/llm_client.py         Ollama: think disabled by default (OLLAMA_THINK).
  agentsim/config.py                     OLLAMA_THINK option.

scripts/
  prepare_hotpot_teacher_guidance.py     build the dataset (HF or local raw json).
  validate_teacher_guidance_run.py       leakage/integrity validator (exit!=0 on leak).
  aggregate_teacher_guidance_run.py       mean EM/correct/F1/recalls, steps, stop reasons,
                                          invalid-JSON, leakage count.
  run_trajectory_viewer.py                launch the UI.

templates/workflows/   hotpot_teacher_guided_b3|b5|b7[,_plan_review].yaml, b10_plan_review
templates/simulations/ hotpot_teacher_guidance_b5_g0..g4[, _g3_plan_review];
                       hotpot_tg_b5_g3_plan_review_openrouter (cloud student);
                       hotpot_tg_b10_g3_plan_review_qwen_local (local student, B=10);
                       hotpot_tg_b5_g3_plan_review_qwen_local (local student, B=5);
                       hotpot_tg_b5_g3_planning_plus_openrouter (multistep+formal demo).

tests/teacher_guidance/  97 tests, all green.
```

Planning/scaffolding files that are intentionally **untracked** (never committed):
`TG-commits.md`, `TG-UI-commits.md`, `TG-UI-fixes-commits.md`, `TG-planning-commits.md`,
`TG-reliability-commits.md`, `TG-robustness-commits.md`, all `commitN_message.txt`, and
`.env`.

User docs (committed): `TEACHER_GUIDANCE.md` (setup/run/planning options),
`TEACHER_GUIDANCE_UI.md` (the viewer).

---

## 3. Environment & setup (IMPORTANT for the new device)

- **Use Python 3.10–3.12** (NOT 3.14 — `torch`/`sentence-transformers` have no wheels
  there, and a full `simulate` imports them). Original dev used **3.12** in a `.venv`.
- Install:
  ```bash
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1        # Windows; or source .venv/bin/activate
  python -m pip install -U pip
  pip install -e .                     # installs agentsim + deps incl. torch (CPU), st
  pip install datasets pytest          # datasets only needed to build the HotpotQA data
  ```
- **Run tests:** `python -m pytest tests/teacher_guidance -q`
  (On the original Windows box the default pytest temp dir errored; if so add
  `--basetemp=<some writable dir>`.)
- **Windows console is cp1252** and chokes on the `✓/✗` glyphs the CLI prints. Set
  `PYTHONUTF8=1` (and `PYTHONIOENCODING=utf-8`) before running `agentsim simulate`.
  Prefer running long jobs so stdout/stderr are merged to a UTF-8 log file.

---

## 4. Models & providers (how the LLM routing works)

The AgentSim `LLMClient` routes by model-id prefix. We use two providers at once:

- **Teacher = OpenRouter** via the OpenAI-compatible **`custom`** provider.
  Model id: `custom/z-ai/glm-5.2` (a reasoning model; returns fenced JSON — parser
  handles it). `.env`:
  ```
  CUSTOM_LLM_ENDPOINT=https://openrouter.ai/api      # NO /v1 — client appends it
  CUSTOM_LLM_API_KEY=sk-or-v1-...                     # user supplies; rotate/revoke freely
  ```
- **Student = local Ollama** (GPU), separate endpoint so it doesn't collide with the
  teacher's `custom` endpoint. Model id: `ollama/qwen3.5:4b`. `.env`:
  ```
  OLLAMA_ENABLED=true
  OLLAMA_ENDPOINT=http://127.0.0.1:11434
  # OLLAMA_THINK defaults to false — we disable Qwen3 "thinking" so it emits JSON
  # directly (else it wastes the token budget and can return empty).
  ```
  Original GPU: RTX 2000 Ada Laptop, **8 GB VRAM**; `qwen3.5:4b` is ~3.4 GB Q4, runs
  **100% on GPU** at ~63 tok/s. Install: `winget install Ollama.Ollama`, then
  `ollama pull qwen3.5:4b`. Ollama bundles its own CUDA, so the CPU-only venv torch is
  irrelevant for the student.
- Cloud student alternative used earlier: `custom/qwen/qwen3.7-plus` (OpenRouter).
- `.env` lives at the repo root (git-ignored). `agentsim/config.py` loads it.

---

## 5. How to run an experiment (end to end)

```bash
# 1. Build the dataset (10 fixed questions). DO NOT regenerate if you want to compare to
#    prior runs — the existing questions file IS the same 10 questions.
python scripts/prepare_hotpot_teacher_guidance.py \
  --subset distractor --split validation --limit 10 --shuffle \
  --out_dir data/datasets/hotpot_teacher_guidance

# 2. Validate + run a simulation template
agentsim simulate hotpot_tg_b5_g3_plan_review_qwen_local --validate-only
agentsim simulate hotpot_tg_b5_g3_plan_review_qwen_local

# 3. Inspect / score
agentsim viewer --port 8000
python scripts/aggregate_teacher_guidance_run.py --run-dir data/simulation_output/<dir>
python scripts/validate_teacher_guidance_run.py  --run-dir data/simulation_output/<dir>
```

Outputs land in `data/simulation_output/<sim_id>/<run_uuid>/hotpot_questions/sample_XXX/`
with raw traces plus clean files: `teacher_guidance_episodes.jsonl`, `student_sft.jsonl`,
`teacher_sft.jsonl`, `student_visible_guidance.jsonl`, `plan_review_rows.jsonl`,
`teacher_guidance_metrics.json`. **All of `data/` is git-ignored** (only READMEs tracked),
so run outputs do NOT transfer via git — regenerate them on the new device.

Key `mode_config.plan_review` knobs: `planner` (`student`|`teacher`), `planning_steps`
(multistep loop, stops early on `accept_plan`), `formal_plan` (programmatic adherence),
`review_guidance_level`. Guidance levels 0–4 controlled by `mode_config.guidance.level`.

---

## 6. The trajectory explorer UI (`agentsim viewer`)

Dependency-free (Python stdlib server + vanilla HTML/CSS/JS in
`agentsim/teacher_guidance/viewer/`). Lists every run under `--output-root`, shows
per-episode metric chips, the 3-phase plan review, and a step timeline. A **"Show
teacher-private"** toggle hides/reveals the gold answer + teacher diagnosis (defaults to
the student's view). Many viewer-side metrics (e.g. `answer_correct`) are recomputed on
the fly, so matching-rule improvements apply **retroactively** to old runs without re-running.

---

## 7. Experiments run so far (same 10 HotpotQA questions)

| Run | Student | Teacher | B | Answer-correct | EM | F1 | Empty ans | Notes |
|---|---|---|---|---|---|---|---|---|
| openrouter B5 | qwen3.7-plus (cloud) | glm-5.2 | 5 | 0.60 | 0.60 | 0.72 | 0 | first session |
| local B10 (pre-reliability) | qwen3.5:4b (GPU) | glm-5.2 | 10 | 0.50 | 0.00 | 0.05 | — | 8.5% invalid JSON |
| local B5 (pre-robustness) | qwen3.5:4b | glm-5.2 | 5 | 0.20 | 0.00 | 0.06 | 7/10 | tool-call bug |
| **local B5 (after fixes)** | qwen3.5:4b | glm-5.2 | 5 | **0.60** | 0.10 | 0.14 | **0/10** | matches cloud! |

**Biggest finding:** the local 4B's low score was a *pipeline/tool-call* problem, not a
capability gap. The 4B wrapped `extract` spans in literal quotes → `span_not_found` → no
facts → empty `synthesize`/`finish`. After the robustness fixes it reaches **0.60
answer-correct at B=5, equal to the cloud model**, with 0 empty answers. The local
student is free (GPU); only the teacher costs money (~$0.18–0.31/run on OpenRouter).
Note: EM/F1 stay low for the local model because it answers verbosely — `answer_correct`
(robust matching) is the metric that reflects reality.

---

## 8. Commit history on the branch (newest first)

41 commits, base `3fd4c26`. Highlights:
- Pipeline build: schemas → json_utils → converter → retrieval → metrics → leakage →
  guidance → prompts → tool executor → agent-step component → plan-review component →
  runner/executor patches → episode exporter → templates → docs.
- Viewer: data-access → stdlib server + CLI → frontend → docs → polish.
- Iterations from UI findings: robust `answer_correct` (bidirectional, prefix-token,
  `&`→`and`); word-boundary leakage; skip plan revision on accept; budget vs used-steps;
  multistep planning; teacher-planner; formal plan; local Ollama student; JSON repair;
  student repair-retry; format example; **tolerant extract**; **never-empty finish**;
  **budget-aware planning**.

Run `git log --oneline 3fd4c26..HEAD` for the full list.

---

## 9. Open items / good next steps

1. **B=10 local rerun with the robustness fixes** — current B=10 local data predates
   them; correctness should climb further (B=5 still hits `budget_forced_finish` 4/10).
2. **Guidance-level sweep** (G=0..4) to study reward-only vs diagnostic vs scaffolded
   supervision — templates exist (`hotpot_teacher_guidance_b5_g0..g4`).
3. **Scale up** beyond 10 questions (100 → 1,000); the converter/retriever already
   support it; just raise `num_samples` (and consider global BM25 / OpenSearch retrieval —
   currently `hotpot_local` is per-question candidate docs only).
4. Stronger local student (7B/8B) now that the GPU/Ollama path is set up.
5. Optional UI: a run-comparison view; surface `plan_adherence` and planning rounds.
6. The `derive_final_answer` last-resort returns `"unknown"` when the model extracted
   nothing — guarantees non-empty but counts as wrong; acceptable by design.

---

## 10. Gotchas to remember

- Don't regenerate the dataset if you want to compare to prior runs (reshuffles).
- `.env` and `data/` are git-ignored — set up `.env` and regenerate data on the new box.
- OpenRouter `custom` endpoint must be `https://openrouter.ai/api` (no `/v1`).
- Keep student (`ollama/`) and teacher (`custom/`) on different providers so endpoints
  don't collide.
- Set `PYTHONUTF8=1` on Windows for the CLI.
- Leakage flags in episodes mean the guard **fired and sanitized** — verify with
  "is the gold actually visible to the student" (it should be 0); detection != exposure.
- Nothing dataset- or gold-specific may go into any prompt (leakage rule).
