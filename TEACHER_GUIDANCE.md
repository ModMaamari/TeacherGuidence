# Teacher Guidance pipeline

Generate step-level retrieval-agent trajectories on QA datasets (HotpotQA first):
a **student** model solves a question with tools while a **teacher** model privately
evaluates each step and exposes only a controlled `guidance_level` of feedback.

The student never sees the gold answer or teacher diagnosis; visibility is enforced in
code (the guidance renderer + leakage checker), not by prompting.

## What was added

```
agentsim/teacher_guidance/      schemas, json_utils, hotpot_converter, local_retrieval,
                                metrics, leakage, guidance_policy, prompts,
                                tool_executor, plan_review, episode_exporter
agentsim/components/control/    teacher_guided_agent_step, teacher_guided_plan_review
scripts/                        prepare_hotpot_teacher_guidance.py,
                                validate_teacher_guidance_run.py
templates/workflows/            hotpot_teacher_guided_b3|b5|b7[,_plan_review].yaml
templates/simulations/          hotpot_teacher_guidance_b5_g0..g4[, _g3_plan_review].yaml
tests/teacher_guidance/         unit + integration tests
```

---

## 1. Set up the system

### Python

Use **Python 3.10–3.12**. AgentSim depends on `sentence-transformers`/`torch`, which
do not yet ship wheels for very new interpreters (e.g. 3.14). The Teacher Guidance
modules themselves are dependency-light, but running a full simulation imports the LLM
client, which imports `sentence-transformers`.

### Install

With Poetry (preferred):

```bash
poetry install
```

Or with pip into a virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install pyyaml loguru python-dotenv httpx sentence-transformers rank-bm25 datasets pytest
pip install -e .
```

### API keys

Copy `.env.example` to `.env` and set the provider key for the models you choose.
The default templates use OpenAI models (`gpt-4o-mini` student, `gpt-4o` teacher):

```bash
cp .env.example .env
# edit .env:
OPENAI_API_KEY=sk-...
```

To use other providers, change `mode_config.student_model` / `mode_config.teacher_model`
in the simulation template (e.g. `claude-...`, `gemini-...`, `ollama/llama3`,
`custom/<model>`), and set the matching key/endpoint in `.env`.

### Verify the implementation (no API key needed)

```bash
python -m pytest tests/teacher_guidance -q
```

---

## 2. Prepare a HotpotQA dataset (10 random samples)

From the Hugging Face `datasets` library (downloads HotpotQA distractor validation):

```bash
python scripts/prepare_hotpot_teacher_guidance.py \
  --subset distractor --split validation --limit 10 --shuffle \
  --out_dir data/datasets/hotpot_teacher_guidance
```

Offline alternative — if you already have a raw HotpotQA JSON
(e.g. `hotpot_dev_distractor_v1.json`):

```bash
python scripts/prepare_hotpot_teacher_guidance.py \
  --input hotpot_dev_distractor_v1.json --split validation --limit 10 --shuffle \
  --out_dir data/datasets/hotpot_teacher_guidance
```

This writes:

```
data/datasets/hotpot_teacher_guidance/
  hotpot_distractor_validation_questions.jsonl   # student-facing tasks (+ private gold)
  hotpot_distractor_validation_corpus.jsonl      # per-question paragraphs
```

The templates ship with `num_samples: 10`. Change it in the simulation YAML to scale up.

---

## 3. Run an experiment

Validate the template, then simulate. Start with diagnostic feedback (G=3):

```bash
# (Poetry) prefix with `poetry run`, or just run inside your venv:
python -m agentsim.cli simulate hotpot_teacher_guidance_b5_g3 --validate-only
python -m agentsim.cli simulate hotpot_teacher_guidance_b5_g3
```

(With Poetry the console script is available too: `poetry run agentsim simulate ...`.)

Output lands under `data/simulation_output/hotpot_teacher_guidance_b5_g3/<run_uuid>/...`.

### Run the guidance-level sweep

Keep the student model, teacher model, retriever, question set, and budget fixed; vary
only the guidance level:

```bash
for G in 0 1 2 3 4; do
  python -m agentsim.cli simulate hotpot_teacher_guidance_b5_g$G
done

# plan-review variant (step guidance G=3, plan-review guidance level 3):
python -m agentsim.cli simulate hotpot_teacher_guidance_b5_g3_plan_review
```

---

## 4. Inspect outputs

Each sample directory contains the raw AgentSim traces plus the clean Teacher Guidance
views:

```
<run>/hotpot_questions/sample_001/
  traces.jsonl                      # raw step traces
  supervised.jsonl                  # one row per LLM call
  trajectories.jsonl                # RL-style transitions
  teacher_guidance_episodes.jsonl   # one clean row per question trajectory
  student_sft.jsonl                 # one row per student step (gold-free input)
  teacher_sft.jsonl                 # one row per teacher step
  student_visible_guidance.jsonl    # exactly what the student received
  plan_review_rows.jsonl            # present only when plan review is enabled
  teacher_guidance_metrics.json     # final answer / retrieval / evidence metrics
```

Peek at an episode:

```bash
python -c "import json,sys; print(json.dumps(json.loads(open(sys.argv[1],encoding='utf-8').readline()),indent=2))" \
  data/simulation_output/hotpot_teacher_guidance_b5_g3/*/hotpot_questions/sample_001/teacher_guidance_episodes.jsonl
```

(`jq .` works too if you have it.)

---

## 5. Validate the run (leakage / integrity)

```bash
python scripts/validate_teacher_guidance_run.py \
  --run-dir data/simulation_output/hotpot_teacher_guidance_b5_g3
```

It checks episode integrity, invalid span extraction, and leakage flags, and exits
non-zero if any student-visible guidance leaked hidden gold data.

---

## 5b. Explore trajectories in the web UI

After a run, launch the trajectory explorer to browse episodes, the plan-review phase,
and every step (with a toggle to reveal teacher-private data):

```bash
agentsim viewer --output-root data/simulation_output --port 8000
```

See [`TEACHER_GUIDANCE_UI.md`](TEACHER_GUIDANCE_UI.md) for the full UI guide.

## 5c. Planning options

The `plan_review` block supports three planning controls:

- **`planner`** — `student` (the student drafts the plan, the teacher reviews it) or
  `teacher` (the teacher writes the full plan for the student to follow; the plan is
  sanitized for leakage before the student sees it).
- **`planning_steps`** — for the student planner, the maximum number of
  review → revise rounds. The loop stops early as soon as the teacher returns
  `accept_plan`. `planning_steps: 1` reproduces the original single-round behaviour.
- **`formal_plan`** — when `true`, the finalized plan is treated as a formal, ordered
  list of steps. The current expected step is surfaced in the student prompt, and each
  executed action is checked against the plan's expected tool. Per-step
  `expected_tool` / `plan_step_followed` labels and an episode-level `plan_adherence`
  score are recorded. The student still chooses every action — this is a deterministic
  verifier, not a rewrite.

A ready-to-run example combining all three is
`hotpot_tg_b5_g3_planning_plus_openrouter`:

```bash
agentsim simulate hotpot_tg_b5_g3_planning_plus_openrouter
```

## 6. Guidance levels

| Level | Name | Student sees |
|---:|---|---|
| 0 | binary reward only | `0/1` score |
| 1 | continuous reward only | `0.0–1.0` score |
| 2 | outcome feedback | score + short note |
| 3 | diagnostic feedback | score + diagnosis (no next action) |
| 4 | scaffolded guidance | score + diagnosis + next-step hint |

Compare runs across levels using the metrics in `teacher_guidance_metrics.json` and the
per-step labels inside `teacher_guidance_episodes.jsonl` (exact match, F1, supporting
doc/fact recall, invalid action rate, leakage rate, stop reason).
