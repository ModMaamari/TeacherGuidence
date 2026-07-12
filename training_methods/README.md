# training_methods — Five ways to train the student on teacher-guidance data

Goal: fine-tune `granite4.1:3b` (HF `ibm-granite/granite-4.1-3b`) so that the teacher's
step-by-step guidance becomes the student's OWN internal reasoning, using the 18k
teacher-guided episodes collected in `data/simulation_output/traces_g3b_*` (six budget
runs × 3000 HotpotQA train questions) plus the best-answer datasets in
`data/datasets/best_answer_v1/`.

## The five methods

| dir | method | data it uses | GPUs | full cost | run order |
|---|---|---|---|---|---|
| `m1_sft` | guidance-as-internal-thought SFT | ds1 + ds4 + b5/b9 accepts (18.5k examples) | 1 | ~7 h | **1st — the foundation** |
| `m3_kto_dpo` | step-level KTO + same-prompt DPO | 37k scored steps + 3.6k pairs | 1 | ~12–16 h | 2nd (stacks on m1) |
| `m2_rft` | iterative rejection-sampling (STaR/ReST) | fresh unused train questions | 1–4 | ~14 h/round (2.5 h rollouts on 4 GPUs) | 3rd (uses m1/m3 policy) |
| `m4_grpo` | online GRPO with verifiable reward | question pool, on-policy rollouts | 1 | ~20 h/100 updates | 4th (only if 1–3 plateau) |
| `m5_rlaif_prm` | PRM distilled from 64k teacher judgments + PRM-reward RL | all scored steps | 1–2 | PRM ~7 h + RL ~15 h | 4th-alt (dense-reward RL; PRM alone is useful for reranking) |

Each method folder contains: `build_dataset.py` (datasets are BUILT and verified),
training script(s), `run_pipeline.sh` (dataset → train → eval → golden test → results
report; `SMOKE=1` validates it in minutes), `REPORT.md` (design + measured stats +
risks) and `GUIDE.md` (step-by-step commands).

## Shared infrastructure (`common/`)

- `episode_lib.py` — internal-thought example assembly: teacher feedback kept verbatim
  (second person) as a `teacher_guidance` JSON key generated BEFORE thought/action;
  `[answer hidden]` placeholders restored only when echo-safe (answer already in the
  question / the student's own output / retrieved docs) else dropped; leakage gates.
- `hf_agent_loop.py` — teacherless rollout/eval engine reusing the harness's exact
  prompt renderer, tool executor, per-question retrieval and metrics, driven by a HF
  policy (base or base+LoRA).
- `eval_agent.py` — evaluation CLI (timestamped artifacts: episodes.jsonl, metrics.json).
- `build_golden100.py` — the **golden test**: 100 of the 147 questions never answered
  correctly by ANY teacher-guided run (265 hard − 118 recovered). Base-model score is
  ~0 by construction; hits here are the strongest evidence of real generalization.
- `compare_evals.py` — merges eval metrics into markdown results tables.
- `teacher_eval_agent.py` — teacher-IN-loop eval arm: same HF student policy, but the
  real teacher (gpt-oss-120b via the FAU→OpenRouter router) reviews the plan and every
  step through the actual harness components; `--concurrency N` overlaps teacher API
  waits across episodes (GPU generation stays serialized).
- `run_eval_sharded.sh` + `merge_eval_shards.py` — split any eval across several GPUs
  (`GPUS=1,2,3 run_eval_sharded.sh <eval|teacher> <out> <tag> [args]`) and merge the
  shards into one `episodes.jsonl` + `metrics.json`.

**Eval speed:** `eval_agent.py --batch-size N` (and `m2_rft/generate_rollouts.py
--batch-size N`) run N episodes in lockstep with batched generation — ~2.5× per GPU at
N=8 with identical semantics (greedy answers can differ cosmetically under batched
padding). All `run_pipeline.sh` evals default to `EVAL_BATCH=8`; m2 rollouts to
`ROLLOUT_BATCH=8`; the m1 orchestration also takes `TEACHER_CONCURRENCY` (default 3).
Combine with sharding for another ~linear factor.

**vLLM serving backend (the fast path for heavy eval campaigns):** `.venv_vllm`
holds a vLLM install isolated from the training stack. `serve_vllm.sh` starts an
OpenAI-compatible server per GPU (bf16, the model's own chat template, LoRA adapters
served unmerged by name via `--enable-lora`); both eval agents take
`--backend vllm --server-url ... --served-model <student|adapter-name>` and keep the
same prompts, sampling params (per-request `seed` for reproducible reps) and token/
time instrumentation (usage comes from the server). The server continuous-batches all
concurrent episodes, so teacher-arm student calls no longer serialize on a GPU lock
and several eval jobs can share one server (`run_experiment.py --backend vllm
--clients-per-server N` starts the servers itself and runs jobs as HTTP clients).
Fairness notes: bf16 weights (no quantization distortion vs the HF path), same chat
template, but vLLM's kernels/batching mean greedy outputs are not guaranteed
bit-identical to HF `generate` — compare arms within one backend, and report tokens
(hardware-independent) alongside wall time.

## Environment

- `.venv_train` (Python 3.11): torch 2.6 cu124, transformers 5.13, trl 1.8, peft 0.19,
  datasets 5.0 + the agentsim deps. Kept separate from the inference `.venv`.
- Hardware budget: 4× A100 80 GB. Every trainer fits on ONE A100 (3.4B bf16 + LoRA);
  only m5's RL stage wants two (policy + PRM). m2 rollouts and all evals shard across
  GPUs with `--shard i/n`.
- All artifacts are timestamped (`runs/<UTC-ts>_<tag>/`) with `.log` files carrying
  per-line UTC timestamps; datasets live in `<method>/data/` (git-ignored, rebuildable
  from the scripts).

## Verification status (2026-07-11, this machine)

**All five `run_pipeline.sh` scripts passed their full SMOKE runs end-to-end**
(dataset → train → dev eval → golden-100 eval → results report), plus:

- m1: dataset built (17,961/544, 0 leaks) · trainer loss 14→2 · adapter reload in loop
- m3: datasets built (37,396 KTO / 3,639 DPO) · KTO + DPO trainers · init-adapter merge
- m2: fresh-question prep from the 90,447-question HF cache · rollouts · filter+merge
- m4: question pool (2,913) · GRPO cycle (rollouts → group advantages → update → save)
- m5: PRM dataset (58,218/1,782) · PRM trainer · digit-scorer correlation eval (smoke
  PRM already AUC 0.77) · two-GPU RLAIF stage via `--reward-fn`
- golden-100 built (64 medium / 27 hard / 9 easy; 94 bridge)

The SMOKE-trained adapters are throwaways (8 optimizer steps); real numbers come from
the full pipeline runs.

## Recommended experiment sequence

1. `GPU=0 bash training_methods/m1_sft/run_pipeline.sh` → the baseline-vs-trained and
   golden-100 tables.
2. `GPU=0 bash training_methods/m3_kto_dpo/run_pipeline.sh` (auto-stacks on m1).
3. One m2 round from the best of (m1, m1+KTO).
4. m4/m5 only if the offline methods plateau below target — and read the reward-
   hacking sections of their REPORTs first.

Evaluation note: these evals run the HF bf16 model greedily inside the training
harness; absolute numbers differ slightly from the Ollama (Q4_K_M) b-run tables —
always compare within the same harness.
