# m1-LODO — Internalizing Teacher Guidance, Tested for Generalization

**Goal.** Train `granite-4.1-3b` so the teacher's guidance becomes the student's *own*
internal reasoning, then measure whether that skill **generalizes to a dataset it never
trained on**. No teacher at inference, ever.

**Status.** Plan v1, 2026-08-31. Awaiting review → smoke tests → full run.

---

## 1. What "m1" is, and why it should generalize

In a teacher-guided episode the teacher critiques each student step. m1 turns that
critique into the student's own first thought: each training target is

```jsonc
{"teacher_guidance": {...verbatim critique, second person...},   // generated FIRST
 "thought": "...", "decision": {...}, "action": {"tool": ..., "params": {...}}}
```

The student learns to *produce* the guidance it used to *receive*, then act on it. At
inference there is no teacher — the model writes its own guidance block and continues.
This is already implemented in `training_methods/common/episode_lib.py` and validated on
HotpotQA (18.1% → 67.5% cover-match, teacher-free). **This plan reuses that machinery
unchanged** and adds the leave-one-dataset-out design around it.

---

## 2. Experimental design

Four datasets: `d1=hotpotqa`, `d2=2wikimultihopqa`, `d3=musique`, `d4=strategyqa`.

### 2.1 Splits (qid-level, deterministic)

Per dataset, every qid is hashed once into one of two pools:

| Pool | Share | Purpose |
|---|---|---|
| `heldin_test` | 10% | **never trained on**; measures in-distribution performance |
| `trainable` | 90% | training pool (a 3% slice becomes `dev` for loss monitoring) |

The hash is `sha256(qid)`, so the assignment is identical across folds and reproducible.

### 2.2 The four folds

For fold *k* (held-out dataset *d_k*):

- **Train on** `trainable` episodes of the other three datasets — *correct episodes only*
- **Evaluate on four test sets**:

| Test set | Size | Question |
|---|---|---|
| `heldin_<da>` | ~200 | in-distribution, unseen questions (a ≠ k) |
| `heldin_<db>` | ~200 | " |
| `heldin_<dc>` | ~200 | " |
| **`unseen_<dk>`** | **~2,000** | **out-of-distribution: a whole dataset never seen** |

4 folds × 4 test sets = **16 evaluation runs**. Every test question is unseen by that
fold's model, by construction — the qid split guarantees it and a preflight check asserts
zero overlap between any fold's train qids and any of its test qids.

### 2.3 Baseline

The **untrained base model** is evaluated on all 16 test sets too, so every fold's numbers
have a same-questions reference. Without it a "65% correct" figure means nothing.

---

## 3. Training data

**Only correct episodes.** An episode qualifies when `final_metrics.answer_correct` is
true. From the 7,999-episode tg_v1 corpus that is ~4,387 episodes (54.8%), yielding
~17–18k examples (one plan example + one per step).

**Leakage controls**, all pre-existing in `episode_lib`:
1. The step target keeps the teacher's *student-visible* guidance only — never
   `teacher_private_diagnosis`.
2. `[answer hidden]` placeholders are restored **only when echo-safe** (the string already
   appears in the question, the student's own prior output, or a retrieved document);
   otherwise the sentence is dropped.
3. `leak_gate_ok` rejects any example whose text contains the gold answer without an
   echo-safe justification.
4. **A post-build audit** (new) re-checks every built example: no gold string in any
   prompt or completion that is not already present in the visible context.

**Class balance note.** MuSiQue contributes far fewer correct episodes (~595) than
HotpotQA (~1,296). We record per-dataset example counts and report them; we do **not**
resample, because the natural mix is what a practitioner would have.

---

## 4. Evaluation protocol

Every test episode runs the **teacherless** agent loop (`common/hf_agent_loop.py` via
`common/eval_agent.py`) with the same tools, retrieval and metrics as collection:

- budget 3, plan review on, budget hidden — identical to the collection protocol
- greedy decoding (temperature 0), fixed seed, batched generation
- vLLM backend for throughput; **arms are only ever compared within one backend**

### Metrics recorded per run

| Group | Fields |
|---|---|
| Correctness | exact match, F1, cover-match, LLM-judge verdict |
| Grounding | supporting-doc recall, supporting-fact recall, answer-grounded |
| Behaviour | steps used, stop reasons, voluntary-finish rate, invalid-action steps, tool mix |
| Cost | prompt/completion tokens, tokens per episode, wall time, GPU peak memory |

Per-episode records (`episodes.jsonl`) keep **raw model output, parsed action, tool
observation and the self-generated guidance block** for every step, so any number in the
report can be traced to the text that produced it.

**The judge.** Final answers are additionally scored by an LLM judge for semantic
correctness, because EM/F1 under-credit verbose-but-correct answers. Judge model: an
open-weight model on FAU (not gpt-oss — excluded by standing instruction).

---

## 5. Compute plan — 3 × H200 (141 GB)

The waste to avoid is running one 3B LoRA job on a 141 GB card. Two forms of parallelism:

**Training (4 jobs).** Each fold is LoRA r32 on a 3.4B model in bf16 — ~15–20 GB with
gradient checkpointing off, batch 8. Run **2 folds per GPU** (~40 GB), 4 folds across 2
GPUs, leaving the third for the base-model evals that can start immediately.

**Evaluation (17 × 4 = 68 runs incl. baseline).** One vLLM server per GPU with
`--enable-lora`, serving **all four fold adapters by name** from a single loaded base
model (LoRA adapters are ~100 MB each). Eval clients then hit the servers over HTTP with
high concurrency — vLLM continuous-batches them, so one server saturates a GPU far better
than one process per job.

Target: ≥80% GPU utilization during eval, measured by an `nvidia-smi` sampler that writes
`gpu_samples.csv` for the whole experiment.

**Everything runs inside a Slurm job** on the `gpu` partition (login node has no GPU).

---

## 6. Robustness

| Property | How |
|---|---|
| Resumable | Every stage writes a completion marker; the orchestrator skips finished folds/evals. Killing and restarting continues where it stopped. |
| Survives disconnection | `setsid` + Slurm batch, never tied to a session's process group (this bit us during collection). |
| Live monitoring | Per-stage `.log` with UTC-stamped lines, `status.json` updated after every job, and a `watch_m1_lodo.py` that prints a live table. |
| Deterministic | Fixed seeds; qid-hash splits; pinned model revision; config hash written per run. |
| Fail-fast preflight | Split disjointness, leak audit, template validity, GPU visibility and a 3-example smoke train + 2-episode smoke eval — all before the long run. |

---

## 7. Execution order

| # | Stage | Script | Gate before proceeding |
|---|---|---|---|
| 0 | Environments (`.venv_train`, `.venv_vllm`) | `setup_envs.sh` | imports succeed on a GPU node |
| 1 | Consolidate corpus | `scripts/consolidate_tgv1_episodes.py` | 7,999 episodes, 0 duplicates |
| 2 | Build folds | `build_folds.py` | disjointness + leak audit pass |
| 3 | Smoke | `--smoke` on train + eval | adapter loads, 2 episodes complete |
| 4 | Train 4 folds | `run_lodo.py --stage train` | 4 adapters + non-degenerate loss |
| 5 | Evaluate 68 runs | `run_lodo.py --stage eval` | all runs have metrics.json |
| 6 | Judge + report | `build_lodo_report.py` | report renders |

---

## 8. What this will and will not show

**Will show:** whether guidance-internalization transfers to an unseen multi-hop dataset,
separately for each of the four; and how much of the gain is dataset-specific versus
general, by comparing `unseen_dk` against `heldin_*` within the same fold.

**Will not show:** whether the *teacher's identity* matters (one teacher family here), or
whether internalization beats a live teacher at inference (that is the four-arm study
already run on HotpotQA). Both are separate experiments.

**Known confound to state in the report:** the four datasets differ in difficulty
(collection-time correctness: 2Wiki 69%, HotpotQA 65%, StrategyQA 56%, MuSiQue 30%), so a
fold that holds out MuSiQue is a harder test than one holding out 2Wiki. Fold results are
therefore reported per-fold and never averaged into a single "generalization score".
