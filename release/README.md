# Teacher-Guidance Distillation Kit

Train a small retrieval agent ("student") to internalise the step-by-step feedback of a
strong "teacher" model, and evaluate it against the teacher itself — on four multi-hop
QA datasets, with a leakage-audited split, resumable scripts, and an LLM judge.

The kit ships the data, the harness, and one command per stage:

| Stage | Command | Needs |
|---|---|---|
| (optional) collect new teacher-guided episodes | `scripts/collect_episodes.py` | student server + teacher API |
| consolidate episodes | `scripts/consolidate_episodes.py` | CPU |
| build train/test splits | `scripts/build_splits.py` | CPU |
| audit the splits for leakage | `scripts/check_leakage.py` | CPU |
| train a student (LoRA SFT) | `scripts/train_sft.py` | 1 GPU |
| serve a student (base + adapters) | `scripts/serve_vllm.sh` | 1 GPU |
| evaluate an arm | `scripts/eval.py --arm student\|guided\|teacher` | GPU / API |
| judge final answers | `scripts/judge.py` | judge API |
| results tables + significance | `scripts/collect_results.py` | CPU |

Every long-running script writes to a fixed directory, appends results as they land,
skips finished work on re-run, and keeps a `status.json` you can watch.

## What is in the box

```
data/questions/<ds>/     2,000 questions each for HotpotQA, 2WikiMultihopQA and MuSiQue,
                         1,999 for StrategyQA, with their per-question document sets
                         (gzipped, with provenance manifests)
data/episodes/           7,999 teacher-guided episodes (student granite-4.1-3b,
                         teacher DeepSeek-V4-Flash), gzipped, plus an index
data/splits/             test question files, pool assignment, stats, leakage report.
                         The SFT train/dev files are NOT shipped: build them once with
                         `make data` (~2 min, byte-identical on every machine)
agentsim/                the simulation harness (prompts, tools, teacher critic, metrics)
tgd/                     library code shared by the scripts
scripts/                 the stage commands above
slurm/                   sbatch templates + a one-command pipeline for HPC clusters
tests/                   unit tests, an offline end-to-end smoke test, a GPU smoke test
docs/                    OVERVIEW, DATA, TRAINING, EVALUATION, PROVIDERS, REPRODUCE, RESULTS
```

## The four evaluation arms

All arms answer the **same questions** with the **same corpus, tools, step budget and
metrics**; only the agent differs.

| Arm | Agent | Teacher at inference | Compute |
|---|---|---|---|
| base student | the untrained student | none | local GPU |
| guided student | the student, with the teacher reviewing its plan and every step | yes | local GPU + API |
| teacher alone | the teacher model is the agent | — | API |
| trained student | the student after LoRA SFT on teacher-guided episodes | none | local GPU |

## Two split groups, one held-out set

* **uniform** — train on the trainable 90 % of every dataset; test on the held-out 10 %
  of every dataset (747 questions). *Does internalised guidance work?*
* **lodo** (leave-one-dataset-out) — four folds; each trains on three datasets and is
  tested on the entire fourth dataset (never seen in any form) plus the held-out 10 % of
  the three training datasets. *Does it transfer to an unseen dataset?*

The 10 % held-out pool is a salted hash of the question id, so it is identical on every
machine and is excluded from every split. `scripts/check_leakage.py` re-derives the
proof from the files (id disjointness, question-text containment, placeholder hygiene,
hash consistency). See `docs/DATA.md`.

## Quick start

```bash
bash setup_env.sh                      # .venv (CPU), .venv_train (GPU), .venv_vllm (GPU)
make data                              # build the SFT train/dev files (~2 min; required before training)
cp .env.example .env                   # add your teacher / judge endpoint + key
.venv/bin/python -m pytest tests -q    # unit tests, seconds
bash tests/smoke_offline.sh            # whole pipeline with mock models, no GPU/API, ~1 min

# reproduce the four-arm comparison on the uniform split (Slurm):
TEACHER=oai-teacher/<model> JUDGE=oai-judge/<model> bash slurm/run_pipeline.sh -p <gpu-partition>
```

Without Slurm, run the same stages by hand — `docs/REPRODUCE.md` lists every command.

## Bring your own models

* **Student**: any Hugging Face causal LM with a chat template (`--model`); served by vLLM
  or in-process by transformers.
* **Teacher / judge / any API model**: a provider-prefixed id. `oai-<name>/<model>` talks to
  any OpenAI-compatible endpoint configured as `OAI_<NAME>_BASE_URL` / `OAI_<NAME>_API_KEY`;
  EdenAI, NVIDIA NIM, OpenRouter-style, Ollama and the native OpenAI/Anthropic/Google SDKs
  are also wired. Comma-separated ids form a fallback chain. See `docs/PROVIDERS.md`.
* **Mock**: `mock/<anything>` is an offline stand-in used by the smoke tests.

## Reference results

With granite-4.1-3b as student, DeepSeek-V4-Flash as teacher and Kimi-K2.6 as judge, on
the 747 held-out questions (judge-correct): base 29.8 %, guided 60.5 %, trained 65.5 %,
teacher 72.3 %. The trained student needs no teacher at inference and 0.43× the tokens
of the guided student. Full tables, per-dataset numbers, the leave-one-dataset-out
transfer results and confidence intervals are in `docs/RESULTS.md`.

## Requirements

Python ≥ 3.11. One GPU with ≥ 40 GB for the 3B student (training with LoRA and 8k-token
sequences peaks around 26 GB; smaller students need less). The CPU environment is enough
for data building, the teacher-alone arm, judging and results.

Disk: about 30 GB. The three virtual environments take ~15 GB (the CUDA and vLLM wheels
dominate), the model cache ~7 GB for a 3B student, the repository ~230 MB, and the built
SFT files ~1.5 GB. Set `HF_HOME` if your home directory is small or quota'd.

## License and attribution

The code in this repository is licensed under the **Apache License 2.0** (`LICENSE`),
which includes the `agentsim/` harness vendored here.

The data keeps its own terms. Question sets and corpora derive from HotpotQA
(CC BY-SA 4.0), 2WikiMultihopQA (Apache-2.0), MuSiQue (CC BY 4.0) and StrategyQA (MIT);
each dataset's manifest next to its question file records source, homepage and license.
The episodes were generated with the models named in every record's `student_model` and
`teacher_models_used` fields. `NOTICE` summarises all of it.
