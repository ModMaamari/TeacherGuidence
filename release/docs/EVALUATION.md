# Evaluation

## The task

Multi-hop question answering with tools. The agent sees the question and, at each step,
its previous actions, document previews and extracted facts; it emits one JSON action
(`search`, `extract`, `verify`, `decompose`, `reformulate`, `synthesize`, `finish`) and
gets the tool's observation back. Retrieval is BM25 over the question's own document set
(the distractor setting). Before the first step it writes a short plan. The budget is 3
tool steps, **hidden** from the agent (it is told to finish as soon as it can); at the
last step a `finish` is forced. Every arm runs this exact loop.

## The four arms

| Arm | `scripts/eval.py` | What acts | Who reviews |
|---|---|---|---|
| base student | `--arm student --served-model student` | the untrained student | nobody |
| trained student | `--arm student --served-model <adapter-name>` | the LoRA-tuned student | nobody |
| guided student | `--arm guided --teacher <id>` | the student | the teacher reviews the plan and every step; the student sees its score + feedback and may revise its plan; the teacher can reject a premature `finish` |
| teacher alone | `--arm teacher --agent-model <id>` | the teacher model | nobody |

The guided and teacher-alone arms are driven by the harness's own components, so the
prompts and stop semantics are identical to the collection runs that produced the
training data; the student arms use the same prompt renderer and tool executor with a
local policy.

## Metrics

Accuracy, from `final_metrics` of every episode and the judge verdicts:

| Metric | Meaning |
|---|---|
| EM | normalised exact match of the final answer with the gold |
| F1 | token F1 with the gold |
| cover | the normalised gold string is contained in the answer (aliases count) |
| judge | an LLM judge reads question, gold and answer and says correct / incorrect |
| doc recall | fraction of the gold supporting documents the agent retrieved |

EM and F1 punish verbose answers; cover rewards them; the judge is the primary metric
and the others are reported alongside so its behaviour can be checked. The judge never
sees the trajectory or the arm.

Efficiency and cost, from the per-call records inside each episode:

| Metric | Meaning |
|---|---|
| steps | tool steps used (≤ budget) |
| voluntary finish | episodes that ended with the agent's own `finish` (or a teacher-accepted one) rather than the forced last step |
| invalid steps | steps whose action failed to parse or validate |
| student / teacher / plan tokens | prompt + completion tokens per episode, by role and phase |
| latency | wall time per episode inside a concurrent run |
| API $ | the per-call cost reported by the provider, when it reports one |

`scripts/collect_results.py` adds paired statistics: for every pair of arms on the same
questions, a paired bootstrap 95 % CI of the difference in judge-correct (10,000
resamples) and an exact McNemar test on the discordant pairs.

## Running the arms

Serve the student once, then evaluate as many arms and test sets as you like against it:

```bash
# 1. serve base + adapters (one GPU; keeps running in the foreground)
scripts/serve_vllm.sh --lora uniform=runs/train/uniform/adapter &

# 2. arms that use the local student
T=data/splits/test; Q=data/questions
for ds in hotpotqa 2wikimultihopqa musique strategyqa; do
  .venv_train/bin/python scripts/eval.py --arm student --served-model student \
      --questions $T/heldout_${ds}_questions.jsonl --corpus $Q/$ds/${ds}_corpus.jsonl.gz --out runs/eval/base/heldout_$ds
  .venv_train/bin/python scripts/eval.py --arm student --served-model uniform \
      --questions $T/heldout_${ds}_questions.jsonl --corpus $Q/$ds/${ds}_corpus.jsonl.gz --out runs/eval/trained_uniform/heldout_$ds
  .venv_train/bin/python scripts/eval.py --arm guided --served-model student --teacher oai-teacher/<model> --concurrency 6 \
      --questions $T/heldout_${ds}_questions.jsonl --corpus $Q/$ds/${ds}_corpus.jsonl.gz --out runs/eval/guided_base/heldout_$ds
done

# 3. the teacher alone (no GPU; runs anywhere)
for ds in hotpotqa 2wikimultihopqa musique strategyqa; do
  .venv/bin/python scripts/eval.py --arm teacher --agent-model oai-teacher/<model> --concurrency 4 \
      --questions $T/heldout_${ds}_questions.jsonl --corpus $Q/$ds/${ds}_corpus.jsonl.gz --out runs/eval/teacher/heldout_$ds
done

# 4. judge everything, then tables
.venv/bin/python scripts/judge.py --judge oai-judge/<model> --episodes 'runs/eval/*/*/episodes.jsonl' --out runs/judge
.venv/bin/python scripts/collect_results.py --runs runs/eval --judge runs/judge/verdicts.jsonl --out runs/results
```

The `runs/eval/<arm>/<test-set>/` layout is what `collect_results.py` expects; arm names
are free text. On Slurm the `slurm/eval_*.sbatch` templates do the same per job and
`slurm/run_pipeline.sh` chains all of it (see `docs/REPRODUCE.md`).

### Leave-one-dataset-out (group A)

Serve the four fold adapters and evaluate each on its unseen dataset and on the
held-out sets of its three training datasets:

```bash
scripts/serve_vllm.sh --lora fold_hotpotqa=runs/train/fold_hotpotqa/adapter --lora fold_musique=... &
.venv_train/bin/python scripts/eval.py --arm student --served-model fold_musique \
    --questions data/splits/test/full_musique_questions.jsonl --corpus data/questions/musique/musique_corpus.jsonl.gz \
    --out runs/eval/fold_musique/full_musique
# + heldout_hotpotqa / heldout_2wikimultihopqa / heldout_strategyqa for the same adapter
```

A fold's number on `full_<ds>` is transfer to a never-seen dataset; the same fold's
numbers on `heldout_<other>` are in-distribution. `collect_results.py` pairs arms on the
test sets they share, so `base` evaluated on `full_<ds>` gives the paired baseline.

## Resumability, monitoring, robustness

* `--out` is fixed; `episodes.jsonl` is appended per finished question; re-running the
  same command skips finished questions and recomputes `metrics.json`; `.done` marks a
  complete run. A failed question (API error after retries) is logged, left out, and
  retried on the next run; the exit code is 3 while any question is missing.
* `status.json` (done / total / rate / ETA) is rewritten after every episode;
  `eval.log` carries UTC timestamps.
* Teacher and judge ids accept fallback chains; provider timeouts are capped so a hung
  endpoint fails over instead of stalling the run.
* `--shard i/n` splits a question file across processes or GPUs; the shards write to
  different `--out` directories and `collect_results.py` treats each directory as a run
  (merge shards by concatenating their `episodes.jsonl` into one directory).
* `--limit N` and `mock/*` models make dry runs cheap (`tests/smoke_offline.sh`).

## Fairness checklist

* Same question files, corpora, tool executor, prompts, budget and hidden-budget setting
  for every arm.
* Greedy student decoding (T = 0, seed 13); teacher temperature 0.1.
* The trained student never saw a test question in any form (`docs/DATA.md`).
* The judge sees only question, gold and answer; EM/F1/cover are reported next to it.
* Tokens are counted per role and phase; compare arms on total tokens, not on latency,
  when hardware differs.
* Report the teacher-alone arm: it is the ceiling the student was distilled toward.
