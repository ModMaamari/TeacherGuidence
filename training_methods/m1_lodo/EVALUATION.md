# m1-LODO — Evaluation Wiki

How the trained students are measured, what every number means, and where the raw
evidence for it lives. Companion to [`PLAN.md`](PLAN.md) (design) and the generated
`reports/m1_lodo/REPORT.md` (results).

---

## 1. What is being evaluated

**Not** next-token prediction on held-out text. Every test question is *solved by the
model itself*, in the same agent loop the corpus was collected with:

```
question
  → the model writes a plan
  → for each of B=3 steps:
        the model emits ONE JSON action
            search | extract | verify | decompose | reformulate | synthesize | finish
        the environment executes it deterministically against that question's own
        candidate paragraphs (BM25, no web, gold never shown)
        the observation is appended to the prompt
  → a final answer is committed (forced at the budget if not offered voluntarily)
```

**There is no teacher at evaluation time.** During collection a teacher critiqued every
step; here nothing external exists. A trained model must generate its own
`teacher_guidance` block — the critique it used to *receive* — and then act on it. That is
the hypothesis under test: the guidance has become internal.

Engine: `training_methods/common/eval_agent.py` → `common/hf_agent_loop.py`, the same
prompt renderer, tool executor and metrics used during collection.

---

## 2. Protocol (identical for every arm)

| Setting | Value | Why |
|---|---|---|
| Step budget | 3 | matches collection, so trained behaviour is in-distribution |
| Plan review | on | " |
| Budget disclosure | hidden | " |
| Decoding | greedy (`temperature 0`), fixed seed | reruns are comparable |
| Backend | vLLM, one server per GPU | base and adapters served from one loaded model |
| Retrieval | per-question local candidates | reproducible; no web variance |

**Arms are only ever compared within one backend.** vLLM's batching means greedy output is
not bit-identical to HF `generate`; mixing the two would make a 1–2 point difference
meaningless.

---

## 3. The evaluation matrix

Eight distinct test sets — a 10% **held-in** slice per dataset (never trained on) and the
**unseen** full dataset:

| Dataset | held-in | unseen |
|---|---:|---:|
| hotpotqa | 189 | 2,000 |
| 2wikimultihopqa | 170 | 2,000 |
| musique | 203 | 2,000 |
| strategyqa | 185 | 1,999 |

**24 configurations** (model × test set):

| Model | Test sets it runs | Episodes |
|---|---|---:|
| `fold_hotpotqa` | held-in of 2wiki, musique, strategyqa + **unseen_hotpotqa** | 2,558 |
| `fold_2wikimultihopqa` | held-in of hotpot, musique, strategyqa + **unseen_2wiki** | 2,577 |
| `fold_musique` | held-in of hotpot, 2wiki, strategyqa + **unseen_musique** | 2,544 |
| `fold_strategyqa` | held-in of hotpot, 2wiki, musique + **unseen_strategyqa** | 2,561 |
| `base` (untrained) | **all eight sets** | 8,746 |
| | | **18,986** |

### Why 40 jobs, not 24

Each `unseen_*` set is split into 3 shards (`--shard i/3`) so it can run on three servers
at once; held-in sets are small enough to run whole.

```
per fold : 3 held-in x1  +  1 unseen x3   =  6 jobs   → 24
base     : 4 held-in x1  +  4 unseen x3   = 16 jobs
                                             --------
                                             40 jobs
```

The report merges shards back per configuration, weighting each metric by episode count.

### Why the baseline costs 46% of the compute

`base` runs all eight sets because **a trained-model number alone is uninterpretable**.
These datasets ranged from 30% to 69% correctness at collection time, so an accuracy figure
reflects *which dataset was held out* as much as it reflects the model. Every cell in the
report is therefore a paired, same-questions comparison against base.

---

## 4. What is recorded

**Per step** (`episodes.jsonl`): full student prompt · **raw model output** ·
parsed action · action validity · tool observation · generation stats (tokens, latency).

**Per episode**: final answer · gold answer · plan · steps used · stop reason · elapsed
time · full metric block.

**Per run** (`metrics.json`):

| Group | Fields |
|---|---|
| Correctness | `em`, `f1`, `cover_match` |
| Grounding | `doc_recall` (supporting-document recall) |
| Behaviour | `mean_steps`, `stop_reasons`, `invalid_action_steps`, `total_steps` |
| Cost | `student_prompt_tokens`, `student_completion_tokens`, `tokens_per_episode`, `wall_time_s`, `gpu_peak_mem_gb` |

Nothing in the report is a number without a traceable episode behind it.

---

## 5. Reading the metrics

- **`cover_match` is the headline.** Exact match under-credits a correct answer wrapped in
  a sentence; F1 sits between. All three are reported and **none should be read alone** —
  this corpus has repeatedly shown them diverging (a verbose-but-correct answer scores
  cover 1.0, EM 0.0).
- **`doc_recall` separates two failure modes.** High recall with a wrong answer means
  retrieval worked and *composition* failed — the characteristic MuSiQue failure. Low
  recall means the model never found the evidence.
- **`invalid_action_steps` is a training outcome, not noise.** Format adherence improves
  sharply with SFT; on earlier HotpotQA work it fell from 51 to 2 on the dev set.
- **Fold results are never averaged.** Holding out MuSiQue is a much harder test than
  holding out 2Wiki; one mean would hide precisely the effect being measured.

---

## 6. The three comparisons the report makes

| # | Question | Comparison |
|---|---|---|
| 1 | Does training help **in-distribution**? | fold model vs base on `heldin_*` |
| 2 | Does it **transfer** to a dataset never seen? | fold model vs base on `unseen_<fold>` |
| 3 | How much is **dataset-specific**? | a fold's mean held-in vs its own unseen |

Comparison 2 is the point of the experiment. Comparison 3 is the honest caveat: a large
gap means the skill was largely dataset-specific even if 2 looks good.

---

## 7. Guarantees behind the numbers

- **No test question was ever trained on.** Splits are by `sha256(qid)`, so a question
  cannot leak between train and test through a second episode. `audit_folds.py`
  re-asserts disjointness on the built files and passes on all four folds.
- **No gold leaked into training.** The same audit checks that no target states a gold
  answer the prompt never showed, that no `teacher_private_diagnosis` text reaches a
  target, and that every target is well-formed JSON for its kind.
- **Only correct episodes were trained on**, and within those, examples asserting an
  ungrounded gold were dropped (see PLAN.md §3).

---

## 8. Running it

```bash
# full run (training + evaluation), resumable
sbatch training_methods/m1_lodo/run_full.sbatch

# evaluation only, once adapters exist
.venv_train/bin/python training_methods/m1_lodo/run_lodo.py \
    --stage eval --gpus 0,1,2 --out training_methods/m1_lodo/runs/lodo

# live status
bash training_methods/m1_lodo/watch.sh

# report
.venv_probe/bin/python training_methods/m1_lodo/build_lodo_report.py \
    --out training_methods/m1_lodo/runs/lodo --md reports/m1_lodo/REPORT.md
```

Resumption is by artifact: a finished eval shard leaves `<tag>.done`, a finished fold
leaves `adapter/adapter_config.json`. Re-running skips whatever exists, so an interrupted
run continues rather than restarting.

---

## 9. Known limitations

1. **One student model** (granite-4.1-3b) and **one teacher family** in the training data —
   this measures whether *this* guidance transfers, not whether the effect is universal.
2. **Budget 3 binds.** ~93% of collection episodes hit the budget, so both arms are
   measured under a tight step limit; a larger budget could change the ranking.
3. **The LLM-judge pass is not yet wired in.** Correctness here is deterministic
   (EM/F1/cover). A semantic judge would credit correct-but-differently-worded answers
   that cover-match misses.
4. **Contamination.** HotpotQA and 2Wiki are almost certainly in the base model's
   pretraining; MuSiQue is the most trustworthy transfer signal.
