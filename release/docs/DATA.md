# Data

## 1. Question files — `data/questions/<dataset>/`

| Dataset | Questions | Docs per question | Answer type | Source | License |
|---|---|---|---|---|---|
| HotpotQA | 2,000 | ~10 (distractor setting) | span | `hotpotqa/hotpot_qa` (HF), distractor, train | CC BY-SA 4.0 |
| 2WikiMultihopQA | 2,000 | 10 | span | `framolfese/2WikiMultihopQA` (HF), train | Apache-2.0 |
| MuSiQue | 2,000 | 20 | span | `dgslibisey/MuSiQue` (HF), train (answerable) | CC BY 4.0 |
| StrategyQA | 1,999 | ~12.5 (gold paragraphs + 8 seeded distractors) | yes/no | official release, train | MIT |

Each dataset was shuffled with seed 13 and truncated to 2,000 questions. One StrategyQA
question could not be converted (no gold paragraphs), hence 1,999.

`<dataset>_questions.jsonl.gz` — one question per line:

```json
{"id": "5ae11c0d55429920d52342c8", "query": "...", "answer": "Angelina Jolie Pitt",
 "type": "comparison", "level": "hard", "source": "hotpotqa", "split": "train",
 "num_hops": 2, "answer_type": "span", "gold_granularity": "sentence",
 "gold": {"answer": "...", "answer_aliases": [], "supporting_titles": [...],
          "supporting_facts": [...], "gold_doc_ids": ["<id>::doc4", "<id>::doc9"]},
 "retrieval_scope": {"backend": "hotpot_local", "doc_ids": [...]}}
```

`<dataset>_corpus.jsonl.gz` — the documents the agent can retrieve, one per line
(`doc_id`, `qid`, `title`, `text`). Retrieval is *per question*: the agent's `search`
tool only sees the documents whose `qid` matches the question (the standard
distractor setting), scored with BM25.

`<dataset>_manifest.json` — conversion statistics, source, license, file checksums (the
checksums are of the uncompressed content: `gunzip -c <file>.gz | sha256sum`).

Question and corpus files ship gzipped. Every reader in this project opens either form,
so `--questions .../x_questions.jsonl` and `.../x_questions.jsonl.gz` both work; the
documented commands use the `.gz` names that are actually on disk.

## 2. Episodes — `data/episodes/`

`episodes.jsonl.gz` holds 7,999 teacher-guided episodes, one per (dataset, question):
the student `ibm-granite/granite-4.1-3b` solved every question with budget 3 (hidden
from the student), a planning turn reviewed by the teacher, and the teacher scoring
every step (guidance level 3, "diagnostic feedback", strict leak policy). The teacher was
DeepSeek-V4-Flash — checkpoint `deepseek-ai/DeepSeek-V4-Flash` for 5,475 episodes and
`deepseek-ai/DeepSeek-V4-Flash-0731` for 2,524; every episode records which one in
`teacher_models_used`. Student temperature 0.2, teacher temperature 0.1. `index.jsonl` has one small row per episode; `stats.json` the totals:

| Dataset | Episodes | Correct final answer | Grounded |
|---|---|---|---|
| HotpotQA | 2,000 | 1,296 (64.8 %) | 1,802 |
| 2WikiMultihopQA | 2,000 | 1,385 (69.3 %) | 1,629 |
| MuSiQue | 2,000 | 595 (29.8 %) | 1,491 |
| StrategyQA | 1,999 | 1,111 (55.6 %) | 1,139 |

An episode record (abridged):

```
qid, dataset, query, gold_answer, budget, used_steps, stop_reason, final_answer,
final_metrics {answer_correct, exact_match, f1, supporting_doc_recall, answer_grounded, ...},
student_model, teacher_model, teacher_models_used, guidance_level, config_hash,
plan_review {initial_student_plan, rounds[{review_calls, revision_calls, ...}], revised_student_plan},
steps[ {t, student_prompt, student_raw, student_action, student_calls[{usage, elapsed_ms}],
        tool_observation, teacher_prompt, teacher_raw, teacher_calls[...],
        student_visible_guidance {score, feedback}, teacher_private_diagnosis,
        leakage_check, metrics, ...} ]
```

`teacher_prompt`, `teacher_raw`, `teacher_private_diagnosis` and the plan-review
equivalents **contain the gold answer** (the teacher sees it). They are kept because
they are the teacher's behaviour; the split builder never copies them into a training
target. What the student saw is `student_prompt` and `student_visible_guidance`, whose
feedback was passed through the harness's leakage gate (`[answer hidden]` masking).

Compared with the raw harness output, published episodes drop provider response bodies,
per-call cost, the provider fallback chain and the generating checkout's commit id, and
model ids are plain model names without provider routing prefixes (`tgd/episodes.py`;
`consolidate_episodes.py --rename-model` does the same for new collections).

## 3. Splits — `data/splits/`

Built by `scripts/build_splits.py` from the episodes and question files; audited by
`scripts/check_leakage.py` (`leakage_report.json`, status OK).

### Pool assignment

`sha256("<salt>:<qid>") mod 10000 < 1000` → **held-out test** (10 %), else
**trainable** (90 %). Salt `m1lodo`. A 3 % dev slice (salt `m1lodo-dev`) is carved
out of the trainable pool for loss monitoring only. `pools.json` lists every
`<dataset>/<qid>` with its pool.

| Dataset | Held-out (test) | Trainable | Correct trainable episodes → SFT examples |
|---|---|---|---|
| HotpotQA | 189 | 1,811 | 1,176 → 4,374 |
| 2WikiMultihopQA | 170 | 1,830 | 1,262 → 4,801 |
| MuSiQue | 203 | 1,797 | 517 → 1,938 |
| StrategyQA | 185 | 1,814 | 1,004 → 3,722 |
| **total** | **747** | 7,252 | 3,959 → 14,835 |

### Test files — `data/splits/test/`

* `heldout_<ds>_questions.jsonl` — the held-out 10 % of `<ds>` (the uniform test set is
  the union, 747 questions).
* `full_<ds>_questions.jsonl` — every question of `<ds>`, the "unseen dataset" test of
  the lodo fold that excludes `<ds>`.

Test files use every question regardless of whether the collection episode was correct.

### Group B — `data/splits/uniform/` (train on all four, test on the held-out 10 %)

| | examples | questions |
|---|---|---|
| train.jsonl | 14,458 | 3,959 |
| dev.jsonl | 377 | (3 % of the same questions) |

### Group A — `data/splits/lodo/fold_<ds>/` (train on the other three)

| Fold (unseen dataset) | train examples | dev | train questions | test sets |
|---|---|---|---|---|
| fold_hotpotqa | 10,180 | 281 | 2,783 | full_hotpotqa + heldout_{2wiki, musique, strategyqa} |
| fold_2wikimultihopqa | 9,801 | 233 | 2,697 | full_2wikimultihopqa + heldout_{hotpotqa, musique, strategyqa} |
| fold_musique | 12,563 | 334 | 3,442 | full_musique + heldout_{hotpotqa, 2wiki, strategyqa} |
| fold_strategyqa | 10,830 | 283 | 2,955 | full_strategyqa + heldout_{hotpotqa, 2wiki, musique} |

Each split directory has a `manifest.json` with these counts and the train/test id
overlap (0 everywhere).

### SFT example format ("guidance as internal thought")

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "<student prompt at step t, teacher block removed>"}],
 "completion": [{"role": "assistant", "content": "{\"teacher_guidance\": {\"score\": 0.8, \"feedback\": \"...\"}, \"thought\": \"...\", \"decision\": {...}, \"action\": {...}, \"new_facts_extracted\": [...]}"}],
 "metadata": {"qid": "...", "dataset": "...", "kind": "action|plan", "step": 2, "tool": "search", "query": "..."}}
```

* The input is the stored student prompt with the `Previous teacher guidance:` block
  removed, so training inputs match teacherless inference exactly.
* The target begins with a `teacher_guidance` block: the teacher's feedback on the
  *previous* step (step 1 receives the plan-review feedback), kept verbatim in the second
  person, followed by the student's thought and action. The model learns to produce the
  critique itself before acting. The runtime action parser ignores the extra key.
* Plan turns train `plan prompt → revised (teacher-approved) plan`.

Filters applied to the training side (counts in `stats.json`):

| Filter | Why | Dropped |
|---|---|---|
| episode not correct | a wrong trajectory teaches a wrong trajectory | 3,612 episodes (incl. held-out ones) |
| held-out pool | never train on test questions | 747 questions |
| target asserts the gold answer that never appeared in the prompt | "correct but ungrounded" — teaches answering without evidence | 119 examples |
| malformed target JSON | truncated generations teach an incomplete shape | 1 example |
| `[answer hidden]` placeholder anywhere in the example | never train on the mask token | 22 examples |
| question text duplicates a held-out question (different id) | dataset-level duplicate | 1 question |

### Leakage audit — `scripts/check_leakage.py`

For every split and every test set it is evaluated on, re-derived from the files:

1. no train/dev example comes from a test question id;
2. no test question text appears anywhere in the training prompts or completions;
3. no `[answer hidden]` token survives;
4. every train/dev id hashes to the trainable pool and every held-out question to the
   held-out pool (the split is the hash, not an edit);
5. every example is a prompt/completion message list with a JSON target carrying the
   required keys;
6. (reported) near-duplicate question texts between train and test.

Exit code 0 and `"status": "OK"` in `leakage_report.json` mean all hard checks passed.

## 4. Building the data yourself

```bash
# new episodes with any student/teacher (see docs/EVALUATION.md for serving the student)
python scripts/collect_episodes.py --datasets hotpotqa 2wikimultihopqa musique strategyqa \
    --num-samples 2000 --student vllm/student --teacher oai-teacher/<model> --shards 6 --out runs/collect
python scripts/consolidate_episodes.py --runs runs/collect --out data/episodes_new --gzip
python scripts/build_splits.py --episodes data/episodes_new/episodes.jsonl.gz --out data/splits_new
python scripts/check_leakage.py --splits data/splits_new
```

`build_splits.py --salt <string>` gives a different (equally valid) held-out set;
`--heldout-fraction` and `--dev-fraction` change the proportions. Adding a dataset means
adding `data/questions/<name>/<name>_{questions,corpus}.jsonl` in the format above and
passing `--datasets ... <name>`.
