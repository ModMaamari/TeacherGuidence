# How the pieces fit

```
                      data/questions/<ds>/{questions,corpus}.jsonl
                                     │
        scripts/collect_episodes.py  │  student (vLLM) + teacher (API)  ──►  runs/collect/
        scripts/consolidate_episodes.py                                 ──►  data/episodes/episodes.jsonl.gz
                                     │
        scripts/build_splits.py      │  pool = hash(qid)  ──►  data/splits/{uniform, lodo/fold_*, test/}
        scripts/check_leakage.py     │                    ──►  data/splits/leakage_report.json
                                     │
        scripts/train_sft.py         │  LoRA SFT          ──►  runs/train/<name>/adapter
        scripts/serve_vllm.sh        │  base + adapters on :8300
                                     │
        scripts/eval.py --arm student|guided|teacher      ──►  runs/eval/<arm>/<test-set>/episodes.jsonl
        scripts/judge.py             │  any judge API     ──►  runs/judge/verdicts.jsonl
        scripts/collect_results.py   │                    ──►  runs/results/{results.json, RESULTS.md}
```

## Code map

| Path | Role |
|---|---|
| `agentsim/` | the simulation harness: prompt renderer (`teacher_guidance/prompts.py`), tool executor and per-question BM25 retrieval (`teacher_guidance/tool_executor.py`, `teacher_guidance/local_retrieval.py`), teacher critic and plan review components (`components/control/`), leakage gate (`teacher_guidance/leakage.py`), metrics, the provider client (`clients/llm_client.py`, `config.py`), the `simulate` CLI used for collection |
| `tgd/hf_agent_loop.py` | the teacherless student loop (used by the student arms), batched over episodes |
| `tgd/guided_loop.py` | the guided and teacher-alone arms, driving the harness components with a hybrid client |
| `tgd/vllm_backend.py`, `tgd/mock_policy.py` | student policies: a vLLM server client; an offline mock |
| `tgd/episode_lib.py` | turns an episode into "guidance as internal thought" SFT examples, with the placeholder and leakage gates |
| `tgd/episodes.py`, `tgd/splits.py`, `tgd/metrics.py`, `tgd/io.py`, `tgd/logging_utils.py` | publishable episode view, hash split, aggregate metrics, JSONL I/O, logging |
| `templates/workflows/` | the harness workflow definitions per step budget; `templates/prompts/` prompt fragments |
| `templates/simulations/` | collection templates written by `collect_episodes.py` |

## Conventions

* Every long job takes a fixed `--out` directory, appends results as they land, writes
  `status.json`, and marks completion with `.done`. Re-running is always safe.
* Model ids carry their provider as a prefix (`docs/PROVIDERS.md`); the student inside
  the student arms is addressed by its served name or HF id instead.
* Logs are UTC-timestamped. Nothing under `runs/` is needed to rebuild anything else.
