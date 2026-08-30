# Teacher-Guidance — TODO to finish the work

Companion to `TEACHER_GUIDANCE_STATUS_REPORT.md`. Ordered by dependency: **P0 unblocks
everything**, P1 is the corpus, P2 is the science, P3 is the release.
`[ ]` = not started · `[~]` = partially done · `[x]` = done, listed for context.

---

## P0 — Unblock: get the project running again on Frigga (est. 2–4 days)

- [ ] **Decide the compute plan on Frigga.** Confirm the current Slurm partitions (they change —
      `innkube` disappeared on 2026-08-12; never trust a stored partition name), GPU type/count,
      per-job time limits, and whether outbound HTTPS to FAU / EdenAI / OpenRouter is reachable
      **from compute nodes** (not just the login node). *Teacher calls are the whole pipeline —
      if compute nodes are network-isolated, the generation design has to change.*
- [ ] **Rebuild the three environments** (none exist in the tree; login node is Python 3.13 and
      unsupported):
      - `.venv` — Python 3.10–3.12, `pip install -e .` + `datasets pytest`
      - `.venv_train` — Python 3.11: torch 2.6 cu124, transformers 5.13, trl 1.8, peft 0.19, datasets 5.0
      - `.venv_vllm` — isolated vLLM install for serving
- [ ] **Run the test suite** — expect ~471 test functions green. Any failure here is a migration
      artifact, fix before anything else.
- [ ] **Fix hard-coded paths.** Run manifests and configs reference `/root/DeKIS/teacher-guidence/...`;
      grep and repoint to `/shared/al-maamari/DeKIS/teacher-guidence`.
- [ ] **Index and stage `data/simulation_output.tar` (21 GB).** Produce
      `data/simulation_output.INDEX.txt` (`tar -tf`, one pass, saved) so runs can be located
      without re-scanning; then extract only what is needed (`traces_g3b_b*_3000`,
      `traces_oss120b_teacheronly_3000`, cross-student runs). Keep the tar as the archive.
- [ ] **Re-probe every model.** `scripts/probe_models.py` — teachers *and* `--students`. Last probe
      (2026-08-02) had glm-5.2/EdenAI at HTTP 500 and every OpenRouter key out of credit. Record
      the result; it decides the teacher roster.
- [ ] **Commit the working tree.** ~40 untracked/modified files, incl. `scripts/analyze_wiki_ab_v2.py`,
      `scripts/run_hard_questions_xr.py`, `run_wiki_ab_v2_suite.sh`, 32 `hardq_xr*` templates,
      `reports/wiki_ab_v2/`, `reports/token_analysis_g3b_3000/`. *No Claude co-author lines.*
- [ ] Refresh `HANDOFF.md` — it is 8 weeks stale (says the 4×5 matrix is 6/20 and the distillation
      pipeline has never been exercised; both are now false).

## P0.5 — Small fixes worth doing before spending GPU-hours

- [ ] **Voluntary-`unknown` finish bug.** Extend the answer-extraction fallback to *any* finish
      resolving to `"unknown"` (not just forced ones), and/or make the teacher `reject_finish` on
      an empty/unknown answer. Add a regression test. *This silently costs correctness in every
      run.*
- [ ] **Fix `analyze_wiki_ab_v2.py::ep_stats`** to match the real op vocabulary
      (`ADD/EDIT/DEL/ANSWER/NEXT/KEEP`); regenerate `reports/wiki_ab_v2/stats.json`.
- [ ] **G7 — global teacher rate limiter.** A process-wide token-bucket in front of
      `get_completion_with_fallback`, shared across workers, tuned to each provider's measured
      ceiling. Without it, a 24+-way run throttles the free gateway and silently falls through to
      paid providers.
- [ ] **G5 — config registry.** Write `resource_paper/configs/*.yaml`, one per matrix cell, so
      `config_hash` maps to a named, reviewable config.

## P1 — Build the corpus (the resource paper's long pole)

### P1.1 Datasets (est. 3–5 days)

- [x] Converters for HotpotQA / 2Wiki / MuSiQue / StrategyQA, with golden-file + retrieval-parity tests
- [ ] **Download StrategyQA's paragraph corpus** and run its first end-to-end conversion —
      the only source never converted. Verify `constructed_candidates` behaves and that boolean
      answers score sanely (EM/F1/cover were designed for spans).
- [ ] **Prepare `data/datasets/tg_v1/` at scale** — `scripts/prepare_dataset.py` per source, with a
      fixed seed and the manifest committed:
      HotpotQA (train, ≥6k) · 2WikiMultihopQA (train, ≥6k) · MuSiQue-Ans (train, ≥6k) ·
      StrategyQA (train, all ~2.7k). Question count must cover
      `12 combinations × (per_combination − anchor)` + anchor.
- [ ] **Run `scripts/smoke_datasets.py --data-root data/datasets/tg_v1`** (offline) — must be 100%
      clean: schema, unique qids, candidate docs present, **gold reachable by the real retriever**.
- [ ] **Per-dataset difficulty/hop profiling report** at full scale (hop counts, answer types,
      candidate-set sizes) — needed for the paper's corpus-analysis section.
- [ ] Prepare the **held-out** eval sets (Bamboogle, FanOutQA, MuSiQue-test) — *no traces generated
      from them, ever*, so the zero-shot generalization claim stays clean.

### P1.2 Models and capacity (est. 3–4 days)

- [ ] **Teacher license audit (G6)** before any teacher's outputs are published — Kimi-K3 is
      "Modified MIT"; verify redistribution of generated text is permitted, or drop the teacher.
      The "fully open, redistributable corpus" claim is worth more than any single teacher.
- [ ] **Measure teacher concurrency ceilings** per provider (not guess): requests/min before 429s,
      p95 latency, cost/1k episodes. Feed the numbers into the rate limiter and the schedule.
- [ ] **Stand up vLLM serving for all 4 students** on Frigga, under real HF ids. Qwen3.5 composites
      need `merge_qwen_composite.py` before serving (vLLM cannot LoRA-serve them unmerged).
- [ ] Decide whether to add the planned **~8B student** and a **small (~8B) local teacher** — the
      small-teacher arm is the practitioner-facing result ("is a cheap teacher enough?").

### P1.3 Pilot and go/no-go (est. 3–4 days)

- [ ] **`scripts/smoke_matrix.py --plan ...`** across every teacher × student cell that survived the
      probe. Fail fast on: parse-failure rate > 0.35, missing provenance, wrong dataset label,
      any leak, no teacher verdict.
- [ ] **Pilot: 500 episodes per new dataset × 2 students × {g0, g3, skip_teacher}.** Report leak
      rate, judge/deterministic agreement, parse-failure rate, tool-error rate, cost/episode,
      episodes/min.
- [ ] **Explicit go/no-go per dataset.** Three good datasets beat four shaky ones — MuSiQue and
      StrategyQA are the plausible drops.

### P1.4 Mass generation (est. 2 weeks wall-clock, mostly unattended)

- [ ] **Generate the collection plan**: `scripts/plan_collection.py --data-root data/datasets/tg_v1
      --out resource_paper/plan --per-combination <N> --anchor 500 --configs g3_plan g0_plan …`
      Commit the manifest — it *is* the experimental design.
- [ ] **Decide final scale.** Plan says ~184K episodes / 100–170 GPU-hours. Consider the trimmed
      version (drop Slice F, halve Slice D) if the deadline is tight; the anchor-set design keeps
      the paired comparisons valid at any scale.
- [ ] **Run in slice order D → A → C → B → E** (D first: training data unblocks every baseline).
      Runners are resumable; re-running a slice continues where it stopped.
- [ ] **Nightly ops**: `scripts/tg_run_status.py` / `watch_trace_collection.py`, cost tracker,
      failure triage, teacher-source breakdown (are we still on the free tier?).
- [ ] **Decide the fate of the 31.6K v0 HotpotQA pilot traces (G8)** — regenerate the useful
      configs under v1, or ship as a separately documented v0. Recommendation in PLAN: regenerate.

## P2 — Finish the science (baselines E1–E8)

- [x] **E2 base → guidance-SFT lift** — done (Qwen-0.8B 3.0% → 60.2% judge-correct, no teacher)
- [x] **E4 cross-student transfer** — done (trace quality beats trace ownership; precision and
      efficiency habits transfer from the trace author)
- [~] **E7 teacher-in-loop vs internalized** — four-arm unseen-100 done for 3 students; re-run on
      v1 data for consistency
- [ ] **E1 SFT dose–response** (the paper's headline curve): train the same student on TG-Core data
      from each guidance level (`skip_teacher`, g0…g4) and plot end-to-end accuracy vs level.
      Report paired per-question tests (McNemar) on the anchor set, not cross-population means.
- [ ] **E3 cross-dataset zero-shot**: train on HotpotQA+2Wiki → test MuSiQue, Bamboogle, FanOutQA.
- [ ] **E5 teacher-size scaling** (Slice B): does teacher identity/size change guidance quality?
- [ ] **E6 PRM from step scores** → best-of-n reranking. `m5_rlaif_prm` already builds a
      58,218-example PRM dataset and a smoke PRM hit AUC 0.77 — this is the cheapest new result
      on the list.
- [ ] **E8 budget/efficiency frontier** (Slice C): budget ∈ {3,5,9} × disclose ∈ {T,F}.
- [ ] **Finish `exp_teacher_only`** — re-run the failed qwen-0.8B / qwen-2B eval arms and publish
      `report_compare/`. Preliminary granite numbers say the *guidance* recipe wins (m1 cover 51.2%
      vs teacher-only 31.2%); confirm or refute across all three students, and state the §10
      confounds (data volume, budget, no error-recovery demonstrations).
- [ ] **Run one real m2/m3/m5 training** (m2 ReST round, m3 KTO→DPO stack on m1, m5 PRM) — all
      currently have smoke-only results with 8-step throwaway adapters. m4 GRPO only if the offline
      methods plateau.
- [ ] **Decide the wiki feature's fate** — gate it off for strong students, or investigate granite's
      36% duplicate-rejection rate and low EDIT usage; it currently costs granite ~11 points.

## P3 — Quality, human validation, release

- [ ] **Full-corpus validation pass**: `validate_teacher_guidance_run.py` on every run;
      100% schema conformance; exact + near-duplicate dedup; contamination check (all eval qids
      banned from every training view).
- [ ] **Independent post-hoc leakage auditor** (string + fuzzy + LLM) over *all* student-visible
      text — publish the true rate with a CI, not a claim of zero. (Remember: an in-episode leakage
      flag means the guard *fired and sanitized*; detection ≠ exposure.)
- [ ] **Human validation study** (required for A\*): 1,200 episodes stratified by dataset ×
      guidance level × student × correctness; 3 annotators; 300 overlapping items; labels =
      guidance helpfulness (1–5), factual correctness, leakage present, answer correct, failure
      category. Report Krippendorff's α, judge-vs-human κ, gate precision/recall with CIs.
      **Needs a decision on annotators and budget — this is on the critical path and has no owner.**
- [ ] **Build the release**: `scripts/build_release.py` → `full` + `train_safe` views, one config
      per slice, parquet + JSONL, qid-level stratified splits.
- [ ] **Datasheet for Datasets** + **Croissant metadata**, with a loud warning that
      `teacher_private_diagnosis` is gold-derived and must not be naively trained on.
- [ ] **Publish artifacts**: HF dataset, framework code + test suite, SFT checkpoints and recipes,
      the fixed-protocol evaluation harness, and the trace explorer polished into a public demo.

## P4 — Decisions the user must make (blocking the schedule)

1. **Target venue + deadline** — NeurIPS D&B is the recommendation; it sets everything else.
2. **Scale** — full ~184K episodes, or the trimmed variant?
3. **Dataset set** — confirm HotpotQA + 2Wiki + MuSiQue + StrategyQA (StrategyQA is the riskiest:
   boolean answers, constructed candidates, never converted).
4. **Teacher roster** — depends on the re-probe and the license audit. If only DeepSeek-V4-Flash
   is free and working, Slice B (teacher scaling) needs paid budget or a local small teacher.
5. **Budget cap** — FAU-first keeps cost ≈ $0; a fully paid run is ~$550, realistically $150–400.
6. **Human annotation** — who, and with what budget/ethics approval?
7. **v0 pilot traces** — regenerate under v1 configs, or ship as documented v0?

---

### Suggested first week

1. P0 environment + tests + path fixes + tar index (days 1–2)
2. `probe_models.py` re-probe and the teacher-roster decision (day 2)
3. P0.5 unknown-finish fix + rate limiter (day 3)
4. StrategyQA conversion + `tg_v1` preparation + offline smoke (days 4–5)
5. `smoke_matrix.py` on the surviving cells → the pilot's go/no-go (day 5)
