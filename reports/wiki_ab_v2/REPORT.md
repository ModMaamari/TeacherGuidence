# Wiki v2 A/B Report — Surgical-Edit Auto-Wiki vs. Baseline

**Suite:** `reports/wiki_ab_v2/` (raw episodes under `data/simulation_output/wikiv2_*`)
**Design:** 3 students × 5 paired repetitions × 10 HotpotQA questions/rep × {no-wiki, wiki} = 300 episodes
**Students:** qwen3.5:0.8b (q08b), qwen3.5:2b (q2b), granite4.1:3b (g3b)
**Correctness metric:** teacher-verdict score ≥ 0.40 ("teacher_rate"); secondary: deterministic cover-match ("cover", out of 10)
**v2 change under test:** wiki is now built from surgical ADD/EDIT/DEL/ANSWER/NEXT edit commands (not a full rewrite each step), injected into every step prompt and the forced-finish path, plus an ANSWER-line fallback in `derive_final_answer`.

---

## 1. Summary verdict

v2's surgical-edit engine **fixed the catastrophic failure mode from v1 for qwen2b**, but **did not fix it for qwen0.8b**, and **introduced a new, milder regression for granite3b** that wasn't present in v1 (granite wasn't tested in v1, per the prompt's context, but it's the strongest baseline model here and the wiki now measurably hurts it).

| Student | nowiki teacher-rate | wiki teacher-rate | Δ (mean) | Verdict |
|---|---|---|---|---|
| qwen3.5:0.8b | 0.322 ± 0.081 | 0.332 ± 0.122 | +0.010 | **Still a wash**, and gold-in-wiki-missed is worse in absolute terms than v1's failure signature |
| qwen3.5:2b | 0.610 ± 0.123 | 0.669 ± 0.195 | +0.059 | **Recovered from v1's disaster** (was −21.6pts in v1), but gains are inconsistent rep-to-rep and variance nearly doubled |
| granite4.1:3b | 0.894 ± 0.063 | 0.782 ± 0.062 | **−0.113** | **New regression.** Wiki costs granite ~11 points, consistently (4 of 5 reps negative) |

No student shows a clean, reliable win from the wiki. qwen2b is the only case where v2's fix clearly worked directionally, and even there the effect is noisy enough (one rep at −0.225) that it shouldn't be called a solved problem yet.

---

## 2. Per-student results

### qwen3.5:0.8b

| Arm | teacher-rate mean ± std | cover mean ± std | mean steps | natural-stop mean | unknown mean |
|---|---|---|---|---|---|
| nowiki | 0.3223 ± 0.0805 | 5.0 ± 1.22 | 8.46 | 0.0 | 0.0 |
| wiki | 0.3321 ± 0.1220 | 2.8 ± 1.10 | 8.06 | 0.6 | 0.2 |

Per-rep:

| Rep | nowiki teacher (n) | wiki teacher (n) | nowiki cover | wiki cover | Δ teacher-rate (wiki−nowiki) |
|---|---|---|---|---|---|
| 1 | 3/7 (n=10) | 2/7 (n=10) | 5 | 3 | −0.1429 |
| 2 | 2/7 (n=9) | 1/6 (n=8) | 5 | 1 | −0.1190 |
| 3 | 3/10 (n=10) | 5/10 (n=10) | 6 | 4 | +0.2000 |
| 4 | 3/8 (n=10) | 3/8 (n=10) | 6 | 3 | 0.0000 |
| 5 | 2/9 (n=10) | 3/9 (n=10) | 3 | 3 | +0.1111 |

teacher-rate is flat overall (+0.01), but **cover dropped from 5.0 to 2.8** — the wiki arm answers "correctly" on the lenient teacher-verdict more often by chance while doing measurably worse on exact/cover matching. Combined with a rise in `unknown` finals (0 → 0.2/rep) and `natural`-stop (0 → 0.6/rep), this reads as noise, not a real improvement.

### qwen3.5:2b

| Arm | teacher-rate mean ± std | cover mean ± std | mean steps | natural-stop mean | unknown mean |
|---|---|---|---|---|---|
| nowiki | 0.6100 ± 0.1232 | 5.6 ± 1.67 | 6.88 | 3.4 | 0.6 |
| wiki | 0.6689 ± 0.1947 | 6.2 ± 1.48 | 7.81 | 3.0 | 0.0 |

Per-rep:

| Rep | nowiki teacher (n) | wiki teacher (n) | nowiki cover | wiki cover | Δ teacher-rate |
|---|---|---|---|---|---|
| 1 | 5/8 (n=10) | 4/10 (n=10) | 6 | 4 | −0.2250 |
| 2 | 5/8 (n=10) | 5/9 (n=9) | 4 | 6 | −0.0694 |
| 3 | 4/8 (n=9) | 8/10 (n=10) | 6 | 7 | +0.3000 |
| 4 | 8/10 (n=10) | 8/9 (n=9) | 8 | 6 | +0.0889 |
| 5 | 4/8 (n=10) | 7/10 (n=10) | 4 | 8 | +0.2000 |

This is the one clear improvement over v1 (v1: wiki 22.7% vs nowiki 44.3%, a −21.6pt collapse). v2 flips the sign to +5.9pts mean, and `unknown` answers disappear entirely with wiki (0.6 → 0). But 2 of 5 reps are still negative, one sharply (rep 1: −0.225), and std nearly doubles (0.123 → 0.195) — the wiki is higher-variance, not uniformly better.

### granite4.1:3b

| Arm | teacher-rate mean ± std | cover mean ± std | mean steps | natural-stop mean | unknown mean |
|---|---|---|---|---|---|
| nowiki | 0.8944 ± 0.0626 | 7.8 ± 1.30 | 6.48 | 3.6 | 0.0 |
| wiki | 0.7817 ± 0.0621 | 7.6 ± 0.55 | 6.32 | 4.4 | 0.0 |

Per-rep:

| Rep | nowiki teacher (n) | wiki teacher (n) | nowiki cover | wiki cover | Δ teacher-rate |
|---|---|---|---|---|---|
| 1 | 7/8 (n=9) | 7/9 (n=9) | 9 | 7 | −0.0972 |
| 2 | 5/6 (n=10) | 7/8 (n=10) | 6 | 8 | +0.0417 |
| 3 | 7/8 (n=10) | 7/10 (n=10) | 8 | 8 | −0.1750 |
| 4 | 8/9 (n=10) | 7/9 (n=10) | 9 | 7 | −0.1111 |
| 5 | 5/5 (n=7) | 7/9 (n=10) | 7 | 8 | −0.2222 |

Granite is the strongest baseline student (89.4% teacher-rate, no `unknown` in either arm, no gold-in-wiki-missed at all — see §3) and the wiki arm is worse in 4 of 5 reps, by a consistent 10–20pt margin. Cover is roughly flat (7.8 → 7.6), so this isn't the "gold literally present but ignored" failure — it looks more like the wiki is diluting an already-effective reasoning process with low-value content (see edit-op volume below).

---

## 3. Did v2 fix the v1 failure modes?

**v1 recap (from prompt context):** qwen2b wiki teacher-rate 22.7% vs 44.3% baseline, 23 gold-in-wiki-missed episodes; qwen0.8b was a noisy wash (+9pts, std 29%).

### gold-in-wiki-missed (gold answer literally present in final wiki text, but final answer wrong)

| Student | wiki arm total (out of ~48–49 episodes) | Rate |
|---|---|---|
| qwen0.8b | 4+3+4+3+4 = **18** | ~37.5% |
| qwen2b | 3+1+1+1+2 = **8** | ~16.7% |
| granite3b | 0+0+0+0+0 = **0** | 0% |

- **qwen2b**: this is the headline fix. v1 had 23 gold-in-wiki-missed episodes for this exact model/question set; v2 cuts it to 8. The surgical-edit engine plus ANSWER-line fallback substantially reduced (but did not eliminate) the "wiki has the answer, student still fails" pathology.
- **qwen0.8b**: **not fixed** — 18 episodes with gold in the wiki but wrong final answer is worse in absolute count than qwen2b's v1 failure (23) relative to a smaller episode pool, and cover dropped sharply (5.0 → 2.8) alongside it. The 0.8b model appears too weak to reliably read its own wiki correctly even when it's populated with the right fact — this is a comprehension/attention problem the edit mechanics don't address.
- **granite3b**: zero gold-in-wiki-missed episodes in either arm. Granite's regression is a different failure mode entirely (see below), not the v1 pathology.

### unknown finals

| Student | nowiki unknown mean | wiki unknown mean |
|---|---|---|
| qwen0.8b | 0.0 | 0.2 |
| qwen2b | 0.6 | **0.0** |
| granite3b | 0.0 | 0.0 |

The ANSWER-line fallback appears to have fully eliminated `unknown` finals for qwen2b (0.6/rep → 0). It introduced a small new trickle of `unknown` for qwen0.8b (0 → 0.2/rep) — not fixed for that model, but not worse than noise-level either.

### edit-op quality (recomputed directly from raw op strings — see caveat below)

| Student | applied ops/episode | ADD | EDIT | DEL | ANSWER | NEXT | KEEP | ignored (total) | ignored rate |
|---|---|---|---|---|---|---|---|---|---|
| qwen0.8b | ~14.4 | 160 | 267 | 3 | 231 | 28 | 0 | 278 (77 bad-idx, 184 dup, 16 unrec.) | ~29% |
| qwen2b | ~16.4 | 203 | 229 | 18 | 257 | 79 | 2 | 178 (43 bad-idx, 125 dup, 8 unrec.) | ~18% |
| granite3b | ~4.5 | 147 | 21 | 6 | 44 | 6 | 0 | 127 (all duplicate) | ~36% |

- qwen0.8b and qwen2b both make heavy use of `EDIT` (i.e., genuinely rewriting existing wiki entries in place, not just appending) — 267 and 229 EDIT ops respectively. This is real surgical-edit behavior working as designed.
- **granite3b writes far fewer edits (~4.5/episode vs ~14–16/episode for the qwen models) and barely uses EDIT/DEL** (21 EDIT, 6 DEL total across 49 episodes). Its wiki entries are mostly `ADD`, and over a third of its attempted ops are rejected as duplicates. This is consistent with the regression: granite is spending step budget and prompt space maintaining a wiki that mostly restates what it already knows, with a high duplicate-rejection rate, and that overhead appears to cost it accuracy rather than help — its `natural`-stop count actually rises (3.6 → 4.4/rep), suggesting it's satisfying the teacher's acceptance criteria slightly more easily while getting the underlying answer wrong more often.
- **Important caveat:** `stats.json`'s `edit_ops.rewrites` and `edit_ops.keeps` sub-fields are always 0 for every rep/arm in this run. This is **not evidence that rewrites/keeps didn't happen** — it's an artifact of `scripts/analyze_wiki_ab_v2.py`'s `ep_stats()`, which checks `applied` entries against the literal strings `"rewrite"` and `"KEEP"`, but the real op vocabulary in the logs is `ADD/EDIT/DEL/ANSWER/NEXT/KEEP` (with `KEEP` appearing only twice, for qwen2b). The `EDIT` op — which functions as the rewrite mechanic — is never matched by that check, so this sub-metric should be treated as broken rather than informative. The applied/ADD/EDIT/DEL/ANSWER/NEXT/KEEP breakdown in the table above was recomputed directly from the raw `wiki_edit_ops` strings in the episode logs to get an accurate picture.

---

## 4. Limitations

- **n=10 per rep.** Each rep is only 10 HotpotQA questions; individual rep teacher-rates swing by 1 correct answer per ~10-14 points. The 5-rep paired design partially compensates, but per-rep numbers in the tables above should not be read as precise.
- **teacher_scored denominators vary and are sometimes small.** E.g. qwen0.8b wiki rep2 has `teacher_scored=6` out of `n=8` — 2 episodes weren't teacher-scored at all in that rep, and the rate is computed only over the scored subset. Cross-rep and cross-arm comparisons of `teacher_rate` are comparing means over different-sized, non-identical scored subsets, not a fixed 10-question denominator.
- **cover vs teacher_rate can diverge** (most visibly for qwen0.8b, where wiki improves teacher_rate slightly while cover drops sharply). The two metrics are not measuring the same thing (lenient teacher verdict vs strict match) and a report reader should not average them.
- **granite4.1:3b was not part of the v1 experiment**, so "fixed v1 failure mode" framing doesn't directly apply to it — its regression is a new finding from v2, not a carryover from v1.
- **The `rewrites`/`keeps` fields in `stats.json` are non-functional** (see §3); anyone consuming `stats.json` programmatically should not trust those two fields without re-deriving them from raw op strings, as done here.
- Single-seed, single-suite run — no repeated full-suite replication to bound run-to-run suite variance beyond the 5 paired reps already included.

---

## 5. Recommendations

1. **Do not ship v2 wiki as a default for granite4.1:3b.** It's a consistent ~11pt regression on the strongest baseline student with no compensating gold-in-wiki-missed improvement (there was nothing to fix). If granite-class students are a target, either gate the wiki off for strong-baseline students or investigate why granite's edit behavior is ADD-heavy/duplicate-heavy (~36% ignored) rather than genuinely revising entries — that ratio itself is a lead worth chasing before another A/B.
2. **qwen2b is a legitimate partial win — validate it isn't a fluke.** Gold-in-wiki-missed fell from 23 (v1) to 8 (v2), and unknown finals hit zero. Before calling this shipped, run a few more reps to see if the rep-1 −0.225 outlier is a recurring failure pattern or noise; right now it's 1-in-5 reps going strongly backward.
3. **qwen0.8b needs a different fix, not more edit-mechanics polish.** Gold is in the wiki 18/48 times and the model still gets it wrong, and cover dropped versus baseline. This model may be too weak to reliably attend to a wiki file at all — consider testing with a much shorter/higher-signal wiki (fewer, terser entries) rather than assuming the surgical-edit format itself is the lever for this size class.
4. **Fix `scripts/analyze_wiki_ab_v2.py`'s `ep_stats()`** so `rewrites`/`keeps` actually match the real op vocabulary (`EDIT`, `KEEP`) instead of silently reporting 0 — this metric is currently misleading anyone who reads `stats.json` without cross-checking raw logs, as had to be done for §3 of this report.
5. **Investigate the granite duplicate-rejection rate (~36%) and low EDIT usage separately from the accuracy question** — it may point to a prompt-formatting issue specific to how granite is instructed to consult/update its own wiki, independent of whether the wiki helps or hurts accuracy.
