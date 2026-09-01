"""Cross-fold report for m1-LODO: does internalized guidance generalize?

Reads every eval run under <out>/eval (merging the shards a big test set was split into)
and answers three questions, each needing a different comparison:

1. **Does training help in-distribution?**  fold model vs base on `heldin_*`
2. **Does it transfer to an unseen dataset?**  fold model vs base on `unseen_<fold>`
3. **How much is lost going out of distribution?**  a fold's own held-in vs its unseen

Every cell is reported against the base model on the SAME questions. A trained-model
number alone is uninterpretable: the datasets differ enormously in difficulty (collection
correctness ranged 30–69%), so "62% correct" means nothing without its baseline.

Fold results are never averaged into a single "generalization score" -- holding out
MuSiQue is a far harder test than holding out 2Wiki, and one mean would hide that.

Usage::

    python training_methods/m1_lodo/build_lodo_report.py \
        --out training_methods/m1_lodo/runs/lodo --md reports/m1_lodo/REPORT.md
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "strategyqa"]
#: Metrics summed across shards weight by episode count; these are simple means.
MEAN_KEYS = ["em", "f1", "cover_match", "doc_recall", "mean_steps", "tokens_per_episode"]
SUM_KEYS = ["n", "invalid_action_steps", "total_steps",
            "student_prompt_tokens", "student_completion_tokens"]


def load_runs(out: Path) -> Dict[str, Dict[str, Any]]:
    """tag -> merged metrics, combining `__sN` shards of the same (model, test set)."""
    shards: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for m in sorted((out / "eval").rglob("metrics.json")):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = m.parent.name
        # the agent writes <ts>_<tag>; strip timestamp and any shard suffix
        base = tag.split("_", 1)[1] if "_" in tag else tag
        base = base.split("__s")[0]
        shards[base].append(d)

    merged: Dict[str, Dict[str, Any]] = {}
    for tag, parts in shards.items():
        total = sum(p.get("n", 0) for p in parts)
        if not total:
            continue
        row: Dict[str, Any] = {"n_shards": len(parts)}
        for k in SUM_KEYS:
            row[k] = sum(p.get(k) or 0 for p in parts)
        for k in MEAN_KEYS:
            vals = [(p.get(k), p.get("n", 0)) for p in parts if p.get(k) is not None]
            row[k] = round(sum(v * n for v, n in vals) / total, 4) if vals else None
        stops: collections.Counter = collections.Counter()
        for p in parts:
            stops.update(p.get("stop_reasons") or {})
        row["stop_reasons"] = dict(stops)
        row["voluntary_finish"] = round(stops.get("finish", 0) / total, 4)
        row["invalid_rate"] = round(row["invalid_action_steps"] / max(row["total_steps"], 1), 4)
        merged[tag] = row
    return merged


def delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    d = (a - b) * 100
    return f"{d:+.1f}"


def pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v*100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_methods/m1_lodo/runs/lodo")
    ap.add_argument("--md", default="reports/m1_lodo/REPORT.md")
    args = ap.parse_args()
    out = Path(args.out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f"no eval runs under {out/'eval'}")

    L = ["# m1-LODO — Does Internalized Teacher Guidance Generalize?", "",
         "Each fold trains on three datasets' correct episodes (guidance-as-internal-thought)",
         "and is evaluated with **no teacher at inference**. Every cell is paired with the",
         "untrained base model on the *same* questions.", ""]

    # --- headline: out-of-distribution transfer ---
    L += ["## 1. Out-of-distribution — the held-out dataset", "",
          "| Fold (held out) | n | base correct | m1 correct | Δ | base F1 | m1 F1 | Δ F1 | m1 steps | m1 invalid |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        m = runs.get(f"fold_{ds}__unseen_{ds}")
        b = runs.get(f"base__unseen_{ds}")
        if not m or not b:
            L.append(f"| {ds} | *(pending)* | | | | | | | | |")
            continue
        L.append(f"| **{ds}** | {m['n']} | {pct(b['cover_match'])} | **{pct(m['cover_match'])}** | "
                 f"**{delta(m['cover_match'], b['cover_match'])}** | {b['f1']} | {m['f1']} | "
                 f"{delta(m['f1'], b['f1'])} | {m['mean_steps']} | {pct(m['invalid_rate'])} |")

    # --- in-distribution ---
    L += ["", "## 2. In-distribution — held-in questions of the training datasets", "",
          "| Fold | Test set | n | base correct | m1 correct | Δ | m1 F1 | m1 doc-recall |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for other in [d for d in DATASETS if d != ds]:
            m = runs.get(f"fold_{ds}__heldin_{other}")
            b = runs.get(f"base__heldin_{other}")
            if not m or not b:
                continue
            L.append(f"| {ds} | heldin_{other} | {m['n']} | {pct(b['cover_match'])} | "
                     f"{pct(m['cover_match'])} | {delta(m['cover_match'], b['cover_match'])} | "
                     f"{m['f1']} | {m['doc_recall']} |")

    # --- the generalization gap ---
    L += ["", "## 3. The generalization gap", "",
          "In-distribution minus out-of-distribution, within the same model. A small gap means",
          "the skill transferred; a large one means it was largely dataset-specific.", "",
          "| Fold | mean held-in correct | unseen correct | gap |", "|---|---:|---:|---:|"]
    for ds in DATASETS:
        hi = [runs[f"fold_{ds}__heldin_{o}"]["cover_match"] for o in DATASETS
              if o != ds and f"fold_{ds}__heldin_{o}" in runs]
        un = runs.get(f"fold_{ds}__unseen_{ds}")
        if not hi or not un:
            continue
        mh = st.mean(hi)
        L.append(f"| {ds} | {pct(mh)} | {pct(un['cover_match'])} | "
                 f"{delta(un['cover_match'], mh)} |")

    # --- cost and behaviour ---
    L += ["", "## 4. Cost and behaviour", "",
          "| Run | n | tokens/episode | steps | voluntary finish | invalid-action rate |",
          "|---|---:|---:|---:|---:|---:|"]
    for tag in sorted(runs):
        r = runs[tag]
        L.append(f"| `{tag}` | {r['n']} | {r['tokens_per_episode']} | {r['mean_steps']} | "
                 f"{pct(r['voluntary_finish'])} | {pct(r['invalid_rate'])} |")

    L += ["", "## Reading these numbers", "",
          "- **`cover_match` is the headline correctness metric.** EM under-credits verbose but",
          "  correct answers; F1 sits between. All three are reported; none should be read alone.",
          "- **Fold results are not averaged.** The datasets differ in difficulty, so a single",
          "  mean would hide that holding out MuSiQue is a much harder test than holding out 2Wiki.",
          "- **Every comparison is same-questions, same-decoding, same-backend.** Trained and base",
          "  arms run identical prompts through one vLLM server, greedy, fixed seed.", ""]

    md = Path(args.md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\nwrote {md} and {out/'summary.json'} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
