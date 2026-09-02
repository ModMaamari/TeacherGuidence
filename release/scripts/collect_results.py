#!/usr/bin/env python
"""Gather every evaluation run into one results table (JSON + Markdown).

Layout convention (what scripts/eval.py + slurm/*.sbatch produce)::

    runs/eval/<arm-name>/<test-set>/episodes.jsonl      e.g. runs/eval/base/heldout_musique
    runs/judge/verdicts.jsonl                            from scripts/judge.py (optional)

Every ``<arm-name>/<test-set>`` becomes one row with accuracy (EM, F1, cover, judge),
efficiency (steps, voluntary finish, invalid steps, tokens, latency) and API cost.
Arms are also pooled over all test sets they share. For every pair of arms evaluated on
the same questions, a paired bootstrap 95% CI and an exact McNemar test on the judge
verdict (or cover-match when no verdicts exist) quantify whether the difference is real.

Usage::

    python scripts/collect_results.py --runs runs/eval --judge runs/judge/verdicts.jsonl \
        --out runs/results
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd.io import load_jsonl, read_jsonl
from tgd.metrics import aggregate


def paired_bootstrap(a: List[int], b: List[int], iters=10000, seed=13):
    rnd = random.Random(seed)
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    obs = sum(diffs) / n
    boots = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"diff": round(obs, 4), "ci95": [round(boots[int(0.025 * iters)], 4), round(boots[int(0.975 * iters) - 1], 4)], "n": n}


def mcnemar_exact(a: List[int], b: List[int]):
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    m = b01 + b10
    if m == 0:
        return {"b_wins": 0, "a_wins": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"b_wins": b01, "a_wins": b10, "p": round(p, 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/eval")
    ap.add_argument("--judge", default=None, help="verdicts.jsonl from scripts/judge.py")
    ap.add_argument("--out", default="runs/results")
    ap.add_argument("--metric", choices=["judge", "cover"], default="judge",
                    help="paired-test metric (falls back to cover when no verdicts)")
    args = ap.parse_args()
    root = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    judge: Dict[str, Dict[str, int]] = collections.defaultdict(dict)   # source file -> qid -> 0/1
    if args.judge and Path(args.judge).exists():
        for r in read_jsonl(args.judge):
            judge[str(Path(r["source"]).resolve())][r["qid"]] = int(bool(r["verdict"]["correct"]))

    rows: Dict[str, Dict[str, Any]] = {}
    per_q: Dict[str, Dict[str, Dict[str, int]]] = collections.defaultdict(dict)  # arm -> testset -> qid -> 0/1
    arms_sets: Dict[str, List[str]] = collections.defaultdict(list)
    for ep_file in sorted(root.glob("*/*/episodes.jsonl")):
        arm, test = ep_file.parent.parent.name, ep_file.parent.name
        eps = load_jsonl(ep_file)
        jv = judge.get(str(ep_file.resolve()), {})
        agg = aggregate(eps, jv if jv else None)
        agg["complete"] = (ep_file.parent / ".done").exists()
        rows[f"{arm}/{test}"] = agg
        arms_sets[arm].append(test)
        key = "judge" if (jv and args.metric == "judge") else "cover"
        per_q[arm][test] = {e["qid"]: (jv.get(e["qid"]) if key == "judge" else int(bool(e["final_metrics"].get("cover_match"))))
                            for e in eps}
        per_q[arm][test] = {q: v for q, v in per_q[arm][test].items() if v is not None}

    # pooled per arm over the test sets shared by every arm that has them
    pooled = {}
    for arm, tests in arms_sets.items():
        eps = []
        jv_all = {}
        for t in tests:
            f = root / arm / t / "episodes.jsonl"
            e = load_jsonl(f)
            eps += e
            jv_all.update({x["qid"]: v for x, v in ((x, judge.get(str(f.resolve()), {}).get(x["qid"])) for x in e) if v is not None})
        pooled[arm] = {**aggregate(eps, jv_all if jv_all else None), "test_sets": sorted(tests)}

    # paired comparisons for every arm pair on their common test sets
    pairs = {}
    arms = sorted(arms_sets)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            common = sorted(set(arms_sets[a]) & set(arms_sets[b]))
            if not common:
                continue
            res = {}
            for scope in common + (["pooled"] if len(common) > 1 else []):
                tests = common if scope == "pooled" else [scope]
                xa = {q: v for t in tests for q, v in per_q[a][t].items()}
                xb = {q: v for t in tests for q, v in per_q[b][t].items()}
                qs = sorted(set(xa) & set(xb))
                if len(qs) < 2:
                    continue
                la, lb = [xa[q] for q in qs], [xb[q] for q in qs]
                res[scope] = {**paired_bootstrap(la, lb), **mcnemar_exact(la, lb)}
            pairs[f"{a} -> {b}"] = res

    result = {"rows": rows, "pooled": pooled, "paired": pairs,
              "paired_metric": "judge" if judge else "cover"}
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # markdown
    md = ["# Results\n", "## Per arm and test set\n",
          "| arm | test set | n | done | EM | F1 | cover | judge | doc recall | steps | vol. finish | tokens/ep | latency s | API $ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(rows.items()):
        arm, test = k.split("/", 1)
        md.append(f"| {arm} | {test} | {r['n']} | {'yes' if r['complete'] else 'partial'} | {r['em']:.3f} | {r['f1']:.3f} | {r['cover_match']:.3f} | "
                  f"{('%.3f' % r['judge_correct']) if r.get('judge_correct') is not None else '—'} | "
                  f"{('%.3f' % r['doc_recall']) if r.get('doc_recall') is not None else '—'} | {r['mean_steps']:.2f} | {r['voluntary_finish']:.2f} | "
                  f"{r['total_tokens_per_ep']:,.0f} | {r['latency_s_per_ep']:.1f} | {r['api_cost_usd']:.4f} |")
    md += ["\n## Pooled per arm\n", "| arm | test sets | n | EM | F1 | cover | judge | steps | tokens/ep | API $ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in sorted(pooled.items()):
        md.append(f"| {arm} | {', '.join(r['test_sets'])} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} | {r['cover_match']:.3f} | "
                  f"{('%.3f' % r['judge_correct']) if r.get('judge_correct') is not None else '—'} | {r['mean_steps']:.2f} | {r['total_tokens_per_ep']:,.0f} | {r['api_cost_usd']:.4f} |")
    if pairs:
        md += [f"\n## Paired comparisons ({result['paired_metric']}-correct, b − a)\n",
               "| a -> b | scope | Δ pts | 95% CI | b wins / a wins | McNemar p |", "|---|---|---|---|---|---|"]
        for k, scopes in pairs.items():
            for scope, s in scopes.items():
                md.append(f"| {k} | {scope} | {s['diff'] * 100:+.1f} | [{s['ci95'][0] * 100:+.1f}, {s['ci95'][1] * 100:+.1f}] | {s['b_wins']} / {s['a_wins']} | {s['p']:.3g} |")
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out / 'results.json'} and {out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
