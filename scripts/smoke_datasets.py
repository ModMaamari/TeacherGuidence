"""Pre-flight checks for prepared datasets -- run this BEFORE any large collection.

Mass generation is the expensive step, so every mistake it is possible to catch cheaply
must be caught first. This script has two layers:

**Offline preflight (free, seconds).** For each prepared dataset: files parse, every
question validates against the canonical schema, qids are unique, every candidate doc
exists, and -- the check that actually matters -- the *real* local retriever can reach each
question's gold documents. A converter can emit schema-valid rows the retriever still
cannot serve; this is what catches that.

**Online smoke (a few API calls, minutes).** Runs a handful of real episodes per dataset
through the full teacher-guidance loop and then audits the produced episodes: the right
dataset label was recorded, provenance is stamped, a teacher actually served the calls, the
student's output parsed, no gold answer leaked, and metrics are present.

Usage::

    # offline only -- run this after every prepare_dataset.py
    .venv/bin/python scripts/smoke_datasets.py --data-root data/datasets/tg_v1

    # add the end-to-end run (needs GPUs + a teacher endpoint)
    .venv/bin/python scripts/smoke_datasets.py --data-root data/datasets/tg_v1 \
        --online --num-samples 3 --student Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.converters.base import (  # noqa: E402
    ConversionError,
    validate_example,
)
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever  # noqa: E402

#: A student that fails to produce parseable actions this often means the run is broken,
#: not merely hard -- observed healthy runs sit near zero.
MAX_PARSE_FAILURE_RATE = 0.35


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_dataset_pairs(data_root: Path) -> List[Tuple[str, Path, Path]]:
    """Discover ``(name, questions, corpus)`` triples written by prepare_dataset.py."""
    pairs = []
    for q_path in sorted(data_root.rglob("*_questions.jsonl")):
        c_path = q_path.with_name(q_path.name.replace("_questions.jsonl", "_corpus.jsonl"))
        if c_path.exists():
            pairs.append((q_path.name.replace("_questions.jsonl", ""), q_path, c_path))
    return pairs


def preflight(name: str, q_path: Path, c_path: Path, sample: int = 25) -> Tuple[List[str], Dict[str, Any]]:
    """Validate one prepared dataset. Returns ``(problems, stats)``."""
    problems: List[str] = []
    questions = _read_jsonl(q_path)
    corpus = _read_jsonl(c_path)
    if not questions:
        return [f"{name}: no questions"], {}
    if not corpus:
        return [f"{name}: no corpus documents"], {}

    by_qid: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for doc in corpus:
        by_qid[doc["qid"]].append(doc)

    qids = [q["id"] for q in questions]
    dupes = [q for q, n in collections.Counter(qids).items() if n > 1]
    if dupes:
        problems.append(f"{name}: {len(dupes)} duplicate qid(s), e.g. {dupes[:3]}")

    for q in questions:
        docs = by_qid.get(q["id"], [])
        if not docs:
            problems.append(f"{name}: {q['id']} has no corpus documents")
            continue
        try:
            validate_example(q, docs)
        except ConversionError as exc:
            problems.append(f"{name}: {exc}")
            if len(problems) > 20:
                problems.append(f"{name}: ... further errors suppressed")
                break

    # Retrieval parity on a sample: schema-valid rows can still be unservable.
    retriever = HotpotLocalRetriever(str(c_path))
    step = max(1, len(questions) // sample)
    checked = 0
    for q in questions[::step][:sample]:
        checked += 1
        results = retriever.search(q["id"], q["query"], k=len(by_qid.get(q["id"], [])) or 10)
        returned = {r["doc_id"] for r in results}
        missing = set(q["gold"]["gold_doc_ids"]) - returned
        if missing:
            problems.append(f"{name}: {q['id']} gold docs unreachable by retriever: {sorted(missing)[:3]}")

    stats = {
        "questions": len(questions),
        "corpus_docs": len(corpus),
        "retrieval_checked": checked,
        "sources": dict(collections.Counter(q.get("source", "?") for q in questions)),
        "gold_granularity": dict(collections.Counter(q.get("gold_granularity", "?") for q in questions)),
        "answer_type": dict(collections.Counter(q.get("answer_type", "?") for q in questions)),
        "num_hops": dict(collections.Counter(str(q.get("num_hops")) for q in questions)),
        "types": dict(collections.Counter(q.get("type", "") for q in questions).most_common(6)),
        "avg_candidates": round(
            sum(len(q["retrieval_scope"]["candidate_doc_ids"]) for q in questions) / len(questions), 1
        ),
        "avg_gold_docs": round(
            sum(len(q["gold"]["gold_doc_ids"]) for q in questions) / len(questions), 2
        ),
    }
    return problems, stats


def audit_episodes(run_root: Path, expected_dataset: str, expected_n: int) -> Tuple[List[str], Dict[str, Any]]:
    """Audit episodes produced by an online smoke run."""
    problems: List[str] = []
    episodes: List[Dict[str, Any]] = []
    for path in run_root.rglob("teacher_guidance_episodes.jsonl"):
        episodes.extend(_read_jsonl(path))

    if not episodes:
        return [f"{expected_dataset}: no episodes produced"], {}
    if len(episodes) < expected_n:
        problems.append(f"{expected_dataset}: {len(episodes)}/{expected_n} episodes produced")

    parse_failures = teacher_missing = leaked = no_metrics = 0
    for ep in episodes:
        if ep.get("dataset") != expected_dataset:
            problems.append(
                f"{expected_dataset}: episode {ep.get('qid')} recorded dataset="
                f"{ep.get('dataset')!r} -- provenance is wrong"
            )
        for field in ("schema_version", "framework_commit", "config_hash", "generated_at"):
            if not ep.get(field):
                problems.append(f"{expected_dataset}: episode {ep.get('qid')} missing {field}")
                break
        if not ep.get("teacher_models_used") and not (ep.get("steps") or [{}])[0].get("teacher_skipped"):
            teacher_missing += 1
        if not ep.get("final_metrics"):
            no_metrics += 1
        for step in ep.get("steps") or []:
            metrics = step.get("metrics") or {}
            if metrics.get("json_valid") is False or metrics.get("invalid_action"):
                parse_failures += 1
            if (step.get("leakage_check") or {}).get("gold_answer_leaked"):
                leaked += 1

    total_steps = sum(len(ep.get("steps") or []) for ep in episodes) or 1
    parse_rate = parse_failures / total_steps
    if parse_rate > MAX_PARSE_FAILURE_RATE:
        problems.append(
            f"{expected_dataset}: {parse_rate:.0%} of steps failed to parse "
            f"(limit {MAX_PARSE_FAILURE_RATE:.0%}) -- prompt/model mismatch?"
        )
    if teacher_missing:
        problems.append(f"{expected_dataset}: {teacher_missing} episode(s) had no teacher call")
    if leaked:
        problems.append(f"{expected_dataset}: {leaked} step(s) leaked the gold answer")
    if no_metrics:
        problems.append(f"{expected_dataset}: {no_metrics} episode(s) missing final_metrics")

    correct = sum(1 for e in episodes if (e.get("final_metrics") or {}).get("answer_correct"))
    stats = {
        "episodes": len(episodes),
        "answer_correct": correct,
        "step_parse_failure_rate": round(parse_rate, 3),
        "teachers": dict(collections.Counter(
            m for e in episodes for m in (e.get("teacher_models_used") or [])
        )),
        "students": dict(collections.Counter(e.get("student_model", "?") for e in episodes)),
    }
    return problems, stats


def run_online(name: str, q_path: Path, c_path: Path, args, out_root: Path) -> Optional[str]:
    """Run a few real episodes for one dataset. Returns an error string on failure."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts/run_tg_vllm.py"),
        "--num-samples", str(args.num_samples),
        "--num-gpus", str(args.num_gpus),
        "--workers-per-gpu", str(args.workers_per_gpu),
        "--student-model", args.student,
        "--budget", str(args.budget),
        "--planning-steps", str(args.planning_steps),
        "--hidden-budget",
        "--questions", str(q_path),
        "--corpus", str(c_path),
        "--out-root", str(out_root),
        "--tag", f"smoke{name.replace('_', '')[:12]}",
        "--run-name", "run",
    ]
    if args.teacher_model:
        cmd += ["--teacher-model", args.teacher_model]
    if args.teacher_router:
        cmd += ["--teacher-router", args.teacher_router]
    log = out_root.parent / f"{name}_online.log"
    out_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"  running {args.num_samples} episodes -> {log}")
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        return f"{name}: online run exited {proc.returncode} (see {log})"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", required=True, help="dir containing prepared datasets")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these prepared-dataset names")
    ap.add_argument("--sample", type=int, default=25, help="questions to retrieval-check")
    ap.add_argument("--online", action="store_true", help="also run real episodes")
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--teacher-router", default=None)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--planning-steps", type=int, default=3)
    ap.add_argument("--out", default="data/simulation_output/dataset_smoke")
    args = ap.parse_args()

    pairs = find_dataset_pairs(Path(args.data_root))
    if args.datasets:
        wanted = set(args.datasets)
        pairs = [p for p in pairs if p[0] in wanted or p[0].split("_")[0] in wanted]
    if not pairs:
        raise SystemExit(f"no prepared datasets found under {args.data_root}")

    all_problems: List[str] = []
    report: Dict[str, Any] = {}

    print(f"=== offline preflight: {len(pairs)} dataset(s) ===")
    ok_pairs = []
    for name, q_path, c_path in pairs:
        problems, stats = preflight(name, q_path, c_path, args.sample)
        report[name] = {"preflight": stats, "problems": problems}
        status = "FAIL" if problems else "ok"
        print(f"  [{status:4}] {name:34} {stats.get('questions', 0):>6} q  "
              f"{stats.get('corpus_docs', 0):>7} docs  "
              f"gold={stats.get('gold_granularity')}  hops={stats.get('num_hops')}")
        for p in problems[:5]:
            print(f"         - {p}")
        all_problems.extend(problems)
        if not problems:
            ok_pairs.append((name, q_path, c_path))

    if args.online:
        if not ok_pairs:
            raise SystemExit("offline preflight failed for every dataset; not running online")
        print(f"\n=== online smoke: {len(ok_pairs)} dataset(s) x {args.num_samples} episodes ===")
        for name, q_path, c_path in ok_pairs:
            out_root = Path(args.out) / name
            err = run_online(name, q_path, c_path, args, out_root)
            if err:
                all_problems.append(err)
                print(f"  [FAIL] {name}: {err}")
                continue
            problems, stats = audit_episodes(out_root, name.split("_")[0], args.num_samples)
            report[name]["online"] = stats
            report[name]["problems"] = report[name].get("problems", []) + problems
            print(f"  [{'FAIL' if problems else 'ok':4}] {name:34} "
                  f"{stats.get('episodes', 0)} episodes, "
                  f"{stats.get('answer_correct', 0)} correct, "
                  f"parse-fail {stats.get('step_parse_failure_rate')}")
            for p in problems[:5]:
                print(f"         - {p}")
            all_problems.extend(problems)

    out_path = Path(args.out) / "smoke_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_path}")

    if all_problems:
        print(f"\n=== {len(all_problems)} PROBLEM(S) -- do not start mass collection ===")
        for p in all_problems[:30]:
            print(f"  - {p}")
        sys.exit(1)
    print("\n=== all checks passed -- safe to collect ===")


if __name__ == "__main__":
    main()
