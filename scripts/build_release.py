"""Build the publishable teacher-guidance dataset from raw generation runs.

Raw traces keep *everything* (provider response bodies, per-call USD cost, the router
chain) because that telemetry is what lets us debug and cost a run. The published dataset
keeps only what a researcher needs, so this script is the boundary between the two.

What is removed, and why:

* ``usage.cost`` -- generation ran on free academic gateways; a per-call USD figure is an
  artefact of which provider happened to serve a call, not a property of the data.
* ``raw_response`` -- entire provider response bodies: provider ids, system fingerprints,
  and megabytes of duplication of text already present in ``response_text``.
* ``teacher_router`` -- the fallback chain is an operational detail.
  ``teacher_models_used`` (which model *actually* answered) is scientifically meaningful
  and is kept.

Token counts, latencies, model ids, every prompt, every raw model output, the teacher's
guidance, the leakage checks and all metrics are kept.

Two views are emitted:

* ``full``       -- everything above, including the teacher's privileged reasoning
                    (``teacher_private_diagnosis`` and the gold-bearing teacher prompts).
* ``train_safe`` -- the privileged fields removed. Fine-tuning naively on ``full`` would
                    train a model on text containing the gold answer; ``train_safe`` is the
                    view that cannot leak that way and is the default for training.

Usage::

    .venv/bin/python scripts/build_release.py \
        --runs data/simulation_output/tg_m3_granite_b3_eden \
               data/simulation_output/tg_m3_qwen08b_b3_eden \
        --out data/release/tg_v1 --version 1.0
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.teacher_guidance.provenance import (  # noqa: E402
    EPISODE_SCHEMA_VERSION,
    framework_commit,
)

EPISODE_FILENAME = "teacher_guidance_episodes.jsonl"

#: Per-LLM-call keys dropped from every call record in the published data.
_CALL_DROP = ("raw_response",)
#: Keys dropped from every call's ``usage`` block.
_USAGE_DROP = ("cost",)
#: Episode-level keys dropped from the published data.
_EPISODE_DROP = ("teacher_router",)

#: Privileged (gold-bearing) fields, present in `full`, removed from `train_safe`.
_PRIVILEGED_STEP = ("teacher_prompt", "teacher_raw", "teacher_private_diagnosis")
_PRIVILEGED_PLAN = (
    "teacher_plan_review_prompt", "teacher_plan_review_raw", "teacher_plan_review_full",
)

#: Anything matching these must never reach a public file.
#:
#: The ``sk-`` pattern needs both guards below or it fires on ordinary corpus text: an
#: earlier version matched "Yakutsk-Kirensk-Krasnoyarsk" in a HotpotQA passage about a
#: Siberian air route. So: require a non-word char before ``sk-`` (otherwise any word
#: ending in "sk" followed by a hyphen matches), and require a digit in the body (real
#: keys are high-entropy; hyphenated place names are not).
_SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk-(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{16,}"),
)


def configured_secrets() -> List[str]:
    """The literal credential values this checkout is configured with.

    Scanning for these exactly is the high-precision half of the check -- it cannot false
    positive on corpus text, and it catches the credentials that could actually leak from
    this machine. Best-effort: missing config is not an error.
    """
    values: List[str] = []
    try:
        from agentsim.config import config as _cfg
        for name in ("FAU_LLM_API_KEY", "EDENAI_API_KEY", "CUSTOM_LLM_API_KEY",
                     "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            value = getattr(_cfg, name, None)
            if value and len(str(value)) >= 12:
                values.append(str(value))
    except Exception:  # noqa: BLE001 -- provenance of secrets must never break the build
        pass
    return values


def _clean_calls(calls: Any, drop: Iterable[str]) -> Any:
    if not isinstance(calls, list):
        return calls
    drop = tuple(drop)
    cleaned = []
    for call in calls:
        if not isinstance(call, dict):
            cleaned.append(call)
            continue
        call = {k: v for k, v in call.items() if k not in drop}
        usage = call.get("usage")
        if isinstance(usage, dict):
            call["usage"] = {k: v for k, v in usage.items() if k not in _USAGE_DROP}
        cleaned.append(call)
    return cleaned


def strip_episode(
    episode: Dict[str, Any], *, train_safe: bool, keep_raw_response: bool = False
) -> Dict[str, Any]:
    """Return the publishable form of one episode."""
    drop = () if keep_raw_response else _CALL_DROP
    ep = copy.deepcopy(episode)
    for key in _EPISODE_DROP:
        ep.pop(key, None)

    for step in ep.get("steps") or []:
        for key in ("student_calls", "teacher_calls"):
            if key in step:
                step[key] = _clean_calls(step[key], drop)
        if isinstance(step.get("wiki_update_call"), dict):
            step["wiki_update_call"] = _clean_calls([step["wiki_update_call"]], drop)[0]
        if train_safe:
            for key in _PRIVILEGED_STEP:
                step.pop(key, None)

    plan = ep.get("plan_review")
    if isinstance(plan, dict):
        if "initial_plan_calls" in plan:
            plan["initial_plan_calls"] = _clean_calls(plan["initial_plan_calls"], drop)
        for rnd in plan.get("rounds") or []:
            for key in ("review_calls", "revision_calls"):
                if key in rnd:
                    rnd[key] = _clean_calls(rnd[key], drop)
            if train_safe:
                for key in _PRIVILEGED_PLAN:
                    rnd.pop(key, None)
        if train_safe:
            for key in _PRIVILEGED_PLAN:
                plan.pop(key, None)
    return ep


def scan_for_secrets(blob: str, literals: Iterable[str] = ()) -> List[str]:
    """Return redacted descriptions of any credential found in serialized output.

    Checks the exact configured credentials first (no false positives), then the generic
    high-entropy patterns as defense in depth.
    """
    hits: List[str] = []
    for literal in literals:
        if literal and literal in blob:
            hits.append(f"configured-credential:{literal[:6]}...")
    for pattern in _SECRET_PATTERNS:
        for match in pattern.findall(blob):
            hits.append(f"pattern:{match[:14]}...")
    return hits


def iter_episodes(run_roots: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for root in run_roots:
        for path in sorted(root.rglob(EPISODE_FILENAME)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", required=True, help="raw generation run roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--views", nargs="+", default=["full", "train_safe"],
                    choices=["full", "train_safe"])
    ap.add_argument("--keep-raw-response", action="store_true",
                    help="keep provider response bodies (huge; off by default)")
    args = ap.parse_args()

    roots = [Path(r) for r in args.runs]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise SystemExit(f"run root(s) not found: {missing}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats: collections.Counter = collections.Counter()
    by_dataset: collections.Counter = collections.Counter()
    by_student: collections.Counter = collections.Counter()
    by_teacher: collections.Counter = collections.Counter()
    by_guidance: collections.Counter = collections.Counter()
    schema_versions: collections.Counter = collections.Counter()
    seen_keys: set = set()
    secrets_found: List[str] = []

    literals = configured_secrets()
    writers = {v: open(out / f"episodes.{v}.jsonl", "w", encoding="utf-8") for v in args.views}
    try:
        for episode in iter_episodes(roots):
            # An episode is identified by (question, config, student) -- the same question
            # legitimately appears many times across the matrix, so dedup on all three.
            key = (
                episode.get("qid"),
                episode.get("config_hash", ""),
                episode.get("student_model", ""),
                episode.get("teacher_model", ""),
            )
            if key in seen_keys:
                stats["duplicate_skipped"] += 1
                continue
            seen_keys.add(key)

            stats["episodes"] += 1
            by_dataset[episode.get("dataset", "unknown")] += 1
            by_student[episode.get("student_model", "unknown")] += 1
            by_teacher[episode.get("teacher_model", "unknown")] += 1
            by_guidance[episode.get("guidance_level")] += 1
            schema_versions[episode.get("schema_version", "unversioned")] += 1
            if (episode.get("final_metrics") or {}).get("answer_correct"):
                stats["answer_correct"] += 1

            for view, writer in writers.items():
                record = strip_episode(
                    episode,
                    train_safe=(view == "train_safe"),
                    keep_raw_response=args.keep_raw_response,
                )
                blob = json.dumps(record, ensure_ascii=False)
                found = scan_for_secrets(blob, literals)
                if found:
                    secrets_found.extend(found)
                writer.write(blob + "\n")
    finally:
        for writer in writers.values():
            writer.close()

    if secrets_found:
        raise SystemExit(
            f"ABORT: {len(secrets_found)} credential-like strings found in the output "
            f"(e.g. {secrets_found[:3]}). The release was written but MUST NOT be published."
        )

    files = {}
    for view in args.views:
        path = out / f"episodes.{view}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[view] = {"file": path.name, "sha256": digest,
                       "bytes": path.stat().st_size}

    manifest = {
        "version": args.version,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "framework_commit": framework_commit(),
        "episodes": stats["episodes"],
        "duplicate_skipped": stats["duplicate_skipped"],
        "answer_correct": stats["answer_correct"],
        "by_dataset": dict(by_dataset),
        "by_student": dict(by_student),
        "by_teacher": dict(by_teacher),
        "by_guidance_level": {str(k): v for k, v in sorted(by_guidance.items(), key=lambda x: (x[0] is None, x[0]))},
        "source_schema_versions": dict(schema_versions),
        "views": files,
        "removed_from_published_data": {
            "per_call": [] if args.keep_raw_response else list(_CALL_DROP),
            "per_call_usage": list(_USAGE_DROP),
            "per_episode": list(_EPISODE_DROP),
            "train_safe_additionally": list(_PRIVILEGED_STEP) + list(_PRIVILEGED_PLAN),
        },
        "source_runs": [str(r) for r in roots],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[release] {stats['episodes']} episodes "
          f"({stats['duplicate_skipped']} duplicates skipped)")
    print(f"[release] by dataset: {dict(by_dataset)}")
    print(f"[release] by guidance level: {manifest['by_guidance_level']}")
    if "unversioned" in schema_versions:
        print(f"[release] WARNING: {schema_versions['unversioned']} episodes predate schema "
              "versioning -- they were generated before provenance stamping was added")
    for view, info in files.items():
        print(f"[release] {view:11} -> {info['file']}  ({info['bytes'] / 1e6:.1f} MB)")
    print(f"[release] no credentials found in output")
    print(f"[release] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
