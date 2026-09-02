#!/usr/bin/env python
"""LLM-as-judge over evaluation episodes: is the final answer correct given the gold?

Works with any model reachable through the provider router (docs/PROVIDERS.md): pass a
comma-separated chain and every verdict uses the first model that answers. Resumable and
incremental: each verdict is appended to ``<out>/verdicts.jsonl`` as it lands, episodes
already judged are skipped on re-run, and an episode whose verdict could not be obtained
is left out (not written as null) so the next pass retries it. ``<out>/status.json``
tracks progress.

The judge sees only the question, the gold answer and the model's final answer -- never
the trajectory, the arm, or the model's identity -- so it cannot favour an arm.

Usage::

    python scripts/judge.py --judge oai-judge/kimi-k2.6,edenchat/mistral/mistral-medium \
        --episodes 'runs/eval/*/*/episodes.jsonl' --out runs/judge
    # custom prompt file (must contain {question}, {gold}, {answer})
    python scripts/judge.py ... --prompt-file my_judge_prompt.txt
"""
from __future__ import annotations

import argparse
import asyncio
import glob as globmod
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd.io import append_jsonl, read_jsonl
from tgd.logging_utils import write_json

DEFAULT_PROMPT = """You are grading one answer to a multi-hop question.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

The model answer is CORRECT if it conveys the gold answer, even if worded differently,
more verbose, or with extra correct detail. It is INCORRECT if it states something
different, says it does not know, or is empty.
Return ONLY a JSON object: {{"correct": 0 or 1, "reason": "<10 words>"}}"""


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        c = int(obj["correct"])
    except Exception:
        return None
    return {"correct": 1 if c else 0, "reason": str(obj.get("reason", ""))[:160]}


async def judge_one(client, sem, row, router, attempts, prompt_tpl, max_tokens):
    prompt = prompt_tpl.format(question=row["query"], gold=row["gold_answer"], answer=row["final_answer"] or "(empty)")
    async with sem:
        for attempt in range(1, attempts + 1):
            try:
                res, used = await client.get_completion_with_fallback(
                    list(router), prompt=prompt, temperature=0.0, max_tokens=max_tokens)
                text = res.get("text", "") if isinstance(res, dict) else res
                v = parse_verdict(text)
                if v:
                    return {**row, "verdict": v, "judge_model": used}
            except Exception:  # noqa: BLE001 -- retried below, then left for the next pass
                pass
            await asyncio.sleep(min(2 ** attempt, 8))
    return None


async def main_async(args) -> int:
    from agentsim.clients.llm_client import LLMClient
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vpath = out / "verdicts.jsonl"
    prompt_tpl = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    try:  # fail now, not after the first API call, if the template is wrong
        prompt_tpl.format(question="q", gold="g", answer="a")
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"--prompt-file template is invalid: unknown placeholder {exc}. "
                         "It may use only {question}, {gold} and {answer}; escape any other "
                         "brace as {{ or }}.")

    done = set()
    if vpath.exists():
        for r in read_jsonl(vpath):
            done.add((r["source"], r["qid"]))
    files = sorted(set(sum((globmod.glob(g, recursive=True) for g in args.episodes), [])))
    rows: List[Dict[str, Any]] = []
    for src in files:
        for e in read_jsonl(src):
            if (src, e["qid"]) in done:
                continue
            rows.append({"source": src, "qid": e["qid"], "query": e["query"],
                         "gold_answer": e.get("gold_answer", ""), "final_answer": e.get("final_answer", "")})
    print(f"{utc()} | {len(files)} episode files | {len(done)} already judged | {len(rows)} to judge", flush=True)
    if not rows:
        return 0

    router = [m.strip() for m in args.judge.split(",") if m.strip()]
    client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    written = failed = 0
    lock = asyncio.Lock()

    async def worker(row):
        nonlocal written, failed
        v = await judge_one(client, sem, row, router, args.attempts, prompt_tpl, args.max_tokens)
        async with lock:
            if v is None:
                failed += 1
            else:
                append_jsonl(vpath, v)
                written += 1
            n = written + failed
            if n % args.report_every == 0 or n == len(rows):
                rate = n / max(time.time() - t0, 1e-9)
                eta = (len(rows) - n) / max(rate, 1e-9) / 60
                print(f"{utc()} | {n}/{len(rows)} judged ({written} ok, {failed} failed) | {rate:.1f}/s | eta {eta:.0f} min", flush=True)
                write_json(out / "status.json", {"judged": n, "total": len(rows), "written": written,
                                                 "failed": failed, "rate_per_s": round(rate, 2),
                                                 "eta_min": round(eta, 1), "router": router})

    await asyncio.gather(*(worker(r) for r in rows))
    print(f"{utc()} | done: {written} written, {failed} unresolved (re-run to retry them) in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0 if failed == 0 else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, help="judge model id(s), comma-separated fallback chain")
    ap.add_argument("--episodes", nargs="+", required=True, help="glob(s) of episodes.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--report-every", type=int, default=100)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
