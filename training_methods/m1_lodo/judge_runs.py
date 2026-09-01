"""Resumable, monitorable LLM-judge pass over eval episodes.

Replaces the one-shot judge for large runs. The original gathered every verdict in memory
and wrote once at the end: a crash at 95% lost everything, and there was no way to see
progress. Over ~19,000 episodes that is the wrong shape.

This one:

* **writes incrementally** -- every verdict is appended to `verdicts.jsonl` as it lands,
  so the file is always a valid partial result;
* **resumes** -- verdicts already on disk are loaded and their episodes skipped, so a
  re-run continues instead of re-judging;
* **reports progress** -- a `status.json` and a log line every N verdicts, so a long run
  can be watched;
* **retries a failed verdict later** rather than writing a null: an episode with no usable
  verdict is left out of the file entirely and picked up on the next pass.

Router default is Kimi-K2.6 first: it answered in 0.6s while MiniMax-M3 was intermittently
throwing 500s and taking 11s. None of the three produced this corpus's training data, so
none can favour its own phrasing.

Usage::

    python training_methods/m1_lodo/judge_runs.py \
        --out training_methods/m1_lodo/runs/lodo/judge \
        --glob 'training_methods/m1_lodo/runs/lodo/eval/*/episodes.jsonl'
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agentsim.clients.llm_client import LLMClient  # noqa: E402

DEFAULT_ROUTER = [
    "fau/moonshotai/Kimi-K2.6",            # fastest, most reliable in probing
    "fau/MiniMaxAI/MiniMax-M3-MXFP8",      # intermittent 500s; kept as second
    "fau/mistralai/Mistral-Medium-3.5-128B",
]

PROMPT = """You are grading one answer to a multi-hop question.

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
    except Exception:
        return None
    if "correct" not in obj:
        return None
    try:
        c = int(obj["correct"])
    except Exception:
        return None
    return {"correct": 1 if c else 0, "reason": str(obj.get("reason", ""))[:120]}


async def judge_one(client: LLMClient, sem: asyncio.Semaphore, row: Dict[str, Any],
                    router: List[str], attempts: int) -> Optional[Dict[str, Any]]:
    prompt = PROMPT.format(question=row["query"], gold=row["gold_answer"],
                           answer=row["final_answer"] or "(empty)")
    async with sem:
        for attempt in range(1, attempts + 1):
            try:
                res, used = await client.get_completion_with_fallback(
                    list(router), prompt=prompt, temperature=0.0, max_tokens=600)
                text = res.get("text", "") if isinstance(res, dict) else res
                v = parse_verdict(text)
                if v:
                    return {**row, "verdict": v, "judge_model": used}
            except Exception:
                pass
            await asyncio.sleep(min(2 ** attempt, 8))
    return None      # leave it for the next pass rather than writing a null


async def main_async(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vpath = out / "verdicts.jsonl"

    done = set()
    if vpath.exists():
        for line in vpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["source"], r["qid"]))
                except Exception:
                    continue
    # recursive=True so a '**' pattern actually descends; without it the glob
    # silently matches nothing and the run reports 'nothing to do'.
    files = sorted(set(sum((globmod.glob(g, recursive=True) for g in args.glob), [])))
    rows: List[Dict[str, Any]] = []
    for src in files:
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            key = (src, e["qid"])
            if key in done:
                continue
            rows.append({"source": src, "qid": e["qid"], "query": e["query"],
                         "gold_answer": e.get("gold_answer", ""),
                         "final_answer": e.get("final_answer", "")})
    print(f"{utc()} | {len(files)} files | {len(done)} already judged | {len(rows)} to judge",
          flush=True)
    if not rows:
        print("nothing to do"); return 0

    router = [m.strip() for m in args.router.split(",") if m.strip()]
    client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    written = failed = 0
    lock = asyncio.Lock()

    async def worker(row):
        nonlocal written, failed
        v = await judge_one(client, sem, row, router, args.attempts)
        async with lock:
            if v is None:
                failed += 1
            else:
                with vpath.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(v, ensure_ascii=False) + "\n")
                written += 1
            n = written + failed
            if n % args.report_every == 0 or n == len(rows):
                rate = n / max(time.time() - t0, 1e-9)
                eta = (len(rows) - n) / max(rate, 1e-9) / 60
                msg = (f"{utc()} | {n}/{len(rows)} judged ({written} ok, {failed} failed) "
                       f"| {rate:.1f}/s | eta {eta:.0f} min")
                print(msg, flush=True)
                (out / "status.json").write_text(json.dumps(
                    {"updated": utc(), "judged": n, "total": len(rows), "written": written,
                     "failed": failed, "rate_per_s": round(rate, 2),
                     "eta_min": round(eta, 1), "router": router}, indent=2), encoding="utf-8")

    await asyncio.gather(*(worker(r) for r in rows))
    print(f"{utc()} | done: {written} written, {failed} unresolved "
          f"(re-run to retry them) in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--router", default=",".join(DEFAULT_ROUTER))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--report-every", type=int, default=200)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
