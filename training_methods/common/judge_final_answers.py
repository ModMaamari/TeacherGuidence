"""Post-hoc teacher verdict on the final answers of any eval run.

For every episode in the given episodes.jsonl files, one gpt-oss-120b call (FAU ->
free OpenRouter -> paid OpenRouter router) judges the final answer against the gold
answer: {"correct": 0|1, "score": 0..1}. This is the same judge model the harness
uses in-loop, so verdicts are comparable to the b-run "teacher-correct" metric
(threshold 0.40 on score).

Usage:
    python training_methods/common/judge_final_answers.py \
        --out <verdict-dir> <episodes.jsonl> [<episodes.jsonl> ...] \
        [--concurrency 8]

Writes <out>/verdicts.jsonl (one row per episode: source file, qid, verdict) and
<out>/summary.json (per-source aggregates).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, write_json  # noqa: E402
from agentsim.clients.llm_client import LLMClient  # noqa: E402
from scripts.gen_fau_smoke_template import TEACHER_ROUTER  # noqa: E402

#: Default judge chain. Deliberately NOT the model that generated the training data --
#: a judge scoring its own phrasing favourably would inflate every result -- and
#: deliberately not gpt-oss (excluded by standing instruction, and its FAU host is down).
#: Both entries were probed answering a real verdict prompt cleanly.
DEFAULT_JUDGE_ROUTER = [
    "fau/MiniMaxAI/MiniMax-M3-MXFP8",
    "fau/moonshotai/Kimi-K2.6",
]

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Gold answer: {gold}
System answer: {answer}

Does the system answer convey the same fact as the gold answer? Judge semantic
equivalence: extra wording, different phrasing or added context is fine as long as
the core answer is right; a wrong, contradictory, empty or evasive answer (e.g.
"unknown") is incorrect.

Reply with ONLY one JSON object, no other text:
{{"correct": 0 or 1, "score": <float 0.0-1.0: the probability that the system answer
is correct; must agree with "correct" (above 0.5 only when correct=1)>}}"""

VERDICT_RE = re.compile(r"\{[^{}]*\}")


def parse_verdict(raw: str) -> Optional[Dict[str, Any]]:
    m = VERDICT_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "correct" not in obj and "score" not in obj:
        return None
    score = obj.get("score")
    try:
        score = max(0.0, min(1.0, float(score))) if score is not None else None
    except (TypeError, ValueError):
        score = None
    correct = obj.get("correct")
    if isinstance(correct, bool):
        correct = int(correct)
    if correct not in (0, 1):
        correct = 1 if (score or 0) >= 0.5 else 0
    if score is None:
        score = float(correct)
    return {"correct": correct, "score": score}


async def judge_one(client: LLMClient, sem: asyncio.Semaphore, row: Dict[str, Any], log) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=row["query"], gold=row["gold_answer"], answer=row["final_answer"] or "(empty)")
    async with sem:
        for attempt in (1, 2):
            try:
                result, used_model = await client.get_completion_with_fallback(
                    list(row.get("_judge_router") or DEFAULT_JUDGE_ROUTER),
                    prompt=prompt, temperature=0.0,
                    max_tokens=1500, return_raw=True,
                )
                text = result.get("text", "") if isinstance(result, dict) else result
                verdict = parse_verdict(text)
                if verdict:
                    return {**row, "verdict": verdict, "judge_model": used_model}
                log.warning(f"qid={row['qid']} unparseable verdict (attempt {attempt}): {text[:100]!r}")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"qid={row['qid']} judge call failed (attempt {attempt}): {exc}")
                await asyncio.sleep(2 * attempt)
    return {**row, "verdict": None, "judge_model": None}


def _attach_router(rows, router):
    for r in rows:
        r["_judge_router"] = router
    return rows


async def run(args, log) -> None:
    client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    rows = []
    for src in args.episodes:
        for line in open(src):
            if not line.strip():
                continue
            e = json.loads(line)
            rows.append({
                "source": str(src), "qid": e["qid"], "query": e["query"],
                "gold_answer": e["gold_answer"], "final_answer": e.get("final_answer", ""),
            })
    router = [m.strip() for m in args.judge_router.split(",") if m.strip()]
    _attach_router(rows, router)
    log.info(f"judging {len(rows)} final answers from {len(args.episodes)} files "
             f"(concurrency {args.concurrency}, router {router})")
    t0 = time.time()
    results = await asyncio.gather(*(judge_one(client, sem, r, log) for r in rows))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.pop("_judge_router", None)
    with open(out / "verdicts.jsonl", "w") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {}
    for src in {r["source"] for r in results}:
        sub = [r for r in results if r["source"] == src]
        judged = [r for r in sub if r["verdict"]]
        summary[src] = {
            "n": len(sub),
            "judged": len(judged),
            "teacher_correct": sum(r["verdict"]["correct"] for r in judged),
            "teacher_correct_rate": round(
                sum(r["verdict"]["correct"] for r in judged) / max(len(judged), 1), 4),
            "mean_score": round(
                sum(r["verdict"]["score"] for r in judged) / max(len(judged), 1), 4),
        }
    write_json(out / "summary.json", summary)
    log.info(f"done in {time.time()-t0:.0f}s: {json.dumps(summary, indent=2)}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--judge-router", default=",".join(DEFAULT_JUDGE_ROUTER),
                    help="comma-separated model chain; each is tried in order per call")
    ap.add_argument("episodes", nargs="+")
    args = ap.parse_args()
    log = setup_logger("judge", Path(args.out) / "judge.log")
    asyncio.run(run(args, log))


if __name__ == "__main__":
    main()
