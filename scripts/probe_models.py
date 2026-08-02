"""Live availability + output-shape probe for every teacher source and student.

Run this before any collection, and whenever a provider misbehaves. It answers the two
questions that actually decide whether a run will work:

1. **Does the source respond at all?** Gateways break and credit runs out independently of
   our code -- the probe reports each source separately rather than letting the router
   silently fall through to a paid one.
2. **Can we parse a verdict out of what it returns?** Teachers are reasoning models, and
   they differ in where the reasoning goes: a separate ``reasoning_content`` field, a
   separate ``reasoning`` item in the EdenAI response array, or inline
   ``<think>``-style tags in the content itself. The probe sends a *real* teacher-style
   prompt and checks the JSON verdict parses -- and flags when reasoning arrived inline,
   because that is the case where a naive parser would return the model's discarded draft.

Usage::

    .venv/bin/python scripts/probe_models.py                 # all teachers, every source
    .venv/bin/python scripts/probe_models.py --students      # also check student serving
    .venv/bin/python scripts/probe_models.py --teacher kimi-k3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentsim.clients.llm_client import LLMClient  # noqa: E402
from agentsim.config import config  # noqa: E402
from agentsim.teacher_guidance.json_utils import (  # noqa: E402
    parse_teacher_evaluation,
    strip_reasoning_blocks,
)
from agentsim.teacher_guidance.model_registry import (  # noqa: E402
    STUDENTS,
    TEACHERS,
    get_student,
    get_teacher,
)

#: A realistic teacher turn: gold visible, JSON verdict demanded. Deliberately close to
#: the real prompt so the probe exercises the same parse path production does.
TEACHER_PROMPT = """You are a teacher evaluating one step of a student's retrieval-agent trajectory.
You can see the gold answer, which is PRIVATE.

Question: Where is the Oberoi Group headquartered?
Gold metadata (PRIVATE): {"answer": "Delhi"}
Current step 1 of budget 3.
Student action: {"thought": "I should search for the company.", "action": {"tool": "search", "params": {"query": "Oberoi Group headquarters"}}}
Tool observation: {"tool": "search", "status": "ok", "results": [{"title": "The Oberoi Group", "text_preview": "The Oberoi Group is headquartered in Delhi."}]}

Think step by step, then return ONLY a JSON object:
{
  "guidance_level": 3,
  "student_visible": {"score_binary": 0, "score_continuous": 0.0, "feedback": "...", "hint": null},
  "private_diagnosis": {"main_error": "..."},
  "teacher_decision": "continue|accept_finish|reject_finish"
}"""

STUDENT_PROMPT = (
    'Return ONLY a JSON object with this shape (no prose outside JSON):\n'
    '{"thought": "...", "action": {"tool": "search", "params": {"query": "..."}}}\n\n'
    "Question: Where is the Oberoi Group headquartered?"
)


async def probe_source(client: LLMClient, model_id: str, max_tokens: int) -> Dict[str, Any]:
    """Call one source directly (no router) and report what came back."""
    result: Dict[str, Any] = {"model_id": model_id}
    t0 = time.time()
    try:
        response = await client.get_completion(
            prompt=TEACHER_PROMPT, model=model_id, temperature=0.1,
            max_tokens=max_tokens, return_usage=True, return_raw=True,
        )
    except Exception as exc:  # noqa: BLE001 -- the point is to report failures
        result.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:160]}",
                      elapsed_s=round(time.time() - t0, 1))
        return result

    text = response.get("text", "") if isinstance(response, dict) else str(response)
    usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
    inline_reasoning = strip_reasoning_blocks(text) != text.strip()
    evaluation, info = parse_teacher_evaluation(text)

    result.update(
        ok=True,
        elapsed_s=round(time.time() - t0, 1),
        chars=len(text),
        json_valid=bool(info.get("json_valid")),
        schema_valid=bool(info.get("schema_valid", info.get("json_valid"))),
        decision=getattr(evaluation, "teacher_decision", None),
        inline_reasoning=inline_reasoning,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
        cost=usage.get("cost"),
        preview=text[:100].replace("\n", " "),
    )
    return result


async def probe_teachers(names: List[str]) -> Dict[str, Any]:
    client = LLMClient()
    report: Dict[str, Any] = {}
    for name in names:
        spec = get_teacher(name)
        print(f"\n--- {spec.display_name} ({spec.license}) ---")
        sources = []
        for source in spec.sources:
            if not config.provider_available(source.model_id):
                print(f"  [SKIP] {source.model_id:46} provider not configured")
                sources.append({"model_id": source.model_id, "ok": False,
                                "error": "provider not configured"})
                continue
            info = await probe_source(client, source.model_id, spec.max_tokens)
            info["provider"] = source.provider
            info["free"] = source.free
            if info["ok"]:
                flags = []
                if info["inline_reasoning"]:
                    flags.append("inline-reasoning")
                if not info["json_valid"]:
                    flags.append("JSON-PARSE-FAILED")
                status = "OK  " if info["json_valid"] else "WARN"
                print(f"  [{status}] {source.model_id:46} {info['elapsed_s']:>5}s  "
                      f"decision={info['decision']}  "
                      f"tok={info['completion_tokens']}  {' '.join(flags)}")
            else:
                print(f"  [FAIL] {source.model_id:46} {info['error'][:80]}")
            sources.append(info)
        usable = [s for s in sources if s.get("ok") and s.get("json_valid")]
        report[name] = {"license": spec.license, "sources": sources,
                        "usable_sources": len(usable)}
        if not usable:
            print(f"  ==> NO USABLE SOURCE for {name}")
    return report


def probe_students(names: List[str]) -> Dict[str, Any]:
    """Check each student's weights are resolvable (does not start a server)."""
    import httpx

    report: Dict[str, Any] = {}
    for name in names:
        spec = get_student(name)
        entry: Dict[str, Any] = {"hf_id": spec.hf_id, "license": spec.license}
        try:
            r = httpx.get(f"https://huggingface.co/api/models/{spec.hf_id}", timeout=30)
            entry["available"] = r.status_code == 200
            if r.status_code == 200:
                data = r.json()
                entry["architectures"] = (data.get("config") or {}).get("architectures")
                entry["gated"] = data.get("gated", False)
        except Exception as exc:  # noqa: BLE001
            entry["available"] = False
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
        cached = Path.home() / ".cache/huggingface/hub" / f"models--{spec.hf_id.replace('/', '--')}"
        entry["cached_locally"] = cached.exists()
        status = "OK  " if entry.get("available") else "FAIL"
        print(f"  [{status}] {spec.hf_id:32} {spec.params_b}B  "
              f"{entry.get('architectures')}  "
              f"cached={entry['cached_locally']}  merge_for_vllm={spec.needs_merge_for_vllm}")
        report[name] = entry
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--teacher", nargs="*", default=None)
    ap.add_argument("--students", action="store_true", help="also probe student weights")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    names = args.teacher or list(TEACHERS)
    print("=== teacher sources ===")
    report: Dict[str, Any] = {"teachers": asyncio.run(probe_teachers(names))}

    if args.students:
        print("\n=== students ===")
        report["students"] = probe_students(list(STUDENTS))

    unusable = [n for n, r in report["teachers"].items() if not r["usable_sources"]]
    print("\n=== summary ===")
    for name, entry in report["teachers"].items():
        print(f"  {name:20} {entry['usable_sources']}/{len(entry['sources'])} sources usable")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  report -> {args.out}")

    if unusable:
        print(f"\nTEACHERS WITH NO USABLE SOURCE: {unusable}")
        sys.exit(1)
    print("\nall teachers have at least one usable source")


if __name__ == "__main__":
    main()
