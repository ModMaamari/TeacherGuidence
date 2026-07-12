"""Merge a Qwen3.5 LoRA adapter into the FULL composite checkpoint for vLLM serving.

Why this exists (PLAN.md G-5): Qwen3.5 loads in vLLM as ``Qwen3_5ForConditionalGeneration``
-- a vision-language composite with a vision tower and a multi-token-prediction (mtp) head
-- but the m1 LoRA adapter is trained against the ``Qwen3_5ForCausalLM`` text-only view. If
you hand vLLM the adapter it loads but is silently ignored (its module names don't match),
so base and adapter outputs are identical. Merging offline and serving the merged model is
the fix -- but a plain ``merge_and_unload().save_pretrained()`` writes a ForCausalLM
checkpoint whose config type vLLM rejects for this architecture.

So we SPLICE: start from the base composite's config + all tensors (vision tower + mtp head
kept verbatim) and overwrite ONLY the language-model tensors with their merged values. The
ForCausalLM view names its LM tensors ``model.X``; the composite names them
``model.language_model.X`` -- that infix is the only transform. The adapter targets
attention/MLP projections only (not embeddings or lm_head), and the model ties its
embeddings, so the ForCausalLM view's tied ``lm_head.weight`` is dropped (the composite has
no separate lm_head tensor).

Usage:
    .venv_train/bin/python training_methods/common/merge_qwen_composite.py \
        --base Qwen/Qwen3.5-0.8B --adapter <run>/adapter --out <run>/merged \
        [--verify-against <existing merged dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LM_PREFIX = "model."
COMPOSITE_LM_PREFIX = "model.language_model."


def _load_safetensors_dir(d: str) -> dict:
    import torch  # noqa: F401
    from safetensors.torch import load_file
    sd = {}
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    for f in files:
        sd.update(load_file(f))
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base model id or local composite dir")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output merged composite dir")
    ap.add_argument("--verify-against", default=None,
                    help="existing merged dir to allclose-check LM tensors against")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir = args.base if os.path.isdir(args.base) else None
    if base_dir is None:
        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(args.base)
    print(f"[merge] base composite dir: {base_dir}")

    # 1. Full base composite tensor set (LM + vision tower + mtp head).
    base_sd = _load_safetensors_dir(base_dir)
    n_lm_composite = sum(1 for k in base_sd if k.startswith(COMPOSITE_LM_PREFIX))
    print(f"[merge] base tensors: {len(base_sd)} total, {n_lm_composite} language-model")

    # 2. Merged language-model tensors (base + LoRA), from the ForCausalLM view.
    print("[merge] loading ForCausalLM view + adapter, merging (CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged_lm = model.state_dict()

    # 3. Splice: overwrite the composite LM tensors with merged values.
    out_sd = dict(base_sd)
    n_written = 0
    for k, v in merged_lm.items():
        if k == "lm_head.weight":
            continue  # tied to embeddings; composite has no separate lm_head tensor
        if not k.startswith(LM_PREFIX):
            raise ValueError(f"unexpected ForCausalLM key (no 'model.' prefix): {k}")
        ck = COMPOSITE_LM_PREFIX + k[len(LM_PREFIX):]
        if ck not in out_sd:
            raise KeyError(f"composite is missing target key: {ck}")
        out_sd[ck] = v.to(out_sd[ck].dtype).contiguous().cpu()
        n_written += 1
    if n_written != n_lm_composite:
        raise AssertionError(f"wrote {n_written} LM tensors, expected {n_lm_composite}")
    print(f"[merge] spliced {n_written} language-model tensors "
          f"(vision + mtp kept from base)")

    # 4. Optional correctness check vs a known-good merged checkpoint.
    if args.verify_against:
        ref = _load_safetensors_dir(args.verify_against)
        bad = 0
        for k in list(out_sd.keys()):
            if k.startswith(COMPOSITE_LM_PREFIX) and k in ref:
                if not torch.allclose(out_sd[k].float(), ref[k].float(), atol=1e-3, rtol=1e-3):
                    bad += 1
        print(f"[merge] verify-against: {bad} LM tensor mismatches vs {args.verify_against}")
        if bad:
            raise AssertionError("merge does not reproduce the reference checkpoint")

    # 5. Save: merged tensors + the base composite config/tokenizer (drop the index; we
    #    write a single shard).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    skip_names = {"license", "license.txt", "readme.md", ".gitattributes"}
    for name in os.listdir(base_dir):
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        if name.lower() in skip_names or name.lower().startswith("readme"):
            continue
        src = os.path.join(base_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, out / name)
    # Make sure the tokenizer is complete/loadable from the merged dir.
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(str(out))
    print(f"[merge] wrote merged composite checkpoint -> {out}")
    print(json.dumps({"out": str(out), "tensors": len(out_sd),
                      "lm_tensors_written": n_written}, indent=2))


if __name__ == "__main__":
    main()
