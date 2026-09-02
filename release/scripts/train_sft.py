#!/usr/bin/env python
"""LoRA SFT of a student on guidance-as-internal-thought examples (TRL SFTTrainer).

Data: prompt/completion conversational JSONL as written by ``scripts/build_splits.py``
(``{"prompt": [messages], "completion": [message], "metadata": {...}}``). TRL applies
the model's own chat template and computes the loss on completion tokens only, so the
model learns to GENERATE the ``teacher_guidance`` block + thought + action and is never
trained on the (long) prompt tokens.

Resumable: ``--out`` is a fixed directory. If ``<out>/adapter/.done`` exists the run is
skipped; if ``<out>/checkpoints/checkpoint-*`` exists training resumes from the latest
checkpoint (optimizer state included). Monitorable: ``<out>/train.log`` (UTC timestamps)
and ``<out>/status.json`` (step, epoch, loss, ETA -- rewritten at every logging step).

Artifacts: ``<out>/adapter/`` (PEFT adapter + tokenizer), ``train_config.json``,
``trainer_state.json`` (full log history), ``final_metrics.json``.

Examples::

    # group B student (train on all four datasets)
    python scripts/train_sft.py --train-file data/splits/uniform/train.jsonl \
        --dev-file data/splits/uniform/dev.jsonl --out runs/train/uniform

    # group A fold (never sees musique)
    python scripts/train_sft.py --train-file data/splits/lodo/fold_musique/train.jsonl \
        --dev-file data/splits/lodo/fold_musique/dev.jsonl --out runs/train/fold_musique

    # any other HF causal LM student
    python scripts/train_sft.py --model Qwen/Qwen2.5-3B-Instruct ...

    # 2-minute pipeline validation
    python scripts/train_sft.py --smoke --out runs/train/smoke ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> import tgd/agentsim

import tgd  # noqa: F401
from tgd.io import load_jsonl
from tgd.logging_utils import setup_logger, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b", help="HF id or local path of the student")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--out", required=True, help="run directory (fixed; resumable)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x grad-accum x GPUs")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None, help="cap training examples")
    ap.add_argument("--smoke", action="store_true", help="64 examples, 8 optimizer steps, no eval")
    ap.add_argument("--force", action="store_true", help="ignore an existing .done marker")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "adapter" / ".done"
    if done.exists() and not args.force:
        print(f"already trained: {out / 'adapter'} (delete {done} or pass --force to retrain)")
        return 0
    log = setup_logger("train_sft", out / "train.log")
    log.info(f"args: {vars(args)}")

    for f in [args.train_file] + ([args.dev_file] if args.dev_file else []):
        if not Path(f).exists():
            log.error(f"{f} not found. The SFT train/dev files are not shipped; they are "
                      f"built from the episodes in data/episodes: run 'make data' "
                      f"(or scripts/build_splits.py) first.")
            return 2

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    limit = 64 if args.smoke else args.limit
    train_rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in load_jsonl(args.train_file, limit)]
    dev_rows = ([{"prompt": r["prompt"], "completion": r["completion"]} for r in load_jsonl(args.dev_file, 32 if args.smoke else None)]
                if args.dev_file else [])
    log.info(f"loaded {len(train_rows)} train / {len(dev_rows)} dev examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    log.info(f"model loaded in {time.time() - t0:.1f}s | cuda={torch.cuda.is_available()} "
             f"gpus={torch.cuda.device_count()}")

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    do_eval = bool(dev_rows) and not args.smoke
    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=8 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.smoke else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        completion_only_loss=True,
    )

    class Status(TrainerCallback):
        """status.json for monitoring: step, epoch, last loss, ETA."""
        def on_log(self, a, state, control, logs=None, **kw):
            if not state.is_world_process_zero:
                return
            elapsed = time.time() - t0
            rate = state.global_step / max(elapsed, 1e-9)
            eta = (state.max_steps - state.global_step) / max(rate, 1e-9)
            write_json(out / "status.json", {
                "step": state.global_step, "max_steps": state.max_steps,
                "epoch": round(state.epoch or 0, 3), "elapsed_s": round(elapsed),
                "eta_s": round(eta), "last_log": {k: v for k, v in (logs or {}).items()},
            })

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(dev_rows) if do_eval else None,
        processing_class=tokenizer, peft_config=peft_config, callbacks=[Status()],
    )
    if trainer.is_world_process_zero():
        write_json(out / "train_config.json", {"args": vars(args), "trl_config": cfg.to_dict()})

    ckpts = sorted(glob.glob(str(out / "checkpoints" / "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts and not args.smoke else None
    log.info(f"training ... {'resuming from ' + resume if resume else 'from scratch'}")
    result = trainer.train(resume_from_checkpoint=resume)
    log.info(f"train done in {time.time() - t0:.1f}s: {result.metrics}")

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    final = dict(result.metrics)
    if do_eval:
        final.update(trainer.evaluate())
        log.info(f"final eval: {final}")
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_dir))
        write_json(out / "final_metrics.json", final)
        with open(out / "trainer_state.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log.info(f"adapter saved: {adapter_dir}")
        print(json.dumps({"out": str(out), "adapter": str(adapter_dir),
                          **{k: v for k, v in final.items() if isinstance(v, (int, float))}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
