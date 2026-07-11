"""m5 PRM trainer: LoRA-SFT the digit judge (TRL SFTTrainer, 1-token completions).

Same training stack as m1 (prompt-completion, completion-only loss); the completion
is a single digit token, so this is effectively distilling the teacher's step-score
distribution into the student backbone. One A100 80GB.

Artifacts: <out-base>/<ts>_<tag>/{adapter/, train.log, ...}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402


def load_rows(path, limit=None):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "completion": r["completion"]})
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--train-file", default=str(Path(__file__).parent / "data" / "prm_train.jsonl"))
    ap.add_argument("--dev-file", default=str(Path(__file__).parent / "data" / "prm_dev.jsonl"))
    ap.add_argument("--out-base", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--tag", default="prm")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out_base, args.tag + ("_smoke" if args.smoke else ""))
    log = setup_logger("m5_prm", run_dir / "train.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    train_rows = load_rows(args.train_file, 64 if args.smoke else None)
    dev_rows = load_rows(args.dev_file, 16 if args.smoke else None)
    log.info(f"loaded {len(train_rows)} train / {len(dev_rows)} dev judge examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)

    cfg = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=8 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 20,
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=500,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(dev_rows) if dev_rows and not args.smoke else None,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    write_json(run_dir / "train_config.json", {"args": vars(args)})
    log.info("training ...")
    result = trainer.train()
    log.info(f"train done in {time.time()-t0:.1f}s: {result.metrics}")
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    write_json(run_dir / "final_metrics.json", dict(result.metrics))
    print(json.dumps({"run_dir": str(run_dir), "adapter": str(run_dir / "adapter")}, indent=2))


if __name__ == "__main__":
    main()
