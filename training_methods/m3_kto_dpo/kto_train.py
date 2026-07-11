"""m3 KTO trainer: step-level binary preference optimization (TRL KTOTrainer).

Learns from BOTH good and bad steps: label=True steps (teacher score >= 0.9) are
made more likely, label=False steps (score <= 0.2) less likely, without needing
pairs. With ``peft_config`` the frozen reference policy is the same model with the
adapter disabled -- no second model in memory; fits on one A100 80GB.

RECOMMENDED: run on top of the m1 SFT adapter (--init-adapter <m1 adapter dir>);
KTO from the raw base model optimizes a policy that cannot yet produce the format.

Artifacts: <out-base>/<ts>_<tag>/{adapter/, train.log, train_config.json, ...}
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
                rows.append({"prompt": r["prompt"], "completion": r["completion"],
                             "label": bool(r["label"])})
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--init-adapter", default=None,
                    help="LoRA adapter (e.g. the m1 SFT adapter) merged into the base before KTO")
    ap.add_argument("--train-file", default=str(Path(__file__).parent / "data" / "kto_train.jsonl"))
    ap.add_argument("--dev-file", default=str(Path(__file__).parent / "data" / "kto_dev.jsonl"))
    ap.add_argument("--out-base", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--tag", default="kto")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out_base, args.tag + ("_smoke" if args.smoke else ""))
    log = setup_logger("m3_kto", run_dir / "train.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import KTOConfig, KTOTrainer

    t0 = time.time()
    train_rows = load_rows(args.train_file, 48 if args.smoke else None)
    dev_rows = load_rows(args.dev_file, 16 if args.smoke else None)
    log.info(f"loaded {len(train_rows)} train / {len(dev_rows)} dev "
             f"(pos={sum(r['label'] for r in train_rows)})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    if args.init_adapter:
        from peft import PeftModel
        log.info(f"merging init adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
    log.info(f"model ready in {time.time()-t0:.1f}s")

    cfg = KTOConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=6 if args.smoke else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=200,
        save_strategy="no",
        seed=args.seed,
        report_to=[],
    )
    trainer = KTOTrainer(
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
    with open(run_dir / "trainer_state.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "adapter": str(run_dir / "adapter")}, indent=2))


if __name__ == "__main__":
    main()
