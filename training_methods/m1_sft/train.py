"""m1_sft trainer: LoRA SFT on guidance-as-internal-thought examples (TRL SFTTrainer).

Prompt-completion conversational data ({"prompt": [messages], "completion": [message]});
TRL applies the model's chat template and computes loss on completion tokens only, so
the model is trained to GENERATE the teacher_guidance block + thought + action and is
never trained on the (long) prompt tokens.

Fits on ONE A100 80GB (3.4B params, bf16 LoRA, gradient checkpointing). Also reused by
m2_rft for its retraining rounds (pass --train-file of the merged round dataset).

Artifacts (timestamped): <out-base>/<ts>_<tag>/{adapter/, train.log, train_config.json,
trainer_state.json, final_metrics.json}
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


def load_examples(path: str, limit: int | None = None):
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
    ap.add_argument("--train-file", default=str(Path(__file__).parent / "data" / "train.jsonl"))
    ap.add_argument("--dev-file", default=str(Path(__file__).parent / "data" / "dev.jsonl"))
    ap.add_argument("--out-base", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--tag", default="train")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (64 examples, 8 optimizer steps) to validate the pipeline")
    args = ap.parse_args()

    run_dir = timestamped_dir(args.out_base, args.tag + ("_smoke" if args.smoke else ""))
    log = setup_logger("m1_train", run_dir / "train.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    limit = 64 if args.smoke else None
    train_rows = load_examples(args.train_file, limit)
    dev_rows = load_examples(args.dev_file, 32 if args.smoke else None)
    log.info(f"loaded {len(train_rows)} train / {len(dev_rows)} dev examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    log.info(f"model loaded in {time.time()-t0:.1f}s")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
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
        logging_steps=1 if args.smoke else 10,
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=200,
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
        peft_config=peft_config,
    )
    write_json(run_dir / "train_config.json", {"args": vars(args), "trl_config": cfg.to_dict()})

    log.info("training ...")
    result = trainer.train()
    log.info(f"train done in {time.time()-t0:.1f}s: {result.metrics}")

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info(f"adapter saved: {adapter_dir}")

    final = dict(result.metrics)
    if dev_rows and not args.smoke:
        final.update(trainer.evaluate())
        log.info(f"final eval: {final}")
    write_json(run_dir / "final_metrics.json", final)
    with open(run_dir / "trainer_state.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "adapter": str(adapter_dir), **{k: v for k, v in final.items() if isinstance(v, (int, float))}}, indent=2))


if __name__ == "__main__":
    main()
