"""m4 GRPO trainer: group-relative policy optimization over multi-turn agent episodes.

For each question we sample G on-policy episodes (temperature > 0) through the SAME
environment the harness uses (prompt renderer + deterministic tool executor), score
them with the verifiable reward in rewards.py, normalize advantages within the group
(A_i = (r_i - mean)/std), and apply a policy-gradient update on the generated tokens
of every step of every episode:

    loss = - (1/N) sum_i A_i * mean_t log pi(y_it | x_it)

This is GRPO in its single-update-per-batch form (no importance ratio/clipping
needed because each batch is used exactly once, on-policy). Advantages are
group-normalized exactly as in GRPO; groups with zero reward variance are skipped
(no learning signal). LoRA keeps optimizer memory small; everything fits on ONE
A100 80GB. Rollouts dominate wall time (~8-12 min per update at Q=4, G=4, B=4).

RECOMMENDED: start from the m1 SFT adapter (--init-adapter). RL from the raw base
model mostly optimizes JSON formatting, wasting rollouts.

Artifacts: <out-base>/<ts>_<tag>/{adapter/, train.log, updates.jsonl, train_config.json}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger, timestamped_dir, write_json  # noqa: E402
from training_methods.common.hf_agent_loop import run_episode, load_questions  # noqa: E402
from training_methods.m4_grpo.rewards import episode_reward  # noqa: E402
from agentsim.teacher_guidance.local_retrieval import HotpotLocalRetriever  # noqa: E402
from agentsim.teacher_guidance.sft_export import DEFAULT_SYSTEM  # noqa: E402


class TrainingPolicy:
    """PolicyModel-compatible wrapper around the LIVE training model (LoRA active)."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, messages, max_new_tokens=700, temperature=0.0):
        import torch

        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        if was_training:
            self.model.train()
        return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def step_texts(episode):
    """(user_prompt, completion_text) pairs for every generated turn of an episode."""
    pairs = []
    if episode.get("plan") and episode["plan"].get("raw"):
        pairs.append((episode["plan"]["prompt"], episode["plan"]["raw"]))
    for s in episode["steps"]:
        pairs.append((s["student_prompt"], s["student_raw"]))
    return pairs


def completion_logprob(model, tokenizer, user_prompt, completion, max_length):
    """Mean log-prob of completion tokens given the chat-formatted prompt (grad-enabled).
    Returns None if the sequence exceeds max_length (skipped, logged by caller)."""
    import torch

    msgs = [{"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user_prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    )
    if not torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids["input_ids"]
    comp_ids = tokenizer(completion + tokenizer.eos_token, return_tensors="pt",
                         add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    if input_ids.shape[1] > max_length:
        return None
    input_ids = input_ids.to(model.device)
    n_prompt = prompt_ids.shape[1]
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp_lp = token_lp[:, n_prompt - 1:]
    return comp_lp.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--questions", default=str(Path(__file__).parent / "data" / "grpo_questions.jsonl"))
    ap.add_argument("--corpus", default="data/datasets/hotpot_teacher_guidance_train3000/hotpot_distractor_train_corpus.jsonl")
    ap.add_argument("--out-base", default=str(Path(__file__).parent / "runs"))
    ap.add_argument("--tag", default="grpo")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--questions-per-update", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reward-fn", default=None, help="dotted path to an alternative "
                    "reward (used by m5_rlaif_prm); default = m4_grpo.rewards.episode_reward")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.updates, args.questions_per_update, args.group_size = 2, 2, 2
        args.budget, args.max_new_tokens = 2, 300

    run_dir = timestamped_dir(args.out_base, args.tag + ("_smoke" if args.smoke else ""))
    log = setup_logger("m4_grpo", run_dir / "train.log")
    log.info(f"args: {vars(args)}")
    log.info(f"artifacts: {run_dir}")
    write_json(run_dir / "train_config.json", {"args": vars(args)})

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    reward_fn = episode_reward
    if args.reward_fn:
        import importlib
        mod, name = args.reward_fn.rsplit(".", 1)
        reward_fn = getattr(importlib.import_module(mod), name)
        log.info(f"using reward fn {args.reward_fn}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    if args.init_adapter:
        log.info(f"merging init adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    policy = TrainingPolicy(model, tokenizer)
    log.info(f"model ready in {time.time()-t0:.1f}s")

    questions = load_questions(args.questions)
    retriever = HotpotLocalRetriever(args.corpus)
    log.info(f"pool={len(questions)} questions | corpus loaded")

    q_iter = iter(questions)
    metrics_f = open(run_dir / "updates.jsonl", "a")
    for update in range(1, args.updates + 1):
        u0 = time.time()
        batch_q = []
        for _ in range(args.questions_per_update):
            try:
                batch_q.append(next(q_iter))
            except StopIteration:
                q_iter = iter(questions)
                batch_q.append(next(q_iter))

        # ---- rollouts (no grad) ----
        groups, update_rewards = [], []
        for q in batch_q:
            eps = []
            for g in range(args.group_size):
                ep = run_episode(policy, q, retriever, budget=args.budget,
                                 temperature=args.temperature,
                                 max_new_tokens=args.max_new_tokens)
                r = reward_fn(ep)
                eps.append((ep, r["reward"]))
            rewards = [r for _, r in eps]
            update_rewards.extend(rewards)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
            log.info(f"update {update} qid={q['id']} rewards={[round(x,3) for x in rewards]}")
            if std_r < 1e-4:
                log.info(f"  group skipped (zero variance)")
                continue
            groups.append([(ep, (r - mean_r) / (std_r + 1e-4)) for ep, r in eps])

        if not groups:
            log.info(f"update {update}: no usable groups, skipping optimizer step")
            continue

        # ---- policy gradient ----
        model.train()
        optimizer.zero_grad()
        n_eps = sum(len(g) for g in groups)
        losses, skipped_steps = [], 0
        for group in groups:
            for ep, adv in group:
                pairs = step_texts(ep)
                lps = []
                for user_prompt, completion in pairs:
                    lp = completion_logprob(model, tokenizer, user_prompt, completion, args.max_length)
                    if lp is None:
                        skipped_steps += 1
                        continue
                    lps.append(lp)
                if not lps:
                    continue
                ep_lp = torch.stack(lps).mean()
                loss = -(adv * ep_lp) / n_eps
                loss.backward()
                losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

        row = {
            "update": update,
            "n_groups": len(groups),
            "n_episodes": n_eps,
            "mean_reward": round(sum(update_rewards) / len(update_rewards), 4),
            "max_reward": round(max(update_rewards), 4),
            "loss_sum": round(sum(losses), 6),
            "grad_norm": round(float(grad_norm), 4),
            "skipped_long_steps": skipped_steps,
            "elapsed_s": round(time.time() - u0, 1),
        }
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()
        log.info(f"UPDATE {update}/{args.updates}: {row}")

        if update % args.save_every == 0 or update == args.updates:
            adapter_dir = run_dir / "adapter"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            log.info(f"adapter saved at update {update}: {adapter_dir}")

    metrics_f.close()
    print(json.dumps({"run_dir": str(run_dir), "adapter": str(run_dir / "adapter")}, indent=2))


if __name__ == "__main__":
    main()
