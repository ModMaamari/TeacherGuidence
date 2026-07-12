"""Render training-curve PNGs from a trainer run dir's trainer_state.json.

Produces <run-dir>/curves/{loss.png, eval_loss.png, token_accuracy.png,
learning_rate.png, grad_norm.png, overview.png} from the HF Trainer log history
(the list-of-dicts dump written by train.py).

Usage:
    python training_methods/common/plot_training_curves.py --run-dir <dir> [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training_methods.common.tm_logging import setup_logger  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_COLOR = "#2a78d6"
EVAL_COLOR = "#1baf7a"


def series(history, key):
    pts = [(e["step"], e[key]) for e in history if key in e and "step" in e]
    seen, out = set(), []
    for s, v in pts:
        if s not in seen:
            seen.add(s)
            out.append((s, v))
    return [p[0] for p in out], [p[1] for p in out]


def single_plot(path, title, xs_ys_labels, ylabel, logy=False):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for xs, ys, label, color, marker in xs_ys_labels:
        ax.plot(xs, ys, label=label, color=color, linewidth=1.6,
                marker=marker, markersize=3 if marker else 0)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    if len(xs_ys_labels) > 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/curves)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "curves"
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger("plot_curves", out / "plot.log")

    history = json.loads((run_dir / "trainer_state.json").read_text())
    tag = run_dir.name

    tr_x, tr_y = series(history, "loss")
    ev_x, ev_y = series(history, "eval_loss")
    acc_x, acc_y = series(history, "mean_token_accuracy")
    eacc_x, eacc_y = series(history, "eval_mean_token_accuracy")
    lr_x, lr_y = series(history, "learning_rate")
    gn_x, gn_y = series(history, "grad_norm")

    made = []
    if tr_y:
        single_plot(out / "loss.png", f"{tag} — train loss (log y)",
                    [(tr_x, tr_y, "train", TRAIN_COLOR, None)]
                    + ([(ev_x, ev_y, "eval", EVAL_COLOR, "o")] if ev_y else []),
                    "loss", logy=True)
        made.append("loss.png")
    if ev_y:
        single_plot(out / "eval_loss.png", f"{tag} — eval loss",
                    [(ev_x, ev_y, "eval", EVAL_COLOR, "o")], "eval loss")
        made.append("eval_loss.png")
    if acc_y or eacc_y:
        rows = []
        if acc_y:
            rows.append((acc_x, acc_y, "train", TRAIN_COLOR, None))
        if eacc_y:
            rows.append((eacc_x, eacc_y, "eval", EVAL_COLOR, "o"))
        single_plot(out / "token_accuracy.png", f"{tag} — mean token accuracy", rows, "accuracy")
        made.append("token_accuracy.png")
    if lr_y:
        single_plot(out / "learning_rate.png", f"{tag} — learning rate",
                    [(lr_x, lr_y, "lr", TRAIN_COLOR, None)], "learning rate")
        made.append("learning_rate.png")
    if gn_y:
        single_plot(out / "grad_norm.png", f"{tag} — gradient norm",
                    [(gn_x, gn_y, "grad_norm", TRAIN_COLOR, None)], "grad norm")
        made.append("grad_norm.png")

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    panels = [
        (axes[0][0], "loss (log y)", [(tr_x, tr_y, "train", TRAIN_COLOR), (ev_x, ev_y, "eval", EVAL_COLOR)], True),
        (axes[0][1], "mean token accuracy", [(acc_x, acc_y, "train", TRAIN_COLOR), (eacc_x, eacc_y, "eval", EVAL_COLOR)], False),
        (axes[1][0], "learning rate", [(lr_x, lr_y, "lr", TRAIN_COLOR)], False),
        (axes[1][1], "gradient norm", [(gn_x, gn_y, "grad_norm", TRAIN_COLOR)], False),
    ]
    for ax, title, rows, logy in panels:
        for xs, ys, label, color in rows:
            if ys:
                ax.plot(xs, ys, label=label, color=color, linewidth=1.4)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        if len([r for r in rows if r[1]]) > 1:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{tag} — training curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "overview.png")
    plt.close(fig)
    made.append("overview.png")

    log.info(f"wrote {len(made)} PNGs to {out}: {made}")
    print(json.dumps({"out": str(out), "pngs": made}))


if __name__ == "__main__":
    main()
