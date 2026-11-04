import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join("outputs", "e1_fashion_mnist")
LABELS = {"adamw": "AdamW", "c-adamw": "C-AdamW"}
COLORS = {"adamw": "tab:blue", "c-adamw": "tab:orange"}


def load_runs():
    runs = {}
    for opt in LABELS:
        for run_dir in sorted(glob.glob(os.path.join(OUT_DIR, opt, "seed*"))):
            steps = np.genfromtxt(os.path.join(run_dir, "steps.csv"), delimiter=",", names=True)
            epochs = np.genfromtxt(os.path.join(run_dir, "epochs.csv"), delimiter=",", names=True)
            with open(os.path.join(run_dir, "summary.json")) as f:
                summary = json.load(f)
            runs.setdefault(opt, []).append({"steps": steps, "epochs": epochs, "summary": summary})
    return runs


def rolling(x, window):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    runs = load_runs()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for opt, rlist in runs.items():
        steps = rlist[0]["steps"]
        axes[0].plot(steps["step"], steps["train_loss"], color=COLORS[opt], alpha=0.15, lw=0.5)
        smooth = rolling(steps["train_loss"], 500)
        axes[0].plot(steps["step"][: len(smooth)], smooth, color=COLORS[opt], label=LABELS[opt], lw=1.5)
        for r in rlist:
            axes[1].plot(r["epochs"]["epoch"], r["epochs"]["test_acc"], color=COLORS[opt], alpha=0.8)
        axes[2].plot(steps["step"][: len(rolling(steps["active_fraction"], 500))],
                    rolling(steps["active_fraction"], 500), color=COLORS[opt], label=LABELS[opt], lw=1.5)

    axes[0].set(xlabel="step", ylabel="training loss", title="Training loss (500-step rolling mean)")
    axes[0].legend()
    axes[1].set(xlabel="epoch", ylabel="test accuracy", title="Test accuracy per epoch")
    axes[2].set(xlabel="step", ylabel="active fraction", title="Coordinates where momentum agrees with gradient")
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_e1_fashion_mnist.png"), dpi=150)

    print(f"{'':14s} {'final acc':>10s} {'best acc':>9s} {'time (s)':>9s}  steps to target loss")
    for opt, rlist in runs.items():
        s = rlist[0]["summary"]
        tgt = s["steps_to_target"]
        shown = ", ".join(f"{k}: {v if v is not None else '-'}" for k, v in tgt.items())
        print(f"{LABELS[opt]:14s} {s['final_test_acc']:10.4f} {s['best_test_acc']:9.4f} "
              f"{s['total_time_s']:9.1f}  {shown}")


if __name__ == "__main__":
    main()
