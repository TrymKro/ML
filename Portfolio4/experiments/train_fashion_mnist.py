import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    MLP,
    active_fraction,
    build_optimizer,
    load_fashion_mnist,
    set_seed,
    steps_to_target,
)

TARGET_LOSSES = [0.40, 0.35, 0.30, 0.25, 0.20]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser(description="E task: AdamW vs C-AdamW on Fashion-MNIST (MLP, chapter 10)")
    ap.add_argument("--optimizer", required=True, choices=["adamw", "c-adamw"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=os.path.join("outputs", "e1_fashion_mnist"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = load_fashion_mnist(args.batch_size)
    model = MLP().to(device)
    opt = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.eps)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = os.path.join(args.out_dir, args.optimizer, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    step = 0
    all_losses = []
    t_start = time.perf_counter()
    with open(os.path.join(run_dir, "steps.csv"), "w", newline="") as fs, open(
        os.path.join(run_dir, "epochs.csv"), "w", newline=""
    ) as fe:
        ws = csv.writer(fs)
        ws.writerow(["step", "epoch", "train_loss", "active_fraction"])
        we = csv.writer(fe)
        we.writerow(["epoch", "train_loss", "test_acc", "active_fraction", "lr", "epoch_time_s"])
        for epoch in range(args.epochs):
            model.train()
            losses, fracs = [], []
            t_epoch = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                step += 1
                losses.append(loss.item())
                all_losses.append(loss.item())
                fracs.append(active_fraction(opt))
                ws.writerow([step, epoch, f"{losses[-1]:.6f}", f"{fracs[-1]:.4f}"])
            test_acc = evaluate(model, test_loader, device)
            epoch_time = time.perf_counter() - t_epoch
            we.writerow(
                [
                    epoch + 1,
                    f"{np.mean(losses):.6f}",
                    f"{test_acc:.4f}",
                    f"{np.mean(fracs):.4f}",
                    opt.param_groups[0]["lr"],
                    f"{epoch_time:.1f}",
                ]
            )
            print(
                f"epoch {epoch + 1:2d}  loss {np.mean(losses):.4f}  "
                f"test_acc {test_acc:.4f}  active {np.mean(fracs):.3f}  ({epoch_time:.0f}s)",
                flush=True,
            )

    epochs_rows = np.atleast_1d(
        np.genfromtxt(os.path.join(run_dir, "epochs.csv"), delimiter=",", names=True)
    )
    summary = {
        "config": vars(args),
        "final_test_acc": float(epochs_rows["test_acc"][-1]),
        "best_test_acc": float(epochs_rows["test_acc"].max()),
        "final_train_loss": float(np.mean(losses)),
        "mean_active_fraction": float(np.mean(fracs)),
        "steps_to_target": {str(t): s for t, s in steps_to_target(all_losses, TARGET_LOSSES).items()},
        "total_time_s": round(time.perf_counter() - t_start, 1),
        "device": str(device),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
