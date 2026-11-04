import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fashion_mnist(batch_size, root="data", num_workers=2):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ]
    )
    train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class MLP(nn.Module):
    """Two hidden layers, like the book's Fashion-MNIST classifier (chapter 10)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_optimizer(name, params, lr, weight_decay, eps=1e-8):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=weight_decay)
    if name == "c-adamw":
        from coptim import CAdamW

        # foreach=False uses the classic per-parameter loop from c_adamw.py,
        # which is the paper's Algorithm 2 without the distributed machinery.
        return CAdamW(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=eps,
            weight_decay=weight_decay,
            foreach=False,
            no_deprecation_warning=True,
        )
    raise ValueError(f"unknown optimizer: {name}")


def active_fraction(optimizer):
    """Share of coordinates where momentum and current gradient agree in sign.

    For C-AdamW this is exactly the unmasked fraction of the last step; for
    AdamW it is the same statistic, logged as a would-be mask diagnostic.
    """
    agree, total = 0.0, 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            m = state.get("exp_avg")
            if m is None or p.grad is None:
                continue
            agree += ((m * p.grad) > 0).sum().item()
            total += p.numel()
    return agree / total


def steps_to_target(step_losses, targets, window=500):
    """First step at which the `window`-step rolling mean of batch loss drops
    to or below each target. Returns None when a target is never reached."""
    arr = np.asarray(step_losses, dtype=np.float64)
    if len(arr) >= window:
        smooth = np.convolve(arr, np.ones(window) / window, mode="valid")
        smooth_steps = np.arange(window - 1, len(arr))
    else:
        smooth, smooth_steps = arr, np.arange(len(arr))
    out = {}
    for t in targets:
        hit = np.nonzero(smooth <= t)[0]
        out[t] = int(smooth_steps[hit[0]]) if len(hit) else None
    return out
