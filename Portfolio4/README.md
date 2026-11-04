# Portfolio 4 — Cautious Optimizers: Improving Training with One Line of Code

Reproduction of Liang et al. (ICLR 2026, arXiv:2411.16085) for IDATG2208,
scaled down per the course brief.

- Dataset 1: Fashion-MNIST with an MLP (book chapter 10)
- Dataset 2: CIFAR-10 with a small ResNet (book chapter 12, week 3)

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Colab, torch is preinstalled; `!pip install matplotlib` is enough.

## Run

```bash
./run.sh
```

trains the chapter-10 MLP on Fashion-MNIST with AdamW and C-AdamW
(seed 42, lr 1e-3, weight decay 0.01, batch 32, 20 epochs) and writes
loss curves, test accuracy and the mask diagnostics to
`outputs/e1_fashion_mnist/`, plus the comparison figure
`fig_e1_fashion_mnist.png`.

## Layout

- `coptim/` — official C-AdamW from kyleliang919/C-Optim
  (commit a506ee3, MIT), vendored unmodified except for the package
  `__init__.py`. We run it with `foreach=False`, which is the classic
  per-parameter loop implementing the paper's Algorithm 2.
- `experiments/` — training and plotting scripts.
- `outputs/` — figures and per-run CSV/JSON logs.

All experiments run with seed 42 (multi-seed runs come with the D tasks).
