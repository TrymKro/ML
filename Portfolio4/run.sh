#!/usr/bin/env bash
# One-command run for the E task: both optimizers on Fashion-MNIST, seed 42.
set -e
PY=.venv/bin/python
cd "$(dirname "$0")"

for opt in adamw c-adamw; do
  $PY experiments/train_fashion_mnist.py --optimizer "$opt" --seed 42
done
$PY experiments/plot_e1.py
