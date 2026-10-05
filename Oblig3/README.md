# Oblig 3 - Deep Neural Networks, Dropout and Transfer Learning

Course assignment IDATG2208. Six exercises over Fashion-MNIST, CIFAR-100,
EuroSAT and KMNIST, all built with Keras.

## Layout
- `exercise1/` - Exercise 1: Fashion-MNIST DNN (Q1.1)
- `exercise2/` - Exercise 2: CIFAR-100 coarse-label DNN (Q2.1)
- `exercise3/` - Exercise 3: AlphaDropout and MC Dropout (Q3.1, Q3.2)
- `exercise4/` - Exercise 4: MobileNetV2 transfer learning on EuroSAT (Q4.1)
- `exercise5/` - Exercise 5: deeper CNN on KMNIST (Q5.1)
- `exercise6/` - Exercise 6: SGD CNN, MC Dropout and epistemic uncertainty
  (Q6.1, Q6.2, Q6.3)

## How to run
TensorFlow has no wheels for the system Python 3.14, so the dependencies live
in a venv built on Python 3.10:

```bash
.venv/bin/python exercise1/q1_1_fashion_mnist.py
```

Every script seeds with `tf.keras.utils.set_random_seed(42)`, takes the subset
from the front of each split without shuffling, and writes its loss curves and
a small metrics CSV to its own `outputs/` folder.