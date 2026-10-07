"""Ex3 Q3.1 and Q3.2 - AlphaDropout and MC Dropout on Fashion-MNIST.

Configuration per the assignment:
- Flatten 28x28 -> 784. Standardize inputs to zero mean, unit variance using
  training set statistics (applied to train and test).
- Three hidden layers, each with 100 neurons, SELU activation, LeCun normal
  initialization, and AlphaDropout rate 0.1.
- Output layer: 10 neurons with softmax.
- Nadam lr=0.001, sparse categorical crossentropy, EarlyStopping on val_loss
  (patience=5, restore_best_weights), epochs=50, batch_size=32.
- Subset: first 2000 training, first 400 test (no shuffling).
- Q3.2: MC Dropout - enable dropout during inference with model(X, training=True)
  and average predictions over 20 stochastic forward passes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:400], y_test[:400]

x_train = x_train / 255.0
x_test = x_test / 255.0

# Standardize using training subset statistics
train_mean = x_train.mean()
train_std = x_train.std()
x_train_std = (x_train - train_mean) / (train_std + 1e-7)
x_test_std = (x_test - train_mean) / (train_std + 1e-7)

model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(0.1),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(0.1),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(0.1),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train_std,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
)

# Q3.1: deterministic inference (dropout is off)
test_loss, test_acc_det = model.evaluate(x_test_std, y_test, verbose=0)

# Q3.2: MC Dropout - average over 20 stochastic passes
probs = []
for _ in range(20):
    probs.append(model(x_test_std, training=True))
probs_mean = np.mean(probs, axis=0)
mc_preds = probs_mean.argmax(axis=1)
mc_acc = (mc_preds == y_test).mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Fashion-MNIST AlphaDropout: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_loss.png", dpi=150)

print(model.count_params(), "parameters")
print("epochs run:", len(history.history["loss"]))
print(f"test loss (det): {test_loss:.4f}")
print(f"Q3.1 test accuracy (AlphaDropout, det): {test_acc_det * 100:.2f}%")
print(f"Q3.2 MC Dropout accuracy (20 passes): {mc_acc * 100:.2f}%")
print(f"saved outputs -> {OUT_DIR}")