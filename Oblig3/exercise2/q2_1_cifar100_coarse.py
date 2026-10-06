"""Ex2 Q2.1 - Deep neural network on CIFAR-100 (coarse labels).

CIFAR-100 with label_mode='coarse' has 20 superclasses. Images are 32x32 RGB,
flattened to 3072 features. The network: four hidden layers of 256 ELU neurons
each with He normal initialization, followed by a 20-way softmax head. Nadam
at lr=0.001, sparse categorical crossentropy, EarlyStopping on validation loss
(patience=5, restore_best_weights), epochs=50, batch_size=128, validation_split
= 0.1. Subset: first 5000 training and first 1000 test samples, no shuffling
before slicing.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
    label_mode="coarse"
)
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:1000], y_test[:1000]

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential(
    [
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(20, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("CIFAR-100 coarse: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_1_loss.png", dpi=150)

print(model.count_params(), "parameters")
print("epochs run:", len(history.history["loss"]))
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_acc * 100:.2f}%")
print(f"saved outputs -> {OUT_DIR}")