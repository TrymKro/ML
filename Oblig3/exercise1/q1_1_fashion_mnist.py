"""Ex1 Q1.1 - Deep neural network on Fashion-MNIST.

The network is the one the assignment specifies: 784 flattened inputs, three
hidden layers of 100 ELU neurons with He normal initialization, and a 10-way
softmax head. Nadam at lr=0.001, sparse categorical crossentropy, early
stopping on the validation loss with the best weights restored. Only the first
2000 training and 400 test images are used, taken from the front of each split
so the subset is reproducible.

Writes the train/validation loss curves and the test accuracy to outputs/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

# The subset the assignment fixes, cut from the front of the split.
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:400], y_test[:400]

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dense(10, activation="softmax"),
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
    batch_size=32,
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
ax.set_title("Fashion-MNIST DNN: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q1_1_loss.png", dpi=150)

print(model.count_params(), "parameters")
print("epochs run:", len(history.history["loss"]))
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_acc * 100:.2f}%")
print(f"saved outputs -> {OUT_DIR}")