"""Ex1 Q1.1 - Fashion-MNIST DNN.

Loads Fashion-MNIST, uses first 2000 training samples and first 400 test
samples (no shuffle). Normalizes to [0,1]. Trains a simple MLP with 3 hidden
layers (100 units, ELU, He normal) using Nadam (lr=0.001), early stopping
(patience=5, restore_best_weights=True), max 50 epochs. Reports parameters,
best epoch/val loss, and test accuracy/loss.
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

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Use subset as specified (first samples, no shuffle)
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:400], y_test[:400]

# Normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build model
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

# Compile
model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train with early stopping
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
)

# Find best epoch
best_idx = min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i])
best_epoch = best_idx + 1
best_val_loss = history.history["val_loss"][best_idx]

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Plot loss curves
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Fashion-MNIST DNN: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q1_1_loss.png", dpi=150)

# Print summary
print("=" * 40)
print("Q1.1 - Fashion-MNIST DNN")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Epochs run (of 50 max): {len(history.history['loss'])}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Saved outputs -> {OUT_DIR}")
print("=" * 40)
