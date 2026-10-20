"""Ex2 Q2.1 - CIFAR-100 coarse DNN.

Loads CIFAR-100 with label_mode='coarse' (20 superclasses), uses first 5000
training samples and first 1000 test samples (no shuffle). Normalizes to
[0,1]. Trains a simple MLP (Flatten + Dense(300,relu) + Dense(100,relu) +
Dense(20,softmax)) with SGD (lr=1e-3), early stopping (patience=5,
restore_best_weights=True), max 50 epochs.
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

# Load CIFAR-100 coarse labels
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="coarse")
# Use subset as specified
x_train, y_train = x_train[:5000], y_train[:5000].flatten()
x_test, y_test = x_test[:1000], y_test[:1000].flatten()
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model
model = keras.Sequential(
    [
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(20, activation="softmax"),
    ]
)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
)

# Best epoch
best_idx = min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i])
best_epoch = best_idx + 1
best_val_loss = history.history["val_loss"][best_idx]

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("CIFAR-100 coarse: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_1_loss.png", dpi=150)

# Print
print("=" * 40)
print("Q2.1 - CIFAR-100 coarse DNN")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Epochs run (of 50 max): {len(history.history['loss'])}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print(f"Saved outputs -> {OUT_DIR}")
print("=" * 40)
