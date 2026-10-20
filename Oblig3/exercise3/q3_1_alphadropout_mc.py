"""Ex3 Q3.1–Q3.2 - Fashion-MNIST AlphaDropout and MC Dropout.

Uses first 2000 train / 400 test samples (no shuffle). Standardizes using
training subset mean/std. Builds a network with SELU activations and
AlphaDropout (for self-normalizing). Q3.1 reports deterministic test
accuracy. Q3.2 does Monte Carlo Dropout by keeping AlphaDropout active at
inference (model(X, training=True)) and averages predictions over 20 passes.
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

# Load
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:400], y_test[:400]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Standardize with training stats
mean = x_train.mean()
std = x_train.std() + 1e-7
x_train_s = (x_train - mean) / std
x_test_s = (x_test - mean) / std

# Model with SELU + AlphaDropout
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
history = model.fit(x_train_s, y_train, epochs=50, batch_size=32, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
test_loss, test_acc = model.evaluate(x_test_s, y_test, verbose=0)

print("=" * 40)
print("Q3.1 - AlphaDropout (deterministic)")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Epochs run: {len(history.history['loss'])}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
print()

# Q3.2: MC Dropout with AlphaDropout active
T = 20
probs_mc = []
for t in range(T):
    p = model(x_test_s, training=True)
    probs_mc.append(p.numpy())
probs_mc = np.array(probs_mc)
mean_probs = probs_mc.mean(axis=0)
y_pred_mc = mean_probs.argmax(axis=1)
mc_acc = (y_pred_mc == y_test).mean()
print("=" * 40)
print("Q3.2 - MC Dropout (20 passes, training=True)")
print("=" * 40)
print(f"Number of passes T: {T}")
print(f"Test accuracy (MC averaged): {mc_acc * 100:.2f}%")
print("=" * 40)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Fashion-MNIST AlphaDropout: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_loss.png", dpi=150)
