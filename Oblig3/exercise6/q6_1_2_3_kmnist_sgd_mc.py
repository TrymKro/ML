"""Ex6 Q6.1–Q6.3 - KMNIST with SGD CNN, MC Dropout, and epistemic uncertainty.

Loads KMNIST, uses first 3000 train / 500 test samples (no shuffle).
Trains a simple CNN with SGD (momentum 0.9, lr 0.01), evaluates plain test
accuracy (Q6.1). For MC Dropout, keep Dropout active at inference and average
predictions over 20 stochastic forward passes (Q6.2). Compute epistemic
uncertainty as the mean predictive variance over samples and classes (Q6.3).
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

# Load KMNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # placeholder
# Use TensorFlow Datasets style? Or tf.keras.datasets.kmnist not available; load via tfds
import tensorflow_datasets as tfds
ds_train_all, ds_test_all = tfds.load('kmnist', split=['train', 'test'], as_supervised=True, shuffle_files=False)
x_train_list, y_train_list = [], []
for img, lbl in tfds.as_numpy(ds_train_all):
    x_train_list.append(img)
    y_train_list.append(lbl)
x_test_list, y_test_list = [], []
for img, lbl in tfds.as_numpy(ds_test_all):
    x_test_list.append(img)
    y_test_list.append(lbl)
x_train = np.array(x_train_list, dtype=np.float32)
y_train = np.array(y_train_list)
x_test = np.array(x_test_list, dtype=np.float32)
y_test = np.array(y_test_list)

# Use first 3000 train / 500 test
x_train, y_train = x_train[:3000], y_train[:3000]
x_test, y_test = x_test[:500], y_test[:500]
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build CNN with dropout for MC
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("=" * 40)
print("Q6.1 - Plain inference (SGD CNN)")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Epochs run: {len(history.history['loss'])}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
print()

# Q6.2: MC Dropout - keep dropout active at inference, average 20 passes
T = 20
probs_mc = []
for t in range(T):
    # training=True keeps dropout active
    p = model(x_test, training=True)
    probs_mc.append(p.numpy())
probs_mc = np.array(probs_mc)  # (T, N, C)
mean_probs = probs_mc.mean(axis=0)
y_pred_mc = mean_probs.argmax(axis=1)
mc_acc = (y_pred_mc == y_test).mean()
print("=" * 40)
print("Q6.2 - MC Dropout (20 stochastic passes)")
print("=" * 40)
print(f"Number of passes T: {T}")
print(f"Test accuracy (MC averaged): {mc_acc * 100:.2f}%")
print("=" * 40)
print()

# Q6.3: Epistemic uncertainty as mean predictive variance
# per-sample class variances over passes, then mean over classes and samples
class_vars = probs_mc.var(axis=0, ddof=0)  # (N, C)
epistemic = class_vars.mean()
print("=" * 40)
print("Q6.3 - Epistemic uncertainty")
print("=" * 40)
print(f"Mean predictive variance (ddof=0, mean over classes & samples): {epistemic:.3f}")
print("=" * 40)

# Plot loss curves
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history['loss'], label='training loss')
ax.plot(history.history['val_loss'], label='validation loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('KMNIST SGD CNN: training and validation loss')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'q6_loss.png', dpi=150)
