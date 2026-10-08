"""Ex4 Q4.1 - Transfer learning on EuroSAT with MobileNetV2.

Configuration:
- tfds.load('eurosat/rgb', split=['train[:2000]','train[2000:2500]'],
  as_supervised=True, shuffle_files=False). First split = 2000 train, second = 500 test.
- Normalize pixel values to [0,1].
- MobileNetV2(include_top=False, weights='imagenet', input_shape=(64,64,3)),
  fully frozen.
- Head: GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(10, softmax).
- Adam lr=0.001, sparse categorical crossentropy, epochs=5, batch_size=32,
  validation_split=0.1 (applies everywhere per assignment).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(42)

# Load EuroSAT
ds_train, ds_test = tfds.load(
    "eurosat/rgb",
    split=["train[:2000]", "train[2000:2500]"],
    as_supervised=True,
    shuffle_files=False,
)

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 32


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


train_ds = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
test_ds = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

# Convert to arrays for validation_split with fit
# Easier: get numpy arrays
x_train_np, y_train_np = [], []
for img, lbl in ds_train:
    x_train_np.append(img.numpy())
    y_train_np.append(lbl.numpy())
x_test_np, y_test_np = [], []
for img, lbl in ds_test:
    x_test_np.append(img.numpy())
    y_test_np.append(lbl.numpy())
x_train_np = np.array(x_train_np, dtype=np.float32) / 255.0
y_train_np = np.array(y_train_np)
x_test_np = np.array(x_test_np, dtype=np.float32) / 255.0
y_test_np = np.array(y_test_np)

# Build model
base = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(64, 64, 3),
)
base.trainable = False

model = keras.Sequential(
    [
        keras.layers.Input(shape=(64, 64, 3)),
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train_np,
    y_train_np,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
)

test_loss, test_acc = model.evaluate(x_test_np, y_test_np, verbose=0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="training loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("EuroSAT MobileNetV2: training and validation loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q4_1_loss.png", dpi=150)

print(model.count_params(), "parameters")
print("epochs run:", len(history.history["loss"]))
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_acc * 100:.2f}%")
print(f"saved outputs -> {OUT_DIR}")