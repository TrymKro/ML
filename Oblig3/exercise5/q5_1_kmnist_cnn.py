"""Ex5 Q5.1 - KMNIST deeper CNN.

Loads KMNIST via tensorflow_datasets, uses first 3000 train / 500 test
samples (no shuffle). Normalizes to [0,1], adds channel dimension. Trains a
deeper CNN (two Conv2D+MaxPool blocks, Flatten, Dense(128), softmax) with
Adam (lr=0.001) for 15 epochs.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# Load KMNIST
ds_train_all, ds_test_all = tfds.load('kmnist', split=['train', 'test'], as_supervised=True, shuffle_files=False)
x_train_list, y_train_list = [], []
for img, lbl in tfds.as_numpy(ds_train_all):
    x_train_list.append(img); y_train_list.append(lbl)
x_test_list, y_test_list = [], []
for img, lbl in tfds.as_numpy(ds_test_all):
    x_test_list.append(img); y_test_list.append(lbl)
x_train = np.array(x_train_list, dtype=np.float32); y_train = np.array(y_train_list)
x_test = np.array(x_test_list, dtype=np.float32); y_test = np.array(y_test_list)

# Subset
x_train, y_train = x_train[:3000], y_train[:3000]
x_test, y_test = x_test[:500], y_test[:500]
x_train = x_train / 255.0; x_test = x_test / 255.0
x_train = x_train[..., np.newaxis]; x_test = x_test[..., np.newaxis]

# Model
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history['loss'], label='training loss')
ax.plot(history.history['val_loss'], label='validation loss')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_title('KMNIST deeper CNN: training and validation loss')
ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR / 'q5_1_loss.png', dpi=150)

print("=" * 40)
print("Q5.1 - KMNIST deeper CNN")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Epochs run: {len(history.history['loss'])}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
