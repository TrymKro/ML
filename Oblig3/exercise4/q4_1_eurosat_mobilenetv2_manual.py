"""Ex4 Q4.1 - Transfer learning on EuroSAT with MobileNetV2 (manual load).

Loads EuroSAT from data/EuroSAT/train.csv, resizes images to 64x64 and
normalizes to [0,1]. Uses the first 2000 train rows for training and the next
500 for testing (matching tfds splits). Freezes MobileNetV2 (ImageNet) backbone
and trains a small head with Adam (lr=0.001) for 5 epochs.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# Load the train split from train.csv
base = Path(__file__).resolve().parent / "data" / "EuroSAT"
print(f"Loading EuroSAT from {base / 'train.csv'}")
X,y=[],[]
with open(base / "train.csv", newline="") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i == 2500:
            break
        img = Image.open(base / row["Filename"]).convert("RGB").resize((64,64))
        X.append(np.array(img))
        y.append(int(row["Label"]))
X = np.array(X,dtype=np.float32)/255.0
y = np.array(y)

# Split as specified: train[:2000] and train[2000:2500]
x_train_np, y_train_np = X[:2000], y[:2000]
x_test_np, y_test_np = X[2000:2500], y[2000:2500]
print(f"Train samples: {len(y_train_np)}, test samples: {len(y_test_np)}")

# Model
base_m = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(64,64,3))
base_m.trainable = False
model = keras.Sequential([
    keras.layers.Input(shape=(64,64,3)),
    base_m,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_np,y_train_np,epochs=5,batch_size=32,validation_split=0.1)
test_loss,test_acc = model.evaluate(x_test_np,y_test_np,verbose=0)

# Plot
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(history.history['loss'],label='training loss')
ax.plot(history.history['val_loss'],label='validation loss')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_title('EuroSAT MobileNetV2: training and validation loss')
ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR/'q4_1_loss.png',dpi=150)

print("=" * 40)
print("Q4.1 - EuroSAT MobileNetV2 (frozen)")
print("=" * 40)
print(f"Total parameters: {model.count_params()}")
print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])}")
print(f"Epochs run: {len(history.history['loss'])}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
