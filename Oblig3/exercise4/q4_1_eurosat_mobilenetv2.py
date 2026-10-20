"""Ex4 Q4.1 - Transfer learning on EuroSAT with MobileNetV2."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# Load EuroSAT with tfds.load as specified
try:
    ds_train, ds_test = tfds.load(
        'eurosat/rgb',
        split=['train[:2000]', 'train[2000:2500]'],
        as_supervised=True,
        shuffle_files=False,
    )
    # convert to arrays
    x_train = []
    y_train = []
    for img, lbl in tfds.as_numpy(ds_train):
        x_train.append(img)
        y_train.append(lbl)
    x_train = np.array(x_train).astype(np.float32) / 255.0
    y_train = np.array(y_train)
    x_test = []
    y_test = []
    for img, lbl in tfds.as_numpy(ds_test):
        x_test.append(img)
        y_test.append(lbl)
    x_test = np.array(x_test).astype(np.float32) / 255.0
    y_test = np.array(y_test)
except Exception:
    # fallback to local data
    base = '/home/tk/Desktop/ML/code/Oblig3/data/eurosat/EuroSAT'
    classes = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base,d))]
    cls_map = {c:i for i,c in enumerate(classes)}
    X,y=[],[]
    for c in classes:
        p = os.path.join(base,c)
        for f in sorted(os.listdir(p)):
            if f.endswith('.jpg'):
                img = Image.open(os.path.join(p,f)).convert('RGB').resize((64,64))
                X.append(np.array(img))
                y.append(cls_map[c])
    X = np.array(X,dtype=np.float32)/255.0; y = np.array(y)
    x_train,y_train = X[:2000], y[:2000]
    x_test,y_test = X[2000:2500], y[2000:2500]

base_m = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
base_m.trainable = False
model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    base_m,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history['loss'], label='training loss')
ax.plot(history.history['val_loss'], label='validation loss')
ax.set_xlabel('epoch'); ax.set_ylabel('loss')
ax.set_title('EuroSAT MobileNetV2: training and validation loss')
ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR / 'q4_1_loss.png', dpi=150)
print(model.count_params(), 'parameters')
print('epochs run:', len(history.history['loss']))
print(f'test loss: {test_loss:.4f}')
print(f'test accuracy: {test_acc * 100:.2f}%')
