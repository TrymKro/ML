"""Ex5 Q5.1 - Deeper CNN on KMNIST."""
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

ds_train, ds_test = tfds.load(
    "kmnist", split=["train[:3000]", "test[:500]"], as_supervised=True, shuffle_files=False
)

x_train_np, y_train_np = [], []
for img, lbl in ds_train:
    x_train_np.append(img.numpy()); y_train_np.append(lbl.numpy())
x_test_np, y_test_np = [], []
for img, lbl in ds_test:
    x_test_np.append(img.numpy()); y_test_np.append(lbl.numpy())
x_train_np = np.array(x_train_np, dtype=np.float32)/255.0; y_train_np = np.array(y_train_np)
x_test_np = np.array(x_test_np, dtype=np.float32)/255.0; y_test_np = np.array(y_test_np)

model = keras.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    keras.layers.Conv2D(32,3,activation='relu'),
    keras.layers.Conv2D(32,3,activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64,3,activation='relu'),
    keras.layers.Conv2D(64,3,activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10,activation='softmax'),
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_np,y_train_np,epochs=15,batch_size=32,validation_split=0.1)
test_loss,test_acc = model.evaluate(x_test_np,y_test_np,verbose=0)
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(history.history['loss'],label='train'); ax.plot(history.history['val_loss'],label='val')
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR/'q5_1_loss.png',dpi=150)
print(model.count_params(),'params'); print('epochs',len(history.history['loss'])); print(f'test_acc={test_acc*100:.2f}%')
