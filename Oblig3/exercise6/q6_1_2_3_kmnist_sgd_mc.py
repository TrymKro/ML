"""Ex6 Q6.1-6.3 - KMNIST with SGD, MC Dropout and epistemic uncertainty."""
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
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64,3,activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10,activation='softmax'),
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train_np,y_train_np,epochs=15,batch_size=32,validation_split=0.1)
test_loss, test_acc_plain = model.evaluate(x_test_np,y_test_np,verbose=0)
probs = []
for _ in range(20):
    probs.append(model(x_test_np, training=True))
probs = np.array(probs)
probs_mean = probs.mean(axis=0)
mc_preds = probs_mean.argmax(axis=1)
mc_acc = (mc_preds == y_test_np).mean()
class_var = probs.var(axis=0, ddof=0)
sample_mean_var = class_var.mean(axis=1)
epistemic_mean = sample_mean_var.mean()
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(history.history['loss'],label='train'); ax.plot(history.history['val_loss'],label='val')
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR/'q6_loss.png',dpi=150)
print(model.count_params()); print(len(history.history['loss'])); print(f'{test_acc_plain*100:.2f} {mc_acc*100:.2f} {epistemic_mean:.3f}')
