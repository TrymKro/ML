from pathlib import Path
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path('/home/tk/Desktop/ML/code/Oblig3/exercise4/outputs')
OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

base = '/home/tk/Downloads/eurosat_unzip/EuroSAT'
classes = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base,d))]
cls_map = {c:i for i,c in enumerate(classes)}
X,y=[],[]
for c in classes:
    p = os.path.join(base,c)
    for f in sorted(os.listdir(p)):
        if f.endswith('.jpg'):
            img = Image.open(os.path.join(p,f)).convert('RGB')
            X.append(np.array(img)); y.append(cls_map[c])
X = np.array(X,dtype=np.float32)/255.0
y = np.array(y)

# assignment: train[:2000], train[2000:2500] - but it's all train split in tfds
# tfds eurosat/rgb is a single 'train' split of 27000; our manual is in class order, so just slice
x_train_np, y_train_np = X[:2000], y[:2000]
x_test_np, y_test_np = X[2000:2500], y[2000:2500]

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
plt.figure(figsize=(6,4)); plt.plot(history.history['loss'],label='train'); plt.plot(history.history['val_loss'],label='val'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'q4_1_loss.png',dpi=150)
print(model.count_params()); print(len(history.history['loss'])); print(f'{test_acc*100:.2f}%')
