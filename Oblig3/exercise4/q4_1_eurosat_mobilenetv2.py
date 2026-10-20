"""Ex4 Q4.1 - Transfer learning on EuroSAT with MobileNetV2."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# Load EuroSAT with tfds.load as specified
ds_train, ds_test = tfds.load(
    'eurosat/rgb',
    split=['train[:2000]', 'train[2000:2500]'],
    as_supervised=True,
    shuffle_files=False,
)
