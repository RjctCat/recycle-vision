
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "recycle_mobilenet_colab.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

print("Entrada:", model.input_shape)
print("Salida:", model.output_shape)
