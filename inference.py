import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

from data.preprocessing import preprocess_image

MODEL_PATH = "models/cnn_pneumonia_best.h5"

# Load model once at startup
model = load_model(MODEL_PATH)

def predict(image_bytes: bytes):

    # Convert bytes → PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to numpy
    image = np.array(image)

    # Apply same preprocessing as training
    image = preprocess_image(image)

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    # Predict
    prob = model.predict(image, verbose=0)[0][0]

    if prob > 0.5:
        label = "PNEUMONIA"
        confidence = float(prob)
    else:
        label = "NORMAL"
        confidence = float(1 - prob)

    return {
        "prediction": label,
        "confidence": confidence
    }
