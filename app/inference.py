import os
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from data.preprocessing import preprocess_image

MODEL_PATH = "models/cnn_pneumonia_best.h5"
MODEL_URL = "https://huggingface.co/omarhatem22/xray-pneumonia-model/resolve/main/cnn_pneumonia_best.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from HuggingFace...")
    os.makedirs("models", exist_ok=True)

    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Model downloaded successfully.")

# Load model once
model = load_model(MODEL_PATH)


def predict(image_bytes: bytes):
    import numpy as np
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    image = preprocess_image(image)
    image = tf.expand_dims(image, axis=0)

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
