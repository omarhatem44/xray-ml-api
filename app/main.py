from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from inference import predict

app = FastAPI(title="Chest X-ray Pneumonia API")

# ---------------- LOGGING CONFIG ----------------

if not os.path.exists("logs"):
    os.makedirs("logs")

logger = logging.getLogger("xray-api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# File handler (rotating)
file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=3
)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------- CONSTANTS ----------------

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# ---------------- HEALTH ENDPOINT ----------------

@app.get("/health")
def health():
    return {"status": "running"}


# ---------------- PREDICT ENDPOINT ----------------

@app.post("/predict")
async def predict_xray(file: UploadFile, request: Request):

    start_time = time.time()

    logger.info(f"Incoming request from {request.client.host}")
    logger.info(f"File received: {file.filename}")

    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        logger.warning("Invalid file type attempted.")
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are allowed."
        )

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        logger.warning("File too large.")
        raise HTTPException(
            status_code=400,
            detail="File too large. Max size is 5MB."
        )

    try:
        result = predict(image_bytes)

        latency = time.time() - start_time

        logger.info(
            f"Prediction: {result['prediction']} | "
            f"Confidence: {result['confidence']:.4f} | "
            f"Latency: {latency:.3f}s"
        )

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_version": "v1",
            "threshold": 0.5
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed."}
        )
