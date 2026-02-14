from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from logging.handlers import RotatingFileHandler
import logging
import os
import time

from app.inference import predict


app = FastAPI(title="Chest X-ray Pneumonia API")

# ======================================================
# LOGGING CONFIGURATION
# ======================================================

if not os.path.exists("logs"):
    os.makedirs("logs")

logger = logging.getLogger("xray-api")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers (important in reload)
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotating file handler (5MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ======================================================
# CONSTANTS
# ======================================================

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# ======================================================
# MIDDLEWARE - Request Logging
# ======================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.4f}s "
        f"client={request.client.host}"
    )

    return response


# ======================================================
# GLOBAL EXCEPTION HANDLER
# ======================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled error on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"}
    )


# ======================================================
# HEALTH CHECK ENDPOINT
# ======================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "xray-api",
        "version": "v1"
    }


# ======================================================
# PREDICT ENDPOINT
# ======================================================

@app.post("/predict")
async def predict_xray(file: UploadFile, request: Request):

    logger.info(f"Prediction request received: filename={file.filename}")

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
        start_time = time.time()

        result = predict(image_bytes)

        latency = time.time() - start_time

        logger.info(
            f"Prediction={result['prediction']} "
            f"confidence={result['confidence']:.4f} "
            f"latency={latency:.3f}s"
        )

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_version": "v1",
            "threshold": 0.5,
            "latency_seconds": round(latency, 4)
        }

    except Exception as e:
        logger.error("Prediction failed.", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )
