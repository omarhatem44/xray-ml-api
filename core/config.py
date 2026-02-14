from pathlib import Path

# ===================== PROJECT ROOT =====================
PROJECT_ROOT = Path(__file__).resolve().parent

# ===================== DATA =====================
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# ===================== IMAGE =====================
IMG_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

# ===================== MODELS =====================
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "cnn_pneumonia_best.h5"
FINAL_MODEL_PATH = MODELS_DIR / "cnn_pneumonia_final.h5"
