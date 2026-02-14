import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import TEST_DIR, TEST_BATCH_SIZE, BEST_MODEL_PATH
from data.preprocessing import preprocess_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ===================== LOAD MODEL =====================
model = load_model(BEST_MODEL_PATH)

# ===================== TEST GENERATOR =====================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=TEST_BATCH_SIZE,
    class_mode="binary",
    shuffle=False,  # IMPORTANT
)

# ===================== PREDICTIONS =====================
y_true = test_gen.classes

y_probs = model.predict(test_gen)
y_pred = (y_probs > 0.5).astype(int).reshape(-1)

# ===================== METRICS =====================
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📊 Evaluation Metrics on Test Set")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred NORMAL", "Pred PNEUMONIA"],
    yticklabels=["Actual NORMAL", "Actual PNEUMONIA"],
)
plt.title("Confusion Matrix - Test Set")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
