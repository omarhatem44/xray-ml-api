import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE,
    BEST_MODEL_PATH, FINAL_MODEL_PATH
)
from data.preprocessing import preprocess_image

# ===================== GENERATORS =====================

def build_train_gen():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=TRAIN_BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )


def build_val_gen():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    return datagen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=VAL_BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )


def build_test_gen():
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    return datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=TEST_BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

# ===================== MODEL =====================

def build_cnn_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===================== TRAINING =====================

def main():
    print("[INFO] Building generators...")
    train_gen = build_train_gen()
    val_gen = build_val_gen()
    test_gen = build_test_gen()

    steps_per_epoch = math.ceil(train_gen.n / TRAIN_BATCH_SIZE)
    val_steps = math.ceil(val_gen.n / VAL_BATCH_SIZE)
    test_steps = math.ceil(test_gen.n / TEST_BATCH_SIZE)

    # ===== Class weights =====
    labels = train_gen.classes
    normal_count = np.sum(labels == 0)
    pneu_count = np.sum(labels == 1)
    total = normal_count + pneu_count

    class_weight = {
        0: total / (2.0 * normal_count),
        1: total / (2.0 * pneu_count),
    }

    print("[INFO] Class weights:", class_weight)

    model = build_cnn_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("[INFO] Training started...")
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print("[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, steps=test_steps)
    print(f"[RESULT] Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")

    model.save(FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
