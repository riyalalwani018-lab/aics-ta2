import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "imageguard_efficientnetb0.keras"

TEST_DIR = Path(
    r"C:\Users\Lenovo\Downloads\CIFAKE_Test\test"
)

OUTPUT_PATH = BASE_DIR / "performance_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )


print("=" * 60)
print("ImageGuard - CIFAKE Model Evaluation")
print("=" * 60)

print(f"Model: {MODEL_PATH}")
print(f"Test dataset: {TEST_DIR}")
print()


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("Loading CIFAKE test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=["REAL", "FAKE"],
)

class_names = test_ds.class_names

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print()
print("Class mapping:")
print(class_names)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ImageGuard EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print()


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("Running predictions on CIFAKE test set...")
print("This may take a few minutes.")

y_true = []
y_prob = []

start_time = time.perf_counter()

for images, labels in test_ds:
    predictions = model(images, training=False).numpy().reshape(-1)

    y_true.extend(labels.numpy().astype(int).reshape(-1))
    y_prob.extend(predictions)


end_time = time.perf_counter()

total_time = end_time - start_time

y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)


# ============================================================
# BASIC INFORMATION
# ============================================================

total_images = len(y_true)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"Total test images: {total_images}")
print(f"Evaluation time: {total_time:.2f} seconds")
print(
    f"Average inference time/image: "
    f"{(total_time / total_images) * 1000:.2f} ms"
)
print()


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("METRICS")
print("-" * 60)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1 Score : {f1:.6f}")
print(f"ROC-AUC  : {auc:.6f}")

print()

print("CONFUSION MATRIX")
print("-" * 60)

print(cm)

print()
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print()

print("ROC POINTS")
print("-" * 60)

print(f"Number of ROC points: {len(fpr)}")


# ============================================================
# CREATE JSON DATA
# ============================================================

metrics = {
    "model": "EfficientNetB0",

    "dataset": {
        "name": "CIFAKE",
        "total_images": 120000,
        "train_images": 100000,
        "test_images": total_images,
        "classes": [
            "REAL",
            "AI-GENERATED"
        ],
        "test_split": {
            "REAL": int(np.sum(y_true == 0)),
            "AI-GENERATED": int(np.sum(y_true == 1))
        }
    },

    "input_size": [
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    ],

    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc)
    },

    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "matrix": [
            [
                int(tn),
                int(fp)
            ],
            [
                int(fn),
                int(tp)
            ]
        ]
    },

    "inference": {
        "total_time_seconds": float(total_time),
        "mean_time_ms": float(
            (total_time / total_images) * 1000
        )
    },

    "roc_curve": {
        "fpr": [
            float(x)
            for x in fpr
        ],

        "tpr": [
            float(x)
            for x in tpr
        ],

        "thresholds": [
            float(x)
            for x in thresholds
        ]
    },

    "architecture_comparison": {
        "EfficientNetB0": {
            "status": "evaluated",
            "accuracy": float(accuracy),
            "auc": float(auc)
        },

        "ResNet50": {
            "status": "not_evaluated"
        },

        "Custom CNN": {
            "status": "not_evaluated"
        },

        "MobileNetV3": {
            "status": "not_evaluated"
        }
    }
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=2
    )


print()
print("=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(
    f"File created:\n{OUTPUT_PATH}"
)

print()
print("You can now use performance_metrics.json")
print("for the ImageGuard Model Performance page.")