import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MODEL_PATH, IMAGE_SIZE


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

MODEL = None
BACKBONE = None
TARGET_LAYER = None
HEAD_LAYERS = []


def load_imageguard_model():
    global MODEL, BACKBONE, TARGET_LAYER, HEAD_LAYERS

    if MODEL is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # The trained ImageGuard model contains a nested EfficientNetB0
    # backbone followed by the classification head.
    BACKBONE = MODEL.get_layer("efficientnetb0")

    # The final spatial feature map is the layer immediately before
    # EfficientNet's global pooling stage.
    TARGET_LAYER = BACKBONE.get_layer("top_activation")

    # Layers after the backbone form the ImageGuard classifier head.
    HEAD_LAYERS.clear()
    found = False
    for layer in MODEL.layers:
        if layer.name == BACKBONE.name:
            found = True
            continue
        if found:
            HEAD_LAYERS.append(layer)



def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def prepare_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Match the preprocessing used during CIFAKE training.
    # image_dataset_from_directory() uses bilinear resizing.
    array = np.asarray(image, dtype=np.float32)

    tensor = tf.convert_to_tensor(array, dtype=tf.float32)
    tensor = tf.image.resize(
        tensor,
        IMAGE_SIZE,
        method="bilinear"
    )

    return tf.expand_dims(tensor, axis=0).numpy()



def predict(image_batch: np.ndarray) -> float:
    prediction = MODEL.predict(image_batch, verbose=0)
    return float(prediction[0][0])



def make_gradcam(image_batch: np.ndarray) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the AI-generated probability."""
    feature_model = tf.keras.Model(
        inputs=BACKBONE.input,
        outputs=TARGET_LAYER.output,
    )

    with tf.GradientTape() as tape:
        conv_outputs = feature_model(image_batch, training=False)
        tape.watch(conv_outputs)

        x = conv_outputs
        passed_target = False
        for layer in BACKBONE.layers:
            if layer.name == TARGET_LAYER.name:
                passed_target = True
                continue
            if passed_target:
                x = layer(x, training=False)

        for layer in HEAD_LAYERS:
            x = layer(x, training=False)

        score = x[:, 0]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients could not be calculated for Grad-CAM.")

    weights = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    weights = weights[0]

    cam = tf.reduce_sum(conv_outputs * weights, axis=-1)
    cam = tf.maximum(cam, 0)

    max_value = tf.reduce_max(cam)
    cam = tf.where(max_value > 0, cam / max_value, tf.zeros_like(cam))
    cam = tf.image.resize(cam[..., tf.newaxis], IMAGE_SIZE)
    cam = tf.squeeze(cam, axis=-1)

    return cam.numpy()



def make_gradcam_overlay(original: Image.Image, cam: np.ndarray) -> str:
    import cv2

    base = np.asarray(original.resize(IMAGE_SIZE, Image.Resampling.LANCZOS), dtype=np.float32)
    heat = np.uint8(np.clip(cam, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)

    overlay = 0.55 * base + 0.45 * heat
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    image = Image.fromarray(overlay)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/api/health")
def health():
    try:
        load_imageguard_model()
        return jsonify({
            "status": "ok",
            "model": "EfficientNetB0",
            "modelLoaded": True,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "modelLoaded": False,
            "error": str(exc),
        }), 500

@app.get("/api/metrics")
def metrics():
    try:
        metrics_path = Path(__file__).parent.parent / "data" / "pixi_model_performance.json"

        if not metrics_path.exists():
            return jsonify({
                "error": "Performance metrics file not found."
            }), 404

        with metrics_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        def clean_json_values(value):
            if isinstance(value, dict):
                return {
                    key: clean_json_values(item)
                    for key, item in value.items()
                }

            if isinstance(value, list):
                return [
                    clean_json_values(item)
                    for item in value
                ]

            if isinstance(value, float) and not math.isfinite(value):
                return None

            return value

        data = clean_json_values(data)

        return jsonify(data)

    except Exception as exc:
        app.logger.exception("Failed to load performance metrics")
        return jsonify({
            "error": "Failed to load performance metrics.",
            "details": str(exc),
        }), 500

@app.get("/api/training-history")
def training_history():
    history_path = Path(__file__).resolve().parent / "imageguard_training_history.json"

    if not history_path.exists():
        return jsonify({
            "error": "Training history file not found"
        }), 404

    with history_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return jsonify(data)

@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()

    if "image" not in request.files:
        return jsonify({"error": "No image file was provided."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No image file was selected."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Unsupported image format. Use JPG, PNG, or WEBP."}), 400

    try:
        file_bytes = uploaded.read()
        if not file_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({"error": "Image is too large."}), 413

        load_imageguard_model()

        original = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        batch = prepare_image(file_bytes)

        probability_ai = predict(batch)
        probability_ai = float(np.clip(probability_ai, 0.0, 1.0))

        if probability_ai >= 0.5:
            prediction = "AI-GENERATED"
            confidence = probability_ai
        else:
            prediction = "REAL"
            confidence = 1.0 - probability_ai

        gradcam = None
        try:
            cam = make_gradcam(batch)
            gradcam = make_gradcam_overlay(original, cam)
        except Exception as gradcam_error:
            # Prediction remains valid even if explainability generation fails.
            app.logger.warning("Grad-CAM generation failed: %s", gradcam_error)

        processing_time = time.perf_counter() - started

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "model": "EfficientNetB0",
            "processingTime": round(processing_time, 3),
            "gradcam": gradcam,
        })

    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify({
            "error": "Image analysis failed.",
            "details": str(exc),
        }), 500


if __name__ == "__main__":
    load_imageguard_model()

    # Warm up TensorFlow and Grad-CAM before accepting requests.
    dummy_image = np.zeros(
        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        dtype=np.float32
    )
    predict(dummy_image)
    make_gradcam(dummy_image)

    print("ImageGuard backend running at http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health")
    app.run(host="127.0.0.1", port=5000, debug=False)
