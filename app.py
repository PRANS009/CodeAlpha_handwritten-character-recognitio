"""
Handwritten Alphabet Recognition — Flask API
===============================================
Serves the web UI and provides a REST endpoint for predictions.
Uses PyTorch for model inference.

Endpoints:
  GET  /          → Serve the frontend
  POST /predict   → Accept base64 canvas image, return top-3 predictions
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "handwritten_model.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_mapping.json")
IMG_SIZE = 28

app = Flask(__name__)
CORS(app)


# ──────────────────────────────────────────────
# CNN Model (must match training architecture)
# ──────────────────────────────────────────────
class HandwrittenCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# Load model & labels at startup
# ──────────────────────────────────────────────
print("🧠 Loading model …", flush=True)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
num_classes = checkpoint.get("num_classes", 10)

model = HandwrittenCNN(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✅ Model loaded (val_acc: {checkpoint.get('val_acc', 0)*100:.1f}%)", flush=True)

with open(LABEL_MAP_PATH, "r") as f:
    label_mapping = json.load(f)
    if num_classes != 26 or list(label_mapping.values()) != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise RuntimeError(
            "An alphabet model is not available. Run 'python train_model.py' first."
        )
print(f"📋 Loaded {len(label_mapping)} label mappings", flush=True)


def preprocess_canvas_image(data_url: str) -> torch.Tensor:
    """
    Convert a base64 data-URL from the HTML canvas into a tensor
    suitable for model inference.

    Steps:
      1. Decode base64 -> PIL Image -> Grayscale
      2. Handle dark or light background gracefully
      3. Threshold and find bounding box
      4. Resize to 20x20 preserving aspect ratio
      5. Center onto 28x28 black canvas (matches MNIST standard)
      6. Normalize pixel values to [0, 1]
    """
    # 1. Decode base64
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 2. Convert to grayscale array
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Auto-detect background brightness from corners
    h, w = arr.shape
    corner_samples = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corner_samples) > 127:
        # Light background -> invert so strokes are bright on dark background
        arr = 255.0 - arr

    # Canvas has dark background #0d0d15 (~14 luminance)
    # Subtract baseline and threshold
    arr = np.clip(arr - 25.0, 0.0, 255.0)
    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    img_clean = Image.fromarray(arr.astype(np.uint8))

    # 3. Crop to bounding box
    bbox = img_clean.getbbox()
    if bbox is None:
        return torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)

    cropped = img_clean.crop(bbox)

    # 4. Resize to fit 20x20 preserving aspect ratio
    cw, ch = cropped.size
    if cw > ch:
        new_w = 20
        new_h = max(1, int(round(ch * 20.0 / cw)))
    else:
        new_h = 20
        new_w = max(1, int(round(cw * 20.0 / ch)))

    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # 5. Centre on 28x28 canvas (0=black, 255=white)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (x_off, y_off))

    # 6. Normalize to [0, 1] tensor of shape [1, 1, 28, 28]
    final_arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(final_arr).unsqueeze(0).unsqueeze(0)
    return tensor


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Preprocess
        img_tensor = preprocess_canvas_image(data["image"])

        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        # Top-3 results
        top_values, top_indices = torch.topk(probabilities, 3)
        results = []
        for val, idx in zip(top_values, top_indices):
            char = label_mapping[str(idx.item())]
            confidence = val.item() * 100
            results.append({
                "character": char,
                "confidence": round(confidence, 2),
            })

        return jsonify({
            "success": True,
            "predictions": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("\n🚀 Starting Handwritten Alphabet Recognition Server …", flush=True)
    print(f"   Open http://127.0.0.1:{port} in your browser\n", flush=True)
    app.run(debug=False, host="0.0.0.0", port=port)
