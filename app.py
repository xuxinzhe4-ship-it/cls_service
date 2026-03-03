"""Flask HTTP inference service for ONNX image classification.

Endpoints:
  - GET  /health          : health check
  - POST /predict         : single image prediction (base64)
  - POST /predict_batch   : batch prediction (base64 list)
"""

from __future__ import annotations

import os
import time
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response


from classifier import DEFAULT_UNICODE_NAMES, ONNXClassifier

app = Flask(__name__)
app.json.ensure_ascii = False
app.json.compact = False    
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# Path to the ONNX model. By default it looks for "best.onnx" in the repo root.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "best.onnx"))
PORT = int(os.environ.get("PORT", "8000"))

clf = ONNXClassifier(MODEL_PATH, class_names=DEFAULT_UNICODE_NAMES)


def _decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes into an OpenCV BGR ndarray."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img


def _require_json() -> dict:
    """Require a JSON object body."""
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("invalid JSON body (expect an object)")
    return data


def _b64_to_bytes(s: str) -> bytes:
    """Parse base64 string (optionally with data URL prefix) into raw bytes."""
    if not isinstance(s, str) or not s.strip():
        raise ValueError("missing field: b64")
    s = s.strip()
    if "base64," in s:
        s = s.split("base64,", 1)[1]
    s = "".join(s.split())  
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        raise ValueError("invalid base64")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    try:
        payload = _require_json()
        data = _b64_to_bytes(payload.get("b64"))
        if not data:
            return jsonify({"error": "empty image"}), 400

        t0 = time.time()
        img_bgr = _decode_upload_to_bgr(data)
        label, conf = clf.predict(img_bgr)
        ms = (time.time() - t0) * 1000.0
        return jsonify({"label": label, "confidence": conf, "ms": ms})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/predict_batch")
def predict_batch():
    # Supported query params:
    # ?limit=20     Only return the first 20 rows
    # ?min_conf=0.9 Only keep results with confidence >= 0.9
    try:
        limit = int(request.args.get("limit") or 0)
        min_conf = float(request.args.get("min_conf") or 0.0)
    except ValueError:
        return jsonify({"error": "invalid query params: limit must be int, min_conf must be float"}), 400

    try:
        payload = _require_json()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return jsonify({"error": "missing field: files (expect a non-empty list)"}), 400

    t0 = time.time()
    rows = []
    ok = 0

    for i, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            rows.append(f"ERROR\t(invalid item, expect object)\t#{i}")
            continue

        name = str(item.get("name") or f"#{i}")
        try:
            data = _b64_to_bytes(item.get("b64"))
        except Exception as e:
            rows.append(f"ERROR\t{str(e)}\t{name}")
            continue

        if not data:
            rows.append(f"ERROR\t(empty image)\t{name}")
            continue

        try:
            img_bgr = _decode_upload_to_bgr(data)
            label, conf = clf.predict(img_bgr)

            if min_conf > 0 and float(conf) < min_conf:
                continue

            rows.append(f"{float(conf):.6f}\t{label}\t{name}")
            ok += 1

            if limit > 0 and len(rows) >= limit:
                break

        except Exception as e:
            rows.append(f"ERROR\t{str(e)}\t{name}")

    ms = (time.time() - t0) * 1000.0
    header = f"total={len(files)}\tok={ok}\tms={ms:.1f}"
    text = header + "\n" + "\n".join(rows) + ("\n" if rows else "")
    return Response(text, mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
