[English](README.md) | [中文](README.zh-CN.md)

# cls_service — Power-grid Image Time/Scene Classification Inference Service (ONNX)

Last updated: 2026-03-03

This repository is an internship practice project created by a beginner to demonstrate learning progress. Some files are not provided due to the use of an internal platform and dataset.

This service is a lightweight HTTP inference server built with Flask and ONNX Runtime. It classifies power-grid related photos into time/scene categories such as daytime, night, fog, and infrared.

## 1. Task Goal

- Input: one image or a folder of power-grid related images
- Output: a time/scene label and confidence score

Default class order:

1. `1-6-7点`
2. `2-夜晚`
3. `3-白天`
4. `4-红外`
5. `5-雾天`

## 2. API Overview

| Method | Path | Request Type | Usage | Response Type |
|---|---|---|---|---|
| GET | `/health` | None | Health check | JSON |
| POST | `/predict` | `application/json` (single image in Base64) | Single-image inference (**recommended**) | JSON |
| POST | `/predict_batch` | `application/json` (multiple images in Base64, equivalent to “folder upload”) | Batch inference | text/plain (TSV) |

---

## 3. Model Weights

`best.onnx` is NOT included in this repository because it was trained with an internal dataset.

- Default path: `./best.onnx` next to `app.py`
- Or set `MODEL_PATH` to your own ONNX model path

## 4. Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_PATH=./best.onnx
python app.py
```

Base URL placeholder: `http://<HOST>:<PORT>`  
Example: `http://127.0.0.1:8000`

## 5. GET /health

**Request:**
```bash
curl http://<HOST>:<PORT>/health
```

**Response (JSON):**
```json
{"status":"ok"}
```

## 6. POST /predict (Single Image)

### 6.1 Request
- Content-Type: `application/json`
- Body (JSON):
  - `b64`: Base64 string of the image

### 6.2 Response
```json
{
  "label": "3-白天",
  "confidence": 0.9999996,
  "ms": 12.3
}
```

Fields:
- `label`: classification result (`index-ChineseClassName`)
- `confidence`: confidence score (0~1)
- `ms`: server-side inference time (ms)

### 6.3 Example

**Linux:**
```bash
curl -X POST http://<HOST>:<PORT>/predict \
  -H 'Content-Type: application/json' \
  -d '{"b64":"'"$(base64 -w0 /path/to/test.jpg)"'"}'
```

---

## 7. POST /predict_batch (Batch / “Folder” Inference)

### 7.1 Request
- Content-Type: `application/json`
- Body (JSON):
  - `files`: a list of images. Each item is an object:
    - `name`: filename 
    - `b64`: image Base64 string

Optional query parameters:
- `limit`: only output the first N rows (e.g. `?limit=20`)
- `min_conf`: only keep results with confidence >= threshold (e.g. `?min_conf=0.9`)

### 7.2 Response (text/plain, TSV)
- Line 1: `total=<num_uploaded>\tok=<num_ok>\tms=<time_ms>`
- From line 2: one result per line  
  `confidence<TAB>label<TAB>filename`

Example:
```text
total=2\tok=2\tms=35.1
0.999999\t3-白天\ta.jpg
0.923457\t5-雾天\tb.jpg
```

### 7.3 Python Example

```python
import os, base64, requests

URL = "http://<HOST>:<PORT>/predict_batch"
DIR = "/path/to/folder"

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode("utf-8")

imgs = [
    os.path.join(DIR, f)
    for f in os.listdir(DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"))
]
payload = {"files": [{"name": os.path.basename(p), "b64": b64(p)} for p in sorted(imgs)]}

r = requests.post(URL, json=payload)
print(r.text)
```

## 8. Docker

`Dockerfile` is a public template. You must pass a base image you can access.

Build:
```bash
docker build --build-arg BASE_IMAGE=python:3.9-slim -t cls_service:latest .
```

Run:
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/best.onnx \
  -v "$PWD/best.onnx:/app/best.onnx:ro" \
  cls_service:latest
```
