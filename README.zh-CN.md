[English](README.md) | [中文](README.zh-CN.md)

# cls_service — 电网图片时间/场景分类推理服务（ONNX）

最后更新：2026.3.3

该仓库是本人作为初学者实习的模拟训练，旨在展示学习成果。由于涉及内部平台与数据集的使用，故未提供部分文件。

本服务为一个轻量的 HTTP 推理服务，基于 Flask 与 ONNX Runtime，用于将电网相关照片按时间/场景进行分类，例如白天、夜晚、雾天、红外。

## 1. 任务目标

- 输入：一张或一个文件夹下的电网相关图片
- 输出：时间/场景分类标签与置信度

默认类别顺序：

1. `1-6-7点`
2. `2-夜晚`
3. `3-白天`
4. `4-红外`
5. `5-雾天`

## 2. 接口总览

| Method | Path | 请求类型 | 主要用途 | 响应类型 |
|---|---|---|---|---|
| GET | `/health` | 无 | 健康检查 | JSON |
| POST | `/predict` | `application/json`（Base64 单张图片） | **客户端图片**推理（推荐） | JSON |
| POST | `/predict_batch` | `application/json`（Base64 多张图片，等价于“上传文件夹”） | **客户端批量图片**推理 | text/plain（TSV） |

---

## 3. 权重文件

由于使用公司内部数据集训练，本仓库不提供 `best.onnx`。

- 默认路径：`./best.onnx`，与 `app.py` 同目录
- 或通过环境变量 `MODEL_PATH` 指定你自己的 ONNX 权重路径

## 4. 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_PATH=./best.onnx
python app.py
```

Base URL 占位符：`http://<HOST>:<PORT>`  
示例：`http://127.0.0.1:8000`

## 5. GET /health

**请求：**
```bash
curl http://<HOST>:<PORT>/health
```

**响应（JSON）：**
```json
{"status":"ok"}
```

## 6. POST /predict（上传单张图片推理）

### 6.1 请求
- Content-Type：`application/json`
- Body（JSON）：
  - `b64`：图片 Base64 字符串

### 6.2 响应
```json
{
  "label": "3-白天",
  "confidence": 0.9999996,
  "ms": 12.3
}
```

字段说明：
- `label`：分类结果（`序号-中文类别名`）
- `confidence`：置信度（0~1）
- `ms`：服务端推理耗时（毫秒）

### 6.3 示例

**Linux：**
```bash
curl -X POST http://<HOST>:<PORT>/predict \
  -H 'Content-Type: application/json' \
  -d '{"b64":"'"$(base64 -w0 /path/to/test.jpg)"'"}'
```

---

## 7. POST /predict_batch（上传包含多张图片的文件夹推理）

### 7.1 请求
- Content-Type：`application/json`
- Body（JSON）：
  - `files`：图片列表，每个元素为对象：
    - `name`：文件名
    - `b64`：图片 Base64 字符串

可选 query 参数：
- `limit`：只输出前 N 行（例如 `?limit=20`）
- `min_conf`：只输出置信度 >= 某值（例如 `?min_conf=0.9`）

### 7.2 响应（text/plain，TSV）
- 第 1 行：`total=<上传文件数>\tok=<成功行数>\tms=<耗时>`
- 从第 2 行起：每行一条结果  
  `confidence<TAB>label<TAB>filename`

示例：
```text
total=2\tok=2\tms=35.1
0.999999\t3-白天\ta.jpg
0.923457\t5-雾天\tb.jpg
```

### 7.3 示例

下面是一个的 Python 示例：

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

`Dockerfile` 为公开模板，需要你传入一个自己可访问的基础镜像。

构建：

```bash
docker build --build-arg BASE_IMAGE=python:3.9-slim -t cls_service:latest .
```

运行：

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/best.onnx \
  -v "$PWD/best.onnx:/app/best.onnx:ro" \
  cls_service:latest
```
