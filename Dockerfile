# NOTE: This is a PUBLIC template Dockerfile.
# You MUST provide a base image that you can access.
# Example:
#   docker build --build-arg BASE_IMAGE=python:3.9-slim -t cls_service:latest .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System libs commonly needed by onnxruntime + opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps.
# Replace opencv-python with headless to avoid GUI dependencies.
COPY requirements.txt /app/requirements.txt
RUN sed -i 's/^opencv-python$/opencv-python-headless/' /app/requirements.txt \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy source code only. Model weights (e.g. best.onnx) are NOT included.
COPY app.py classifier.py /app/

EXPOSE 8000

# Start the service (PORT can be overridden by env).
CMD ["sh", "-c", "python3 -m gunicorn -w 2 -b 0.0.0.0:${PORT:-8000} app:app"]
