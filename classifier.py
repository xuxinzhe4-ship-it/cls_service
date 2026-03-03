from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_UNICODE_NAMES = ["6-7点", "夜晚", "白天", "红外", "雾天"]


def collect_images(source: str) -> List[str]:
    """Collect image paths from a file or recursively from a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if os.path.isdir(source):
        out: List[str] = []
        for root, _, files in os.walk(source):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    return [source]


class ONNXClassifier:
    """ONNX Runtime classifier with lightweight pre/post-processing.

    Notes:
      - Input is OpenCV BGR ndarray.
      - Output is (label, confidence).
    """
    def __init__(
        self,
        onnx_path: str,
        class_names: Optional[List[str]] = None,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
        providers: Optional[List[str]] = None,
    ):
        avail = ort.get_available_providers()
        if providers is None:
            prefer = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in prefer if p in avail] or avail
        else:
            providers = [p for p in providers if p in avail] or avail

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names or DEFAULT_UNICODE_NAMES

        if imgsz is None:
            auto = self._infer_imgsz_from_onnx()
            self.imgsz: Union[int, Tuple[int, int]] = auto if auto is not None else 224
        else:
            self.imgsz = imgsz

    def _resize_shortest_edge_pil(self, img: Image.Image, target: int, interpolation: int) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Invalid image with zero width/height")
        short = min(w, h)
        if short == target:
            return img
        scale = float(target) / float(short)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=interpolation)

    def _center_crop_pil(self, img: Image.Image, crop_hw: Tuple[int, int]) -> Image.Image:
        crop_h, crop_w = crop_hw
        w, h = img.size
        if crop_w > w or crop_h > h:
            return img.resize((crop_w, crop_h), resample=Image.BILINEAR)
        left = int(round((w - crop_w) / 2.0))
        top = int(round((h - crop_h) / 2.0))
        right = left + crop_w
        bottom = top + crop_h
        return img.crop((left, top, right, bottom))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _infer_imgsz_from_onnx(self) -> Optional[Tuple[int, int]]:
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return (h, w)
        return None

    # Preprocess
    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image input")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        interp = Image.BILINEAR
        if isinstance(self.imgsz, int):
            pil = self._resize_shortest_edge_pil(pil, self.imgsz, interp)
            pil = self._center_crop_pil(pil, (self.imgsz, self.imgsz))
        else:
            out_h, out_w = int(self.imgsz[0]), int(self.imgsz[1])
            pil = pil.resize((out_w, out_h), resample=interp)
            pil = self._center_crop_pil(pil, (out_h, out_w))

        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    # Postprocess
    def postprocess(self, raw_output) -> Tuple[int, float]:
        raw = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
        raw = np.asarray(raw)

        if raw.ndim == 1:
            raw = raw[None, :]

        row_sum = raw.sum(axis=1)
        looks_like_probs = (raw.min() >= -1e-6) and (raw.max() <= 1.0 + 1e-3) and np.allclose(row_sum, 1.0, atol=1e-2)
        probs = raw.astype(np.float32) if looks_like_probs else self._softmax(raw.astype(np.float32))

        p = probs[0]
        idx = int(np.argmax(p))
        conf = float(p[idx])
        return idx, conf

    def predict(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        x = self.preprocess(img_bgr)
        raw = self.session.run(None, {self.input_name: x})
        idx, conf = self.postprocess(raw)

        name = self.class_names[idx] if 0 <= idx < len(self.class_names) else str(idx)
        label = f"{idx + 1}-{name}"
        return label, conf


def predict_path(clf: ONNXClassifier, path: str) -> Tuple[str, float]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return clf.predict(img)


def predict_source(clf: ONNXClassifier, source: str) -> List[Tuple[str, str, float]]:
    paths = collect_images(source)
    out: List[Tuple[str, str, float]] = []
    for p in paths:
        label, conf = predict_path(clf, p)
        out.append((p, label, conf))
    return out
