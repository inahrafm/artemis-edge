"""
server/inference.py
====================
YOLO26x GPU inference wrapper untuk ARTEMIS v2 server.
Dipindahkan dari server_inference_v2.py — logika tidak berubah.
"""

import logging
import time
from io import BytesIO
from typing import Dict

import numpy as np

from shared.constants import CLASS_FIRE, CLASS_SMOKE, IMG_SIZE, DEFAULT_THRESHOLDS

log = logging.getLogger("artemis.server.inference")

_model      = None
_model_path = None
_device     = "cuda"
_thresholds = {}


def load_model(model_path: str, device: str = "cuda"):
    """Load YOLO26x dan jalankan 3x warmup inference."""
    global _model, _model_path, _device
    from ultralytics import YOLO
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA tidak tersedia, fallback ke CPU")
        device = "cpu"

    log.info(f"Loading model: {model_path} ({device})")
    t0    = time.perf_counter()
    model = YOLO(model_path)
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, device=device, verbose=False, conf=0.001, imgsz=IMG_SIZE)
    elapsed = (time.perf_counter() - t0) * 1000

    _model      = model
    _model_path = model_path
    _device     = device
    log.info(f"Model siap dalam {elapsed:.0f} ms (termasuk 3x warmup)")


def set_thresholds(thresholds: Dict):
    global _thresholds
    _thresholds = thresholds


def get_thresholds() -> Dict:
    return dict(_thresholds) if _thresholds else dict(DEFAULT_THRESHOLDS["server_model"])


def run_inference(image_bytes: bytes) -> Dict:
    """
    Jalankan YOLOv26x inference pada image bytes.

    Returns dict dengan:
        confmax_fire, confavg_fire, count_fire,
        confmax_smoke, confavg_smoke, count_smoke,
        decision, server_inference_ms
    """
    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    t0      = time.perf_counter()
    results = _model.predict(arr, device=_device, verbose=False,
                             conf=0.001, imgsz=IMG_SIZE)
    inf_ms  = (time.perf_counter() - t0) * 1000

    fire_c  = []
    smoke_c = []
    result  = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        for j in range(len(result.boxes)):
            cls_  = int(result.boxes.cls[j].item())
            conf_ = float(result.boxes.conf[j].item())
            if cls_ == CLASS_FIRE:    fire_c.append(conf_)
            elif cls_ == CLASS_SMOKE: smoke_c.append(conf_)

    cmf = float(max(fire_c))      if fire_c  else 0.0
    caf = float(np.mean(fire_c))  if fire_c  else 0.0
    cms = float(max(smoke_c))     if smoke_c else 0.0
    cas = float(np.mean(smoke_c)) if smoke_c else 0.0

    thresh = get_thresholds()
    tl_f   = thresh.get("fire_local",  DEFAULT_THRESHOLDS["server_model"]["fire_local"])
    tl_s   = thresh.get("smoke_local", DEFAULT_THRESHOLDS["server_model"]["smoke_local"])
    td_f   = thresh.get("fire_drop",   DEFAULT_THRESHOLDS["server_model"]["fire_drop"])
    td_s   = thresh.get("smoke_drop",  DEFAULT_THRESHOLDS["server_model"]["smoke_drop"])

    if   cmf >= tl_f:               decision = "FIRE"
    elif cms >= tl_s:               decision = "SMOKE"
    elif cmf < td_f and cms < td_s: decision = "NONE"
    else:                           decision = "UNCERTAIN"

    return {
        "confmax_fire":        round(cmf, 4),
        "confavg_fire":        round(caf, 4),
        "count_fire":          len(fire_c),
        "confmax_smoke":       round(cms, 4),
        "confavg_smoke":       round(cas, 4),
        "count_smoke":         len(smoke_c),
        "decision":            decision,
        "server_inference_ms": round(inf_ms, 2),
    }


def is_loaded() -> bool:
    return _model is not None
