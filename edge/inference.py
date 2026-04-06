"""
edge/inference.py
=================
YOLO26n wrapper untuk edge inference di Raspberry Pi.
Support dua format: ONNX FP32 (Pi3) dan TFLite FP32 (Pi4B, Pi5).

Refactored dari EdgeModel di phase_f_pi_evaluation.py dengan:
- Interface yang lebih bersih
- Breakdown latensi tetap ada (disk_read, preprocess, inference)
- Tidak ada perubahan logika inference
"""

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from shared.constants import CLASS_FIRE, CLASS_SMOKE, IMG_SIZE

log = logging.getLogger("artemis.edge.inference")


class EdgeInference:
    """
    YOLO26n inference wrapper untuk edge device.

    Mendukung ONNX (Pi3) dan TFLite (Pi4B, Pi5).
    Setiap call ke infer() menghasilkan detections + breakdown latensi.
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Args:
            model_path: path ke file model (.onnx atau .tflite)
            model_type: 'onnx' atau 'tflite'
        """
        if model_type not in ("onnx", "tflite"):
            raise ValueError(f"model_type harus 'onnx' atau 'tflite', bukan '{model_type}'")

        self.model_path = model_path
        self.model_type = model_type
        self._session   = None   # ONNX
        self._interp    = None   # TFLite
        self._load(model_path)

    def _load(self, path: str):
        if self.model_type == "onnx":
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"])
            log.info(f"ONNX model loaded: {path}")

        elif self.model_type == "tflite":
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            self._interp = tflite.Interpreter(
                model_path=path, num_threads=4)
            self._interp.allocate_tensors()
            log.info(f"TFLite model loaded: {path}")

    # ── Public interface ──────────────────────────────────────────────────────

    def infer(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Jalankan inference dengan breakdown latensi lengkap.

        Args:
            image_path: path ke file gambar JPEG

        Returns:
            (detections, latency_breakdown)

            detections: list of {"class_id": int, "confidence": float}
            latency_breakdown: {
                "disk_read_ms":      float,
                "preprocess_ms":     float,
                "edge_inference_ms": float,
                "edge_total_ms":     float,   # preprocess + inference
            }
        """
        raw,    disk_ms = self._read_image(image_path)
        inp,    pre_ms  = self._preprocess(raw)
        output, inf_ms  = self._forward(inp)
        dets            = self._parse(output)

        return dets, {
            "disk_read_ms":      round(disk_ms, 3),
            "preprocess_ms":     round(pre_ms,  3),
            "edge_inference_ms": round(inf_ms,  3),
            "edge_total_ms":     round(pre_ms + inf_ms, 3),
        }

    def read_raw(self, image_path: str) -> Tuple[bytes, float]:
        """Baca file dari disk saja (untuk server_only yang tidak butuh inference)."""
        return self._read_image(image_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_image(self, image_path: str) -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        with open(image_path, "rb") as f:
            raw = f.read()
        return raw, (time.perf_counter() - t0) * 1000

    def _preprocess(self, raw_bytes: bytes) -> Tuple[np.ndarray, float]:
        from PIL import Image
        t0  = time.perf_counter()
        img = Image.open(BytesIO(raw_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.model_type == "tflite":
            out = arr[np.newaxis, ...]                    # (1, H, W, C)
        else:
            out = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, C, H, W)

        return out, (time.perf_counter() - t0) * 1000

    def _forward(self, inp: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()

        if self.model_type == "onnx":
            name   = self._session.get_inputs()[0].name
            output = self._session.run(None, {name: inp})[0]

        else:  # tflite
            inp_d  = self._interp.get_input_details()[0]
            out_d  = self._interp.get_output_details()[0]
            self._interp.set_tensor(inp_d["index"], inp)
            self._interp.invoke()
            output = self._interp.get_tensor(out_d["index"])

        return output, (time.perf_counter() - t0) * 1000

    @staticmethod
    def _parse(output: np.ndarray, conf_th: float = 0.01) -> List[Dict]:
        pred = output[0]
        if pred.ndim == 2:
            pred = pred.T
        results = []
        n_cls = pred.shape[1] - 4
        for anchor in pred:
            scores = anchor[4:4 + n_cls]
            cls_id = int(np.argmax(scores))
            conf   = float(scores[cls_id])
            if conf >= conf_th:
                results.append({"class_id": cls_id, "confidence": conf})
        return results
