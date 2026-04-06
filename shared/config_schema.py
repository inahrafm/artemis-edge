"""
shared/config_schema.py
========================
Dataclass schema untuk konfigurasi edge node ARTEMIS v2.
Dipakai oleh edge/config.py untuk validasi dan type hints.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EdgeNodeConfig:
    """Konfigurasi lengkap satu edge node."""

    # ── Identity ──────────────────────────────────────────────────────────────
    node_id:     str = "unknown_node"
    device_type: str = "pi5"        # pi3 | pi4b | pi5

    # ── Model paths ───────────────────────────────────────────────────────────
    model_edge:  str = "models/best.onnx"
    model_type:  str = "onnx"       # onnx | tflite
    model_de:    str = "models/lightgbm_de_v2.pkl"

    # ── Server ────────────────────────────────────────────────────────────────
    server_url:      str   = "http://localhost:8000"
    request_timeout: float = 15.0   # detik — sesuaikan per kondisi jaringan

    # ── Data ─────────────────────────────────────────────────────────────────
    images_dir: str = "data/full_test/images"
    sequences:  str = "sequences/sequence_list_v2.json"
    thresholds: str = "thresholds_v2.json"

    # ── Experiment params ─────────────────────────────────────────────────────
    forced_offload_interval: int  = 50
    output_dir:              str  = "results"

    # ── Thresholds (populated from thresholds file) ───────────────────────────
    edge_thresh: Dict[str, float] = field(default_factory=lambda: {
        "fire_local":  0.695, "fire_drop":   0.151,
        "smoke_local": 0.797, "smoke_drop":  0.128,
    })


@dataclass
class ServerConfig:
    """Konfigurasi inference server."""
    model_path:      str   = "best.pt"
    device:          str   = "cuda"
    thresholds_path: str   = "thresholds_v2.json"
    host:            str   = "0.0.0.0"
    port:            int   = 8000
    log_level:       str   = "INFO"
    n_threads:       int   = 8      # Gunicorn gthread workers
