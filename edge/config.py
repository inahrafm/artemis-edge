"""
edge/config.py
==============
Auto-detect tipe perangkat Raspberry Pi dan load konfigurasi yang tepat.

Zero-touch design: Pi baru cukup jalankan run_edge.py tanpa argumen tambahan.
Device type (Pi3/Pi4B/Pi5) dideteksi otomatis dari hardware, lalu config
YAML yang sesuai dimuat, termasuk format model (ONNX/TFLite) yang optimal.

Device detection order:
  1. --device CLI flag (override manual)
  2. ARTEMIS_DEVICE env variable
  3. /proc/device-tree/model (Raspberry Pi hardware info)
  4. Fallback: pi5 (paling konservatif dari sisi format)
"""

import json
import logging
import os
import platform
from pathlib import Path
from typing import Optional

import yaml

from shared.config_schema import EdgeNodeConfig
from shared.constants import DEFAULT_THRESHOLDS

log = logging.getLogger("artemis.edge.config")


# ── Device detection ──────────────────────────────────────────────────────────

def detect_device_type() -> str:
    """
    Auto-detect tipe Raspberry Pi dari hardware info.

    Returns: 'pi3' | 'pi4b' | 'pi5' | 'unknown'
    """
    # Coba baca model string dari device tree
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        try:
            model_str = model_file.read_bytes().decode("utf-8", errors="ignore").lower()
            if "raspberry pi 5" in model_str:
                return "pi5"
            if "raspberry pi 4" in model_str:
                return "pi4b"
            if "raspberry pi 3" in model_str:
                return "pi3"
        except Exception:
            pass

    # Fallback: coba dari /proc/cpuinfo
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text().lower()
            if "bcm2712" in text:   # Pi5 chip
                return "pi5"
            if "bcm2711" in text:   # Pi4B chip
                return "pi4b"
            if "bcm2837" in text:   # Pi3B/3B+ chip
                return "pi3"
        except Exception:
            pass

    log.warning("Tidak bisa auto-detect device type, fallback ke pi5")
    return "pi5"


# ── Format optimal per device ─────────────────────────────────────────────────

# Format model yang terbukti optimal dari benchmark Topik 1:
#   Pi3:  ONNX FP32   → 744ms  (TFLite tidak lebih cepat di Pi3)
#   Pi4B: TFLite FP32 → 342ms  (vs ONNX 374ms, selisih 32ms / 8.5%)
#   Pi5:  TFLite FP32 → 113ms  (vs ONNX 131ms, selisih 18ms / 13.4%)
OPTIMAL_FORMAT = {
    "pi3":  {"model_type": "onnx",    "model_file": "best.onnx"},
    "pi4b": {"model_type": "tflite",  "model_file": "best_float32.tflite"},
    "pi5":  {"model_type": "tflite",  "model_file": "best_float32.tflite"},
}

RECOMMENDED_TIMEOUT = {
    "pi3":  30.0,   # Pi3 inference lambat → toleransi lebih tinggi
    "pi4b": 20.0,
    "pi5":  15.0,
}


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None,
                device_override: Optional[str] = None) -> EdgeNodeConfig:
    """
    Load EdgeNodeConfig dari YAML file atau buat default berdasarkan device type.

    Priority:
      1. config_path YAML jika diberikan
      2. config/<device_type>.yaml jika ada
      3. Default programmatic berdasarkan device type

    Args:
        config_path:     path ke YAML config (opsional)
        device_override: override device type ('pi3'|'pi4b'|'pi5')

    Returns:
        EdgeNodeConfig yang sudah terisi lengkap
    """
    # Tentukan device type
    device_type = (
        device_override
        or os.environ.get("ARTEMIS_DEVICE_TYPE")
        or detect_device_type()
    )

    if device_type not in OPTIMAL_FORMAT:
        log.warning(f"Device type tidak dikenal: '{device_type}', fallback ke pi5")
        device_type = "pi5"

    log.info(f"Device type: {device_type}")

    # Coba load dari YAML
    yaml_data = {}
    yaml_source = None

    if config_path and Path(config_path).exists():
        yaml_source = config_path
    elif Path(f"config/{device_type}.yaml").exists():
        yaml_source = f"config/{device_type}.yaml"

    if yaml_source:
        with open(yaml_source) as f:
            yaml_data = yaml.safe_load(f) or {}
        log.info(f"Config dimuat dari: {yaml_source}")
    else:
        log.info(f"Config YAML tidak ditemukan, pakai defaults untuk {device_type}")

    # Tentukan model path — prioritaskan YAML, fallback ke optimal per device
    opt = OPTIMAL_FORMAT[device_type]
    model_dir  = Path(yaml_data.get("models_dir", "models"))
    model_file = yaml_data.get("model_edge") or str(model_dir / opt["model_file"])
    model_type = yaml_data.get("model_type") or opt["model_type"]

    cfg = EdgeNodeConfig(
        node_id     = yaml_data.get("node_id",     f"{device_type}_node"),
        device_type = device_type,
        model_edge  = model_file,
        model_type  = model_type,
        model_de    = yaml_data.get("model_de",    "models/lightgbm_de_v2.pkl"),
        server_url  = yaml_data.get("server_url",  "http://localhost:8000"),
        request_timeout = float(yaml_data.get(
            "request_timeout", RECOMMENDED_TIMEOUT[device_type])),
        images_dir  = yaml_data.get("images_dir",  "data/full_test/images"),
        sequences   = yaml_data.get("sequences",   "sequences/sequence_list_v2.json"),
        thresholds  = yaml_data.get("thresholds",  "thresholds_v2.json"),
        forced_offload_interval = int(yaml_data.get("forced_offload_interval", 50)),
        output_dir  = yaml_data.get("output_dir",  "results"),
    )

    # Load thresholds dari file
    cfg.edge_thresh = _load_edge_thresholds(cfg.thresholds)

    _validate_paths(cfg)
    _print_config(cfg)
    return cfg


def _load_edge_thresholds(thresholds_path: str) -> dict:
    """Load edge model thresholds dari file, fallback ke defaults."""
    p = Path(thresholds_path)
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            thresh = data.get("edge_model", {})
            if thresh:
                log.info(f"Thresholds dimuat: {thresh}")
                return thresh
        except Exception as e:
            log.warning(f"Gagal baca thresholds ({e}), pakai defaults")
    else:
        log.warning(f"Thresholds file tidak ditemukan: {thresholds_path}")

    return dict(DEFAULT_THRESHOLDS["edge_model"])


def _validate_paths(cfg: EdgeNodeConfig):
    """Validasi file-file penting ada sebelum mulai."""
    checks = {
        "model_edge": cfg.model_edge,
        "model_de":   cfg.model_de,
    }
    missing = [name for name, path in checks.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"File/folder tidak ditemukan: {missing}\n"
            f"Jalankan dari root direktori artemis-v2/ atau periksa config."
        )


def _print_config(cfg: EdgeNodeConfig):
    """Print ringkasan config ke log."""
    log.info("=" * 55)
    log.info("ARTEMIS v2 — Edge Node Config")
    log.info(f"  Node ID     : {cfg.node_id}")
    log.info(f"  Device Type : {cfg.device_type}")
    log.info(f"  Model Edge  : {cfg.model_edge} ({cfg.model_type})")
    log.info(f"  Model DE    : {cfg.model_de}")
    log.info(f"  Server URL  : {cfg.server_url}")
    log.info(f"  Timeout     : {cfg.request_timeout}s")
    log.info(f"  Thresholds  : {cfg.edge_thresh}")
    log.info("=" * 55)
