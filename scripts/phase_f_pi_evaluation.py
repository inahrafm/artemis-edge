"""
phase_f_pi_evaluation.py
==========================
Phase F: Evaluasi 4 method di Raspberry Pi dengan BREAKDOWN LATENSI LENGKAP.

Komponen latensi yang diukur per frame:
  disk_read_ms      : waktu membaca file dari SD card/storage
  preprocess_ms     : resize + normalize gambar (CPU)
  edge_inference_ms : ONNX/TFLite model inference (model saja, tidak termasuk preprocess)
  de_ms             : LightGBM decision engine (feature build + predict)
  network_send_ms   : waktu kirim gambar ke server (Pi → Ubuntu)
  server_inference_ms: YOLOv26x inference di GPU server (dari response JSON)
  network_total_ms  : total round-trip network (send + server + receive)
  total_ms          : end-to-end dari baca file sampai keputusan alarm

Breakdown ini memungkinkan tabel tesis yang memisahkan:
  - Overhead komunikasi (network)
  - Overhead komputasi edge (preprocess + inference)
  - Overhead DE (LightGBM)
  - Bottleneck I/O (disk read)

Run on EACH Pi device:
    python3 scripts/phase_f_pi_evaluation.py \\
        --pi_id       pi5 \\
        --model_edge  models/best.onnx \\
        --model_type  onnx \\
        --model_de    models/lightgbm_de_v2.pkl \\
        --sequences   sequences/sequence_list_v2.json \\
        --images_dir  data/full_test/images \\
        --server_url  http://<UBUNTU_IP>:8000 \\
        --thresholds  thresholds_v2.json \\
        --output_dir  results \\
        --forced_offload_interval 50
"""

import argparse
import gc
import json
import pickle
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils import (LABEL_NAMES, WINDOW_SIZE, build_feature_vector,
                   extract_frame_features, get_logger)

log = get_logger("phase_f_eval")

CLASS_SMOKE = 0
CLASS_FIRE  = 1


# ── Hardware telemetry ────────────────────────────────────────────────────────
class HardwareTelemetry:
    def __init__(self, interval_ms: int = 500):
        self._interval = interval_ms / 1000
        self._data     = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running  = False
        self._thread   = None

    def start(self):
        self._data    = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            try:
                self._data["cpu"].append(psutil.cpu_percent(interval=None))
                self._data["ram_mb"].append(
                    psutil.virtual_memory().used / 1024 / 1024)
                self._data["temp_c"].append(self._read_temp())
            except Exception:
                pass
            time.sleep(self._interval)

    @staticmethod
    def _read_temp() -> float:
        try:
            temps = psutil.sensors_temperatures()
            for k in ["cpu_thermal", "cpu-thermal", "thermal_zone0",
                       "coretemp", "k10temp"]:
                if k in temps and temps[k]:
                    return temps[k][0].current
            p = Path("/sys/class/thermal/thermal_zone0/temp")
            if p.exists():
                return float(p.read_text().strip()) / 1000
        except Exception:
            pass
        return 0.0

    def summary(self) -> Dict:
        def s(lst):
            return ({"avg": round(float(np.mean(lst)), 2),
                     "max": round(float(np.max(lst)), 2)}
                    if lst else {"avg": 0.0, "max": 0.0})
        return {
            "cpu":    s(self._data["cpu"]),
            "ram_mb": s(self._data["ram_mb"]),
            "temp_c": s(self._data["temp_c"]),
        }


# ── Edge model ────────────────────────────────────────────────────────────────
class EdgeModel:
    def __init__(self, model_path: str, model_type: str):
        self.model_type = model_type
        self._session   = None
        self._interp    = None
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
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _read_image_bytes(self, image_path: str) -> Tuple[bytes, float]:
        """Baca file dari disk — diukur terpisah untuk breakdown I/O."""
        t0 = time.perf_counter()
        with open(image_path, "rb") as f:
            raw = f.read()
        disk_ms = (time.perf_counter() - t0) * 1000
        return raw, disk_ms

    def _preprocess_from_bytes(self, raw_bytes: bytes) -> Tuple[np.ndarray, float]:
        """Preprocess dari bytes (resize + normalize) — diukur terpisah."""
        from PIL import Image
        from io import BytesIO
        t0  = time.perf_counter()
        img = Image.open(BytesIO(raw_bytes)).convert("RGB").resize((640, 640))
        arr = np.array(img, dtype=np.float32) / 255.0
        if self.model_type == "tflite":
            out = arr[np.newaxis, ...]
        else:
            out = arr.transpose(2, 0, 1)[np.newaxis, ...]
        pre_ms = (time.perf_counter() - t0) * 1000
        return out, pre_ms

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess dari path (untuk server-only yang tidak perlu breakdown)."""
        raw, _ = self._read_image_bytes(image_path)
        arr, _ = self._preprocess_from_bytes(raw)
        return arr

    def infer_with_breakdown(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Jalankan inferensi dengan breakdown latensi lengkap.

        Returns:
            (detections, latency_breakdown_dict)

        latency_breakdown_dict berisi:
            disk_read_ms      : waktu baca file SD card
            preprocess_ms     : resize + normalize
            edge_inference_ms : model forward pass saja
            edge_total_ms     : preprocess + inference (tanpa disk)
        """
        raw,    disk_ms = self._read_image_bytes(image_path)
        inp,    pre_ms  = self._preprocess_from_bytes(raw)

        t0 = time.perf_counter()
        if self.model_type == "onnx":
            name   = self._session.get_inputs()[0].name
            output = self._session.run(None, {name: inp})[0]
        else:
            inp_d = self._interp.get_input_details()[0]
            out_d = self._interp.get_output_details()[0]
            self._interp.set_tensor(inp_d["index"], inp)
            self._interp.invoke()
            output = self._interp.get_tensor(out_d["index"])
        inf_ms = (time.perf_counter() - t0) * 1000

        dets = self._parse(output)
        return dets, {
            "disk_read_ms":      round(disk_ms, 3),
            "preprocess_ms":     round(pre_ms,  3),
            "edge_inference_ms": round(inf_ms,  3),
            "edge_total_ms":     round(pre_ms + inf_ms, 3),
        }

    def infer(self, image_path: str) -> Tuple[List[Dict], float]:
        """Legacy interface — returns (dets, inference_ms) tanpa breakdown."""
        dets, bd = self.infer_with_breakdown(image_path)
        return dets, bd["edge_inference_ms"]

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


# ── Server communication ──────────────────────────────────────────────────────
def offload_to_server_with_breakdown(image_path: str, server_url: str,
                                      session: requests.Session) -> Tuple[Dict, Dict]:
    """
    Kirim gambar ke server dan ukur breakdown latensi network.

    Field dari server response: 'server_total_ms' (waktu inference GPU di server).
    Field di breakdown: 'server_inference_ms' (alias untuk konsistensi penamaan).

    network_overhead_ms = network_total_ms - server_inference_ms
                        = waktu murni kirim+terima data di jaringan
    """
    t0 = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            resp = session.post(
                f"{server_url}/infer",
                files={"file": (Path(image_path).name, f, "image/jpeg")},
                timeout=30,
            )
        data = resp.json()
    except Exception as e:
        log.debug(f"Server request failed: {e}")
        data = {"decision": "NONE", "server_total_ms": 0.0,
                "confmax_fire": 0.0, "confmax_smoke": 0.0, "error": str(e)}

    network_total_ms = (time.perf_counter() - t0) * 1000

    # server_inference_ms = hanya GPU inference (baru ada setelah server_inference.py diupdate)
    # Fallback ke server_total_ms untuk kompatibilitas dengan server versi lama
    server_inference_ms = float(
        data.get("server_inference_ms") or data.get("server_total_ms") or 0.0
    )
    network_overhead_ms = max(0.0, network_total_ms - server_inference_ms)

    return data, {
        "network_total_ms":    round(network_total_ms,    3),
        "server_inference_ms": round(server_inference_ms, 3),
        "network_overhead_ms": round(network_overhead_ms, 3),
    }


def offload_to_server(image_path: str, server_url: str,
                       session: requests.Session) -> Dict:
    """Legacy interface tanpa breakdown."""
    data, _ = offload_to_server_with_breakdown(image_path, server_url, session)
    return data


# ── Alarm determination ───────────────────────────────────────────────────────
def determine_alarm(source: str, features: Dict = None,
                     server_resp: Dict = None,
                     edge_thresh: Dict = None) -> str:
    if source == "local" and features and edge_thresh:
        if features.get("confmax_fire",  0) >= edge_thresh["fire_local"]:
            return "FIRE_CRITICAL"
        if features.get("confmax_smoke", 0) >= edge_thresh["smoke_local"]:
            return "SMOKE_EARLY_WARNING"
        return "NONE"
    elif source == "server" and server_resp:
        d = server_resp.get("decision", "NONE")
        if d == "FIRE":   return "FIRE_CRITICAL"
        if d == "SMOKE":  return "SMOKE_EARLY_WARNING"
        return "NONE"
    return "NONE"


# ── Static cooperative decision ───────────────────────────────────────────────
def static_decision(feats: Dict, thresh: Dict) -> str:
    if (feats["confmax_fire"]  >= thresh["fire_local"] or
            feats["confmax_smoke"] >= thresh["smoke_local"]):
        return "LOCAL"
    if (feats["confmax_fire"]  <  thresh["fire_drop"] and
            feats["confmax_smoke"] <  thresh["smoke_drop"]):
        return "DROP"
    return "OFFLOAD"


# ── Method 1: Device-Only ─────────────────────────────────────────────────────
def run_device_only(sequences: List[Dict], images_dir: Path,
                     edge_model: EdgeModel, edge_thresh: Dict,
                     telemetry: HardwareTelemetry) -> List[Dict]:
    results = []
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0 = time.perf_counter()
            dets, bd = edge_model.infer_with_breakdown(str(img_path))
            feats    = extract_frame_features(dets)
            alarm    = determine_alarm("local", feats, edge_thresh=edge_thresh)
            total_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "filename":          fname,
                "seq_id":            seq["seq_id"],
                "seq_type":          seq["seq_type"],
                "method":            "device_only",
                # ── Breakdown latensi ─────────────────────────────────────
                "disk_read_ms":      bd["disk_read_ms"],
                "preprocess_ms":     bd["preprocess_ms"],
                "edge_inference_ms": bd["edge_inference_ms"],
                "edge_total_ms":     bd["edge_total_ms"],
                "network_total_ms":  0.0,   # tidak ada network
                "server_inference_ms": 0.0,
                "network_overhead_ms": 0.0,
                "de_ms":             0.0,   # tidak ada DE
                "total_ms":          round(total_ms, 3),
                # ── Lainnya ───────────────────────────────────────────────
                "alarm":             alarm,
                "confmax_fire":      round(feats["confmax_fire"],  4),
                "confmax_smoke":     round(feats["confmax_smoke"], 4),
                "is_warmup":         False,
                "de_decision":       "LOCAL",   # device-only selalu LOCAL/alarm check
            })
    telemetry.stop()
    return results


# ── Method 2: Server-Only ─────────────────────────────────────────────────────
def run_server_only(sequences: List[Dict], images_dir: Path,
                     edge_model: EdgeModel, server_url: str,
                     telemetry: HardwareTelemetry) -> List[Dict]:
    """
    Server-Only: baca file → kirim ke server → terima hasil.
    Tidak ada edge inference — hanya disk read + network + server GPU.
    """
    results = []
    session = requests.Session()
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0 = time.perf_counter()

            # Disk read diukur terpisah
            t_disk = time.perf_counter()
            with open(str(img_path), "rb") as _f:
                _raw = _f.read()
            disk_ms = (time.perf_counter() - t_disk) * 1000

            # Kirim ke server dengan breakdown network
            srv, net_bd = offload_to_server_with_breakdown(
                str(img_path), server_url, session
            )
            total_ms = (time.perf_counter() - t0) * 1000

            results.append({
                "filename":            fname,
                "seq_id":              seq["seq_id"],
                "seq_type":            seq["seq_type"],
                "method":              "server_only",
                # ── Breakdown latensi ─────────────────────────────────────
                "disk_read_ms":        round(disk_ms, 3),
                "preprocess_ms":       0.0,   # tidak ada preprocess lokal
                "edge_inference_ms":   0.0,   # tidak ada edge inference
                "edge_total_ms":       0.0,
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms":               0.0,
                "total_ms":            round(total_ms, 3),
                # ── Lainnya ───────────────────────────────────────────────
                "alarm":               determine_alarm("server", server_resp=srv),
                "confmax_fire":        srv.get("confmax_fire",  0.0),
                "confmax_smoke":       srv.get("confmax_smoke", 0.0),
                "is_warmup":           False,
                "de_decision":         "OFFLOAD",
            })
    telemetry.stop()
    return results


# ── Method 3: Static Cooperative ─────────────────────────────────────────────
def run_static_cooperative(sequences: List[Dict], images_dir: Path,
                             edge_model: EdgeModel, server_url: str,
                             edge_thresh: Dict,
                             telemetry: HardwareTelemetry) -> List[Dict]:
    results = []
    session = requests.Session()
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0 = time.perf_counter()

            dets, bd = edge_model.infer_with_breakdown(str(img_path))
            feats    = extract_frame_features(dets)
            decision = static_decision(feats, edge_thresh)

            alarm   = "NONE"
            net_bd  = {"network_total_ms": 0.0, "server_inference_ms": 0.0,
                        "network_overhead_ms": 0.0}
            if decision == "LOCAL":
                alarm = determine_alarm("local", feats, edge_thresh=edge_thresh)
            elif decision == "OFFLOAD":
                srv, net_bd = offload_to_server_with_breakdown(
                    str(img_path), server_url, session
                )
                alarm = determine_alarm("server", server_resp=srv)
            # DROP: tidak ada alarm, tidak ada network

            total_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "filename":            fname,
                "seq_id":              seq["seq_id"],
                "seq_type":            seq["seq_type"],
                "method":              "static_cooperative",
                "de_decision":         decision,
                # ── Breakdown latensi ─────────────────────────────────────
                "disk_read_ms":        bd["disk_read_ms"],
                "preprocess_ms":       bd["preprocess_ms"],
                "edge_inference_ms":   bd["edge_inference_ms"],
                "edge_total_ms":       bd["edge_total_ms"],
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms":               0.0,   # static tidak pakai DE
                "total_ms":            round(total_ms, 3),
                # ── Lainnya ───────────────────────────────────────────────
                "alarm":               alarm,
                "confmax_fire":        round(feats["confmax_fire"],  4),
                "confmax_smoke":       round(feats["confmax_smoke"], 4),
                "is_warmup":           False,
            })
    telemetry.stop()
    n_off = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    log.info(f"  [Static] offload_rate={n_off/len(results):.1%}")
    return results


# ── Method 4: Adaptive Cooperative ───────────────────────────────────────────
def run_adaptive_cooperative(sequences: List[Dict], images_dir: Path,
                               edge_model: EdgeModel, de_model,
                               server_url: str, edge_thresh: Dict,
                               forced_interval: int,
                               telemetry: HardwareTelemetry) -> List[Dict]:
    """
    Adaptive dengan per-sequence window reset (Blocker 2 fix) +
    breakdown latensi lengkap per frame.
    """
    results          = []
    session          = requests.Session()
    global_frame_idx = 0
    telemetry.start()

    for seq in sequences:
        seq_id   = seq["seq_id"]
        seq_type = seq["seq_type"]

        # Reset semua state temporal di setiap sequence boundary
        window          = deque(maxlen=WINDOW_SIZE)
        recent_offloads = deque(maxlen=WINDOW_SIZE)
        prev_feats      = None

        for local_idx, fname in enumerate(seq["frames"]):
            img_path = images_dir / fname
            if not img_path.exists():
                global_frame_idx += 1
                continue

            t0 = time.perf_counter()

            # Edge inference dengan breakdown
            dets, bd = edge_model.infer_with_breakdown(str(img_path))
            feats    = extract_frame_features(dets, prev_feats)
            prev_feats = feats

            is_warmup = local_idx < WINDOW_SIZE - 1
            if len(window) == 0:
                for _ in range(WINDOW_SIZE - 1):
                    window.append(feats)
            window.append(feats)

            ror = (sum(recent_offloads) / len(recent_offloads)
                   if recent_offloads else 0.0)

            # DE prediction — diukur terpisah
            t_de  = time.perf_counter()
            fv    = build_feature_vector(list(window), ror)
            de_label = LABEL_NAMES[int(de_model.predict(fv.reshape(1, -1))[0])]
            de_ms = (time.perf_counter() - t_de) * 1000

            # Forced OFFLOAD
            is_forced = (forced_interval > 0 and
                         global_frame_idx % forced_interval == 0)
            if is_forced:
                de_label = "OFFLOAD"

            recent_offloads.append(1 if de_label == "OFFLOAD" else 0)

            alarm  = "NONE"
            net_bd = {"network_total_ms": 0.0, "server_inference_ms": 0.0,
                       "network_overhead_ms": 0.0}
            if de_label == "LOCAL":
                alarm = determine_alarm("local", feats, edge_thresh=edge_thresh)
            elif de_label == "OFFLOAD":
                srv, net_bd = offload_to_server_with_breakdown(
                    str(img_path), server_url, session
                )
                alarm = determine_alarm("server", server_resp=srv)
            # DROP: tidak ada alarm, tidak ada network

            total_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "filename":            fname,
                "seq_id":              seq_id,
                "seq_type":            seq_type,
                "method":              "adaptive_cooperative",
                "local_frame_idx":     local_idx,
                "global_frame_idx":    global_frame_idx,
                "de_decision":         de_label,
                "is_forced_offload":   is_forced,
                "is_warmup":           is_warmup,
                # ── Breakdown latensi ─────────────────────────────────────
                "disk_read_ms":        bd["disk_read_ms"],
                "preprocess_ms":       bd["preprocess_ms"],
                "edge_inference_ms":   bd["edge_inference_ms"],
                "edge_total_ms":       bd["edge_total_ms"],
                "de_ms":               round(de_ms, 4),
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "total_ms":            round(total_ms, 3),
                # ── Lainnya ───────────────────────────────────────────────
                "alarm":               alarm,
                "confmax_fire":        round(feats["confmax_fire"],  4),
                "confmax_smoke":       round(feats["confmax_smoke"], 4),
                "recent_offload_rate": round(ror, 4),
            })
            global_frame_idx += 1

    telemetry.stop()
    n_off   = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    n_total = len(results) or 1
    log.info(f"  [Adaptive] offload_rate={n_off/n_total:.1%}")
    return results


# ── Summary helpers ───────────────────────────────────────────────────────────
def compute_summary(results: List[Dict], method_key: str) -> Dict:
    """
    Hitung statistik ringkasan per method.
    Memisahkan warm-up vs steady-state, dan breakdown latensi per komponen.
    """
    n = len(results)
    if n == 0:
        return {"n_frames": 0}

    warmup_res = [r for r in results if r.get("is_warmup")]
    steady_res = [r for r in results if not r.get("is_warmup")]

    def _stats(lst, key):
        """Hitung avg, std, p50, p95 untuk satu kolom."""
        vals = [r[key] for r in lst if key in r and r[key] is not None]
        if not vals:
            return {"avg": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
        arr = np.array(vals)
        return {
            "avg": round(float(arr.mean()), 3),
            "std": round(float(arr.std()),  3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
        }

    def _mean(lst, key):
        vals = [r[key] for r in lst if key in r]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    dist  = {k: sum(1 for r in results if r.get("de_decision") == k)
             for k in ["LOCAL", "OFFLOAD", "DROP"]}
    n_off = dist.get("OFFLOAD", 0)

    # Breakdown latensi — steady state saja (lebih representatif)
    latency_keys = [
        "disk_read_ms", "preprocess_ms", "edge_inference_ms",
        "edge_total_ms", "de_ms",
        "network_total_ms", "server_inference_ms", "network_overhead_ms",
        "total_ms",
    ]
    latency_breakdown = {}
    for key in latency_keys:
        latency_breakdown[key] = _stats(steady_res, key)

    # Untuk frame yang OFFLOAD saja (network latency lebih bermakna)
    offload_frames = [r for r in steady_res if r.get("de_decision") == "OFFLOAD"
                      or r.get("method") in ("server_only",)]
    if offload_frames:
        latency_breakdown["network_total_ms_offload_only"] = _stats(
            offload_frames, "network_total_ms"
        )
        latency_breakdown["server_inference_ms_offload_only"] = _stats(
            offload_frames, "server_inference_ms"
        )

    s = {
        "n_frames":          n,
        "n_warmup":          len(warmup_res),
        "n_steady":          len(steady_res),
        "avg_total_ms":      _mean(results,     "total_ms"),
        "steady_avg_total_ms": _mean(steady_res, "total_ms"),
        "alarm_count":       sum(1 for r in results if r.get("alarm") != "NONE"),
        "offload_rate":      round(n_off / n, 4),
        "decision_dist":     dist,
        "avg_de_ms":         _mean(results, "de_ms"),
        "n_forced_offload":  sum(1 for r in results if r.get("is_forced_offload")),
        "latency_breakdown": latency_breakdown,
    }

    for seq_type in ["gradual_escalation", "confident_smoke",
                      "fire_smoke_simultaneous"]:
        sub = [r for r in results if r.get("seq_type") == seq_type]
        if sub:
            n_off_sub = sum(1 for r in sub if r.get("de_decision") == "OFFLOAD")
            s[f"offload_rate_{seq_type}"] = round(n_off_sub / len(sub), 4)

    return s


def print_latency_breakdown(pi_id: str, method: str, summary: Dict):
    """Cetak tabel breakdown latensi ke terminal."""
    bd = summary.get("latency_breakdown", {})
    if not bd:
        return

    print(f"\n  [{pi_id.upper()} | {method}] Latency Breakdown (steady-state, ms)")
    print(f"  {'Component':<28} {'avg':>7} {'std':>7} {'p50':>7} {'p95':>7}")
    print(f"  {'-'*56}")

    rows = [
        ("disk_read_ms",         "Disk Read (SD card)"),
        ("preprocess_ms",        "Preprocess (resize/norm)"),
        ("edge_inference_ms",    "Edge Inference (ONNX/TFLite)"),
        ("de_ms",                "Decision Engine (LightGBM)"),
        ("network_total_ms",     "Network Round-Trip (avg all)"),
        ("network_total_ms_offload_only", "Network RT (offload frames only)"),
        ("server_inference_ms",  "  └ Server GPU Inference"),
        ("network_overhead_ms",  "  └ Network Overhead (send+recv)"),
        ("total_ms",             "TOTAL end-to-end"),
    ]
    for key, label in rows:
        if key in bd:
            v = bd[key]
            print(f"  {label:<28} {v['avg']:>7.2f} {v['std']:>7.2f} "
                  f"{v['p50']:>7.2f} {v['p95']:>7.2f}")

    n_off = summary.get("decision_dist", {}).get("OFFLOAD", 0)
    n_tot = summary.get("n_steady", 1) or 1
    print(f"\n  Offload rate: {n_off/n_tot:.1%} ({n_off}/{n_tot} steady frames)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pi evaluation (Phase F)")
    parser.add_argument("--pi_id",                  required=True)
    parser.add_argument("--model_edge",              required=True)
    parser.add_argument("--model_type",              required=True,
                        choices=["onnx", "tflite"])
    parser.add_argument("--model_de",                required=True)
    parser.add_argument("--sequences",               required=True)
    parser.add_argument("--images_dir",              required=True)
    parser.add_argument("--server_url",              required=True)
    parser.add_argument("--thresholds",              default="thresholds_v2.json")
    parser.add_argument("--output_dir",              default="results")
    parser.add_argument("--forced_offload_interval", type=int, default=50)
    parser.add_argument("--methods",                 default="all")
    args = parser.parse_args()

    out        = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    if not images_dir.exists():
        log.error(f"images_dir tidak ditemukan: {images_dir}")
        sys.exit(1)

    # Load thresholds
    edge_thresh = {
        "fire_local": 0.695, "fire_drop":   0.151,
        "smoke_local": 0.797, "smoke_drop":  0.128,
    }
    if Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            t_data = json.load(f)
        edge_thresh = t_data.get("edge_model", edge_thresh)
    log.info(f"Edge thresholds: {edge_thresh}")

    # Load models
    log.info(f"[{args.pi_id}] Loading edge model: {args.model_edge}")
    edge_model = EdgeModel(args.model_edge, args.model_type)

    log.info(f"[{args.pi_id}] Loading DE model: {args.model_de}")
    with open(args.model_de, "rb") as f:
        de_model = pickle.load(f)

    # Load sequences
    with open(args.sequences) as f:
        seq_data = json.load(f)
    sequences = seq_data.get("sequences", [])
    log.info(f"[{args.pi_id}] {len(sequences)} sequences "
             f"(source: {seq_data.get('source', 'unknown')})")

    # Filter ke sequences yang frame-nya tersedia
    valid_sequences = []
    for seq in sequences:
        valid_frames = [f for f in seq["frames"]
                        if (images_dir / f).exists()]
        if valid_frames:
            valid_sequences.append({**seq, "frames": valid_frames})
    log.info(f"  {len(valid_sequences)}/{len(sequences)} sequences valid")

    if not valid_sequences:
        log.error(f"Tidak ada sequence valid. Cek --images_dir: {images_dir}")
        sys.exit(1)

    methods    = (["device_only", "server_only", "static", "adaptive"]
                  if args.methods == "all" else args.methods.split(","))
    all_results = {}
    telemetry   = HardwareTelemetry(interval_ms=300)

    if "device_only" in methods:
        log.info(f"\n[{args.pi_id}] === Device-Only ===")
        res = run_device_only(
            valid_sequences, images_dir, edge_model, edge_thresh, telemetry
        )
        all_results["device_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "device_only"),
        }
        print_latency_breakdown(args.pi_id, "Device-Only",
                                all_results["device_only"]["summary"])
        gc.collect()

    if "server_only" in methods:
        log.info(f"\n[{args.pi_id}] === Server-Only ===")
        res = run_server_only(
            valid_sequences, images_dir, edge_model, args.server_url, telemetry
        )
        all_results["server_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "server_only"),
        }
        print_latency_breakdown(args.pi_id, "Server-Only",
                                all_results["server_only"]["summary"])
        gc.collect()

    if "static" in methods:
        log.info(f"\n[{args.pi_id}] === Static Cooperative ===")
        res = run_static_cooperative(
            valid_sequences, images_dir, edge_model, args.server_url,
            edge_thresh, telemetry
        )
        all_results["static_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "static_cooperative"),
        }
        print_latency_breakdown(args.pi_id, "Static Cooperative",
                                all_results["static_cooperative"]["summary"])
        gc.collect()

    if "adaptive" in methods:
        log.info(f"\n[{args.pi_id}] === Adaptive Cooperative ===")
        res = run_adaptive_cooperative(
            valid_sequences, images_dir, edge_model, de_model,
            args.server_url, edge_thresh,
            args.forced_offload_interval, telemetry
        )
        all_results["adaptive_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "adaptive_cooperative"),
        }
        print_latency_breakdown(args.pi_id, "Adaptive Cooperative",
                                all_results["adaptive_cooperative"]["summary"])
        gc.collect()

    # Simpan hasil
    out_file = out / f"results_{args.pi_id}.json"
    with open(out_file, "w") as f:
        json.dump({
            "pi_id":       args.pi_id,
            "model_edge":  args.model_edge,
            "model_type":  args.model_type,
            "n_sequences": len(valid_sequences),
            "thresholds":  edge_thresh,
            "methods":     all_results,
        }, f, indent=2)
    log.info(f"\n[{args.pi_id}] Hasil disimpan: {out_file}")

    # Print ringkasan akhir
    print(f"\n{'='*65}")
    print(f"RINGKASAN LATENSI — {args.pi_id.upper()}")
    print(f"{'='*65}")
    print(f"  {'Method':<28} {'Avg Total':>10} {'P95 Total':>10} {'Offload%':>9}")
    print(f"  {'-'*59}")
    for mk, md in all_results.items():
        s  = md["summary"]
        bd = s.get("latency_breakdown", {})
        total_avg = bd.get("total_ms", {}).get("avg", s.get("steady_avg_total_ms", 0))
        total_p95 = bd.get("total_ms", {}).get("p95", 0)
        off_pct   = s.get("offload_rate", 0) * 100
        print(f"  {mk:<28} {total_avg:>9.1f}ms {total_p95:>9.1f}ms {off_pct:>8.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()