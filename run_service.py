#!/usr/bin/env python3
"""
run_service.py
==============
ARTEMIS v2 — Entry Point Layanan Edge Node (Topik 3)

Menggantikan pemanggilan manual phase_f_pi_evaluation.py dengan
satu command yang auto-configure berdasarkan config file per device.

CARA PAKAI:

  # Di Raspberry Pi 5 (real hardware)
  python3 run_service.py --config config/pi5.yaml

  # Dengan label lokasi (disimpan di hasil)
  python3 run_service.py --config config/pi5.yaml \
      --location jayagiri_hutan_rendah \
      --operator telkomsel

  # Jalankan method tertentu saja
  python3 run_service.py --config config/pi5.yaml \
      --methods adaptive,server_only

  # Mode folder langsung (tanpa sequences JSON)
  python3 run_service.py --config config/pi5.yaml \
      --mode folder

  # Di VPS (simulasi Pi3/Pi4B) — tc-netem sudah diinjeksikan duluan
  python3 run_service.py --config config/pi3_vps.yaml \
      --location lab_simulation \
      --experiment_id rq3_n3_nodes

CARA KERJA:
  1. Baca config YAML → setup semua parameter otomatis
  2. Auto-detect device type dari config (tidak perlu --pi_id manual)
  3. Gunakan sequences JSON atau folder gambar (dua mode)
  4. Panggil logika dari phase_f_pi_evaluation.py (tidak duplikasi)
  5. Simpan hasil dengan metadata lengkap: lokasi, kondisi jaringan, timestamp
  6. Kirim X-Node-ID header ke server untuk tracking RQ3

Output disimpan di: results/<experiment_id>_<node_id>_<timestamp>.json
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

# Import dari phase_f_pi_evaluation yang sudah ada
# Tidak duplikasi kode — reuse semua fungsi yang sudah berjalan
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from phase_f_pi_evaluation import (
    EdgeModel,
    HardwareTelemetry,
    compute_summary,
    offload_to_server_with_breakdown,
    print_latency_breakdown,
    run_adaptive_cooperative,
    run_device_only,
    run_server_only,
    run_static_cooperative,
)

# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config file, validasi field wajib."""
    p = Path(config_path)
    if not p.exists():
        print(f"ERROR: Config file tidak ditemukan: {config_path}")
        print("Buat config dengan: python3 run_service.py --init-config pi5")
        sys.exit(1)

    with open(p) as f:
        cfg = yaml.safe_load(f)

    required = ["node_id", "device_type", "model_edge", "model_type",
                "model_de", "server_url", "images_dir"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"ERROR: Field wajib hilang dari config: {missing}")
        sys.exit(1)

    return cfg


def print_config_template(device_type: str):
    """Print template config YAML ke stdout."""
    templates = {
        "pi5": {
            "node_id": "pi5_real",
            "device_type": "pi5",
            "model_edge": "models/best.onnx",
            "model_type": "onnx",
            "model_de": "models/lightgbm_de_v2.pkl",
            "server_url": "http://artemis.domain.com:8000",
            "images_dir": "data/full_test/images",
            "sequences": "sequences/sequence_list_v2.json",
            "thresholds": "thresholds_v2.json",
            "request_timeout": 15,
            "forced_offload_interval": 50,
            "output_dir": "results",
        },
        "pi4b": {
            "node_id": "pi4b_real",
            "device_type": "pi4b",
            "model_edge": "models/best_tflite/best_float32.tflite",
            "model_type": "tflite",
            "model_de": "models/lightgbm_de_v2.pkl",
            "server_url": "http://artemis.domain.com:8000",
            "images_dir": "data/full_test/images",
            "sequences": "sequences/sequence_list_v2.json",
            "thresholds": "thresholds_v2.json",
            "request_timeout": 20,
            "forced_offload_interval": 50,
            "output_dir": "results",
        },
        "pi3": {
            "node_id": "pi3_real",
            "device_type": "pi3",
            "model_edge": "models/best.onnx",
            "model_type": "onnx",
            "model_de": "models/lightgbm_de_v2.pkl",
            "server_url": "http://artemis.domain.com:8000",
            "images_dir": "data/full_test/images",
            "sequences": "sequences/sequence_list_v2.json",
            "thresholds": "thresholds_v2.json",
            "request_timeout": 30,
            "forced_offload_interval": 50,
            "output_dir": "results",
        },
        "pi3_vps": {
            "node_id": "pi3_vps_sim",
            "device_type": "pi3",
            "model_edge": "models/best.onnx",
            "model_type": "onnx",
            "model_de": "models/lightgbm_de_v2.pkl",
            "server_url": "http://HOMESERVER_IP:8000",
            "images_dir": "data/full_test/images",
            "sequences": "sequences/sequence_list_v2.json",
            "thresholds": "thresholds_v2.json",
            "request_timeout": 30,
            "forced_offload_interval": 50,
            "output_dir": "results",
            "_note": "tc-netem diinjeksikan via simulate_node.py sebelum run ini",
        },
    }

    if device_type not in templates:
        print(f"ERROR: device_type harus salah satu dari: {list(templates.keys())}")
        sys.exit(1)

    cfg = templates[device_type]
    print(f"# Config template untuk {device_type}")
    print(f"# Simpan sebagai: config/{device_type}.yaml")
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


# ── Server health check ───────────────────────────────────────────────────────

def check_server(server_url: str, timeout: float = 5.0) -> bool:
    """Verifikasi server bisa dijangkau sebelum mulai eksperimen."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=timeout)
        data = resp.json()
        if data.get("status") == "ok":
            print(f"  ✓ Server OK — model: {Path(data.get('model', '?')).name}, "
                  f"device: {data.get('device', '?')}, "
                  f"uptime: {data.get('uptime_s', 0):.0f}s")
            return True
        return False
    except Exception as e:
        print(f"  ✗ Server tidak bisa dijangkau: {e}")
        return False


# ── Sequence loader ───────────────────────────────────────────────────────────

def load_sequences(cfg: dict, images_dir: Path, mode: str = "sequences"):
    """
    Load frame sequences sesuai mode:
    - 'sequences': dari sequences JSON (Topik 2 compatible)
    - 'folder': semua gambar di images_dir, sorted by filename
    """
    if mode == "sequences":
        seq_path = cfg.get("sequences", "sequences/sequence_list_v2.json")
        if not Path(seq_path).exists():
            print(f"  WARNING: sequences file tidak ada ({seq_path}), fallback ke folder mode")
            mode = "folder"
        else:
            with open(seq_path) as f:
                seq_data = json.load(f)
            sequences = seq_data.get("sequences", [])
            # Filter ke frame yang tersedia
            valid = []
            for seq in sequences:
                valid_frames = [f for f in seq["frames"] if (images_dir / f).exists()]
                if valid_frames:
                    valid.append({**seq, "frames": valid_frames})
            print(f"  Sequences: {len(valid)}/{len(sequences)} valid "
                  f"(source: {seq_data.get('source', 'unknown')})")
            return valid

    if mode == "folder":
        exts   = {".jpg", ".jpeg", ".png"}
        frames = sorted([f.name for f in images_dir.iterdir()
                         if f.suffix.lower() in exts])
        if not frames:
            print(f"  ERROR: Tidak ada gambar di {images_dir}")
            sys.exit(1)
        # Bungkus sebagai satu sequence besar
        sequences = [{"seq_id": "folder_all", "seq_type": "continuous",
                      "frames": frames}]
        print(f"  Folder mode: {len(frames)} gambar sebagai 1 sequence")
        return sequences

    return []


# ── Main service runner ───────────────────────────────────────────────────────

def run_service(cfg: dict, args: argparse.Namespace):
    """
    Main service loop — jalankan semua method sesuai config.
    Ini adalah 'service wrapper' di atas phase_f_pi_evaluation.py.
    """
    node_id     = cfg["node_id"]
    device_type = cfg["device_type"]
    server_url  = cfg["server_url"]
    images_dir  = Path(cfg["images_dir"])
    out_dir     = Path(cfg.get("output_dir", "results"))
    timeout     = cfg.get("request_timeout", 15)
    forced_int  = cfg.get("forced_offload_interval", 50)

    # Override dari CLI jika ada
    location    = args.location or "unspecified"
    operator    = args.operator or "unspecified"
    exp_id      = args.experiment_id or f"{device_type}_{location}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ARTEMIS v2 — Edge Service Node")
    print(f"{'='*60}")
    print(f"  Node ID     : {node_id}")
    print(f"  Device Type : {device_type}")
    print(f"  Server URL  : {server_url}")
    print(f"  Location    : {location} ({operator})")
    print(f"  Experiment  : {exp_id}")
    print(f"  Timeout     : {timeout}s per request")
    print(f"  Mode        : {args.mode}")
    print(f"{'='*60}\n")

    # 1. Cek server
    print("Checking server...")
    if not check_server(server_url):
        print("\nERROR: Server tidak bisa dijangkau. Pastikan server_inference_v2.py jalan.")
        if not args.force:
            sys.exit(1)
        print("  (--force aktif, lanjut tanpa server — device_only saja)")

    # 2. Load thresholds
    edge_thresh = {
        "fire_local": 0.695, "fire_drop":   0.151,
        "smoke_local": 0.797, "smoke_drop": 0.128,
    }
    thresh_path = cfg.get("thresholds", "thresholds_v2.json")
    if Path(thresh_path).exists():
        with open(thresh_path) as f:
            t_data = json.load(f)
        edge_thresh = t_data.get("edge_model", edge_thresh)
    print(f"  Thresholds: {edge_thresh}")

    # 3. Load models
    print(f"\nLoading edge model: {cfg['model_edge']}")
    edge_model = EdgeModel(cfg["model_edge"], cfg["model_type"])

    print(f"Loading DE model: {cfg['model_de']}")
    with open(cfg["model_de"], "rb") as f:
        de_model = pickle.load(f)

    # 4. Load sequences
    print(f"\nLoading sequences...")
    sequences = load_sequences(cfg, images_dir, mode=args.mode)
    if not sequences:
        print("ERROR: Tidak ada sequence valid.")
        sys.exit(1)

    # 5. Tentukan methods yang akan dijalankan
    if args.methods == "all":
        methods = ["device_only", "server_only", "static", "adaptive"]
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Jika server tidak bisa dijangkau, skip server-dependent methods
    if not check_server(server_url, timeout=2.0):
        methods = [m for m in methods if m == "device_only"]
        print("  WARNING: Hanya device_only yang bisa dijalankan tanpa server.")

    print(f"\nMethods: {methods}")
    print(f"Sequences: {len(sequences)} | Total frames: "
          f"{sum(len(s['frames']) for s in sequences)}")

    # 6. Jalankan semua methods
    all_results  = {}
    telemetry    = HardwareTelemetry(interval_ms=300)
    run_metadata = {
        "node_id":        node_id,
        "device_type":    device_type,
        "experiment_id":  exp_id,
        "location":       location,
        "operator":       operator,
        "server_url":     server_url,
        "request_timeout": timeout,
        "mode":           args.mode,
        "timestamp_start": datetime.now().isoformat(),
        "config_file":    str(args.config),
    }

    # Inject X-Node-ID ke session requests
    # Patch requests.Session agar selalu kirim header node tracking
    _orig_session_init = requests.Session.__init__
    def _patched_session_init(self_s, *a, **kw):
        _orig_session_init(self_s, *a, **kw)
        self_s.headers.update({
            "X-Node-ID":       node_id,
            "X-Device-Type":   device_type,
            "X-Experiment-ID": exp_id,
        })
    requests.Session.__init__ = _patched_session_init

    if "device_only" in methods:
        print(f"\n[{node_id}] === Device-Only ===")
        t0  = time.time()
        res = run_device_only(sequences, images_dir, edge_model,
                              edge_thresh, telemetry)
        elapsed = time.time() - t0
        all_results["device_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "device_only"),
            "elapsed_s":     round(elapsed, 1),
        }
        print_latency_breakdown(node_id, "Device-Only",
                                all_results["device_only"]["summary"])
        gc.collect()

    if "server_only" in methods:
        print(f"\n[{node_id}] === Server-Only ===")
        t0  = time.time()
        res = run_server_only(sequences, images_dir, edge_model,
                              server_url, telemetry,
                              request_timeout=timeout)
        elapsed = time.time() - t0
        all_results["server_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "server_only"),
            "elapsed_s":     round(elapsed, 1),
        }
        print_latency_breakdown(node_id, "Server-Only",
                                all_results["server_only"]["summary"])
        gc.collect()

    if "static" in methods:
        print(f"\n[{node_id}] === Static Cooperative ===")
        t0  = time.time()
        res = run_static_cooperative(sequences, images_dir, edge_model,
                                     server_url, edge_thresh, telemetry,
                                     request_timeout=timeout)
        elapsed = time.time() - t0
        all_results["static_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "static_cooperative"),
            "elapsed_s":     round(elapsed, 1),
        }
        print_latency_breakdown(node_id, "Static Cooperative",
                                all_results["static_cooperative"]["summary"])
        gc.collect()

    if "adaptive" in methods:
        print(f"\n[{node_id}] === Adaptive Cooperative ===")
        t0  = time.time()
        res = run_adaptive_cooperative(sequences, images_dir, edge_model,
                                       de_model, server_url, edge_thresh,
                                       forced_int, telemetry,
                                       request_timeout=timeout)
        elapsed = time.time() - t0
        all_results["adaptive_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "adaptive_cooperative"),
            "elapsed_s":     round(elapsed, 1),
        }
        print_latency_breakdown(node_id, "Adaptive Cooperative",
                                all_results["adaptive_cooperative"]["summary"])
        gc.collect()

    # Restore session init
    requests.Session.__init__ = _orig_session_init

    # 7. Simpan hasil
    run_metadata["timestamp_end"] = datetime.now().isoformat()
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"{exp_id}_{node_id}_{ts}.json"
    out_path  = out_dir / out_fname

    output = {
        "metadata": run_metadata,
        "methods":  all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # 8. Print ringkasan akhir
    print(f"\n{'='*65}")
    print(f"RINGKASAN — {node_id.upper()} | {location} | {operator}")
    print(f"{'='*65}")
    print(f"  {'Method':<28} {'Avg(ms)':>8} {'P95(ms)':>8} {'Offload%':>9} {'Time(s)':>8}")
    print(f"  {'-'*63}")
    for mk, md in all_results.items():
        s     = md["summary"]
        bd    = s.get("latency_breakdown", {})
        avg   = bd.get("total_ms", {}).get("avg", s.get("steady_avg_total_ms", 0))
        p95   = bd.get("total_ms", {}).get("p95", 0)
        off   = s.get("offload_rate", 0) * 100
        elpsd = md.get("elapsed_s", 0)
        n_err = s.get("n_network_errors", 0)
        err_s = f" ({n_err} net_err)" if n_err > 0 else ""
        print(f"  {mk:<28} {avg:>7.1f}ms {p95:>7.1f}ms {off:>8.1f}% {elpsd:>7.0f}s{err_s}")
    print(f"{'='*65}")
    print(f"\nHasil disimpan → {out_path}")
    print(f"\nCek status server:")
    print(f"  curl {server_url}/status | python3 -m json.tool")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIS v2 — Edge Service Entry Point (Topik 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Jalankan di Pi5 dengan semua method
  python3 run_service.py --config config/pi5.yaml

  # Di lapangan dengan label lokasi
  python3 run_service.py --config config/pi5.yaml \\
      --location bukit_tunggul --operator telkomsel

  # Hanya adaptive + server_only
  python3 run_service.py --config config/pi5.yaml \\
      --methods adaptive,server_only

  # Buat config template
  python3 run_service.py --init-config pi5
  python3 run_service.py --init-config pi3_vps
        """
    )

    parser.add_argument("--config",        default="config/pi5.yaml",
                        help="Path ke config YAML (default: config/pi5.yaml)")
    parser.add_argument("--location",      default=None,
                        help="Label lokasi pengukuran (misal: jayagiri_hutan_rendah)")
    parser.add_argument("--operator",      default=None,
                        help="Nama operator seluler (misal: telkomsel, xl)")
    parser.add_argument("--experiment_id", default=None,
                        help="ID eksperimen untuk nama file output")
    parser.add_argument("--methods",       default="all",
                        help="Method yang dijalankan: all / device_only,server_only,static,adaptive")
    parser.add_argument("--mode",          default="sequences",
                        choices=["sequences", "folder"],
                        help="sequences: pakai JSON (Topik 2 compatible) | folder: semua gambar")
    parser.add_argument("--force",         action="store_true",
                        help="Lanjut meski server tidak bisa dijangkau (device_only saja)")
    parser.add_argument("--init-config",   metavar="DEVICE_TYPE",
                        help="Print template config YAML untuk device tertentu dan keluar")

    args = parser.parse_args()

    # Handle --init-config
    if args.init_config:
        print_config_template(args.init_config)
        return

    # Install yaml jika belum ada
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        os.system("pip install pyyaml --break-system-packages -q")
        import yaml

    cfg = load_config(args.config)
    run_service(cfg, args)


if __name__ == "__main__":
    main()
