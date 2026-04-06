#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
scripts/run_edge.py
====================
Entry point edge node ARTEMIS v2 Topik 3.

Zero-touch: cukup jalankan tanpa argumen di Pi yang benar,
device type dan format model dipilih otomatis.

CARA PAKAI:

  # Minimal — auto-detect semua
  python3 scripts/run_edge.py

  # Dengan label lokasi (disimpan di hasil)
  python3 scripts/run_edge.py \
      --location jayagiri_hutan_rendah \
      --operator telkomsel

  # Override server URL
  python3 scripts/run_edge.py \
      --server http://artemis.domain.com:8000

  # Jalankan method tertentu saja
  python3 scripts/run_edge.py --methods adaptive,server_only

  # Override device type (untuk VPS simulasi)
  python3 scripts/run_edge.py --device pi3 --node_id pi3_vps_sim

  # Buat config template untuk device ini
  python3 scripts/run_edge.py --init-config
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


from edge.notifier import AlarmNotifier

# Pastikan root project ada di path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.config import load_config
from edge.decision_engine import DecisionEngine
from edge.inference import EdgeInference
from edge.node import (
    HardwareTelemetry,
    run_adaptive_cooperative,
    run_device_only,
    run_server_only,
    run_static_cooperative,
)
from edge.offloader import Offloader
from shared.config_schema import EdgeNodeConfig


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def compute_summary(results, method_key: str) -> dict:
    """Hitung statistik ringkasan — identik dengan phase_f untuk kompatibilitas."""
    import numpy as np
    n = len(results)
    if n == 0:
        return {"n_frames": 0}

    warmup_res = [r for r in results if r.get("is_warmup")]
    steady_res = [r for r in results if not r.get("is_warmup")]

    def _stats(lst, key):
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

    latency_keys = [
        "disk_read_ms", "preprocess_ms", "edge_inference_ms",
        "edge_total_ms", "de_ms", "network_total_ms",
        "server_inference_ms", "network_overhead_ms", "total_ms",
    ]
    latency_breakdown = {key: _stats(steady_res, key) for key in latency_keys}

    offload_frames = [r for r in steady_res
                      if r.get("de_decision") == "OFFLOAD"
                      or r.get("method") == "server_only"]
    if offload_frames:
        latency_breakdown["network_total_ms_offload_only"]    = _stats(offload_frames, "network_total_ms")
        latency_breakdown["server_inference_ms_offload_only"] = _stats(offload_frames, "server_inference_ms")

    n_net_err = sum(1 for r in results if r.get("network_error", False))
    return {
        "n_frames":            n,
        "n_warmup":            len(warmup_res),
        "n_steady":            len(steady_res),
        "avg_total_ms":        _mean(results, "total_ms"),
        "steady_avg_total_ms": _mean(steady_res, "total_ms"),
        "alarm_count":         sum(1 for r in results if r.get("alarm") != "NONE"),
        "offload_rate":        round(n_off / n, 4),
        "decision_dist":       dist,
        "avg_de_ms":           _mean(results, "de_ms"),
        "n_forced_offload":    sum(1 for r in results if r.get("is_forced_offload")),
        "latency_breakdown":   latency_breakdown,
        "n_network_errors":    n_net_err,
        "network_error_rate":  round(n_net_err / n, 4) if n > 0 else 0.0,
    }


def load_sequences(cfg: EdgeNodeConfig, mode: str = "sequences"):
    """Load frame sequences dari JSON atau folder."""
    images_dir = Path(cfg.images_dir)

    if mode == "sequences" and Path(cfg.sequences).exists():
        with open(cfg.sequences) as f:
            seq_data = json.load(f)
        sequences = seq_data.get("sequences", [])
        valid = []
        for seq in sequences:
            valid_frames = [f for f in seq["frames"] if (images_dir / f).exists()]
            if valid_frames:
                valid.append({**seq, "frames": valid_frames})
        print(f"  Sequences: {len(valid)}/{len(sequences)} valid")
        return valid

    # Folder mode
    exts   = {".jpg", ".jpeg", ".png"}
    frames = sorted([f.name for f in images_dir.iterdir()
                     if f.suffix.lower() in exts])
    print(f"  Folder mode: {len(frames)} gambar")
    return [{"seq_id": "folder_all", "seq_type": "continuous", "frames": frames}]


def print_summary(node_id: str, location: str, operator: str,
                  all_results: dict):
    print(f"\n{'='*65}")
    print(f"RINGKASAN — {node_id.upper()} | {location} | {operator}")
    print(f"{'='*65}")
    print(f"  {'Method':<28} {'Avg(ms)':>8} {'P95(ms)':>8} "
          f"{'Offload%':>9} {'NetErr':>7}")
    print(f"  {'-'*62}")
    for mk, md in all_results.items():
        s     = md["summary"]
        bd    = s.get("latency_breakdown", {})
        avg   = bd.get("total_ms", {}).get("avg", s.get("steady_avg_total_ms", 0))
        p95   = bd.get("total_ms", {}).get("p95", 0)
        off   = s.get("offload_rate", 0) * 100
        n_err = s.get("n_network_errors", 0)
        print(f"  {mk:<28} {avg:>7.1f}ms {p95:>7.1f}ms {off:>8.1f}% {n_err:>7}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIS v2 — Edge Node Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",        default=None,
                        help="Path ke config YAML (auto-detect jika tidak diisi)")
    parser.add_argument("--device",        default=None,
                        choices=["pi3", "pi4b", "pi5"],
                        help="Override device type")
    parser.add_argument("--node_id",       default=None,
                        help="Override node ID")
    parser.add_argument("--server",        default=None,
                        help="Override server URL")
    parser.add_argument("--location",      default="unspecified")
    parser.add_argument("--operator",      default="unspecified")
    parser.add_argument("--experiment_id", default=None)
    parser.add_argument("--methods",       default="all",
                        help="all | device_only,server_only,static,adaptive")
    parser.add_argument("--mode",          default="sequences",
                        choices=["sequences", "folder"])
    parser.add_argument("--log_level",     default="INFO")
    parser.add_argument("--init-config",   action="store_true",
                        help="Print config template dan keluar")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.init_config:
        device = args.device or "pi5"
        print(f"# Config template untuk {device}")
        print(f"# Simpan sebagai: config/{device}.yaml\n")
        template = {
            "node_id":     f"{device}_node",
            "device_type": device,
            "server_url":  "http://artemis.domain.com:8000",
            "images_dir":  "data/full_test/images",
            "sequences":   "sequences/sequence_list_v2.json",
            "thresholds":  "thresholds_v2.json",
            "output_dir":  "results",
            "request_timeout": {"pi3": 30, "pi4b": 20, "pi5": 15}[device],
            "forced_offload_interval": 50,
        }
        import yaml
        print(yaml.dump(template, default_flow_style=False, sort_keys=False))
        return

    # Load config
    cfg = load_config(config_path=args.config, device_override=args.device)

    # Apply CLI overrides
    if args.node_id:
        cfg.node_id = args.node_id
    if args.server:
        cfg.server_url = args.server

    exp_id = args.experiment_id or f"{cfg.device_type}_{args.location}"

    # Load components
    print(f"\nLoading edge model: {cfg.model_edge} ({cfg.model_type})")
    inference = EdgeInference(cfg.model_edge, cfg.model_type)

    print(f"Loading DE model: {cfg.model_de}")
    de = DecisionEngine(cfg.model_de)

    notifier = AlarmNotifier(
        alarm_url   = "https://weartemis.me/alarm",
        node_id     = cfg.node_id,
        device_type = cfg.device_type,
    )

    offloader = Offloader(
        server_url    = cfg.server_url,
        node_id       = cfg.node_id,
        device_type   = cfg.device_type,
        timeout       = cfg.request_timeout,
        experiment_id = exp_id,
    )

    # Health check
    print("\nChecking server...")
    server_ok = offloader.health_check()
    if not server_ok:
        print("WARNING: Server tidak bisa dijangkau.")

    # Load sequences
    images_dir = Path(cfg.images_dir)
    sequences  = load_sequences(cfg, mode=args.mode)
    if not sequences:
        print("ERROR: Tidak ada sequence valid.")
        sys.exit(1)

    # Tentukan methods
    if args.methods == "all":
        methods = ["device_only", "server_only", "static", "adaptive"]
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    if not server_ok:
        methods = [m for m in methods if m == "device_only"]
        print("  Hanya device_only (server tidak tersedia)")

    print(f"\nMethods: {methods}")
    print(f"Frames: {sum(len(s['frames']) for s in sequences)}")

    # Run
    all_results = {}
    telemetry   = HardwareTelemetry(interval_ms=300)
    out_dir     = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "device_only" in methods:
        print(f"\n[{cfg.node_id}] === Device-Only ===")
        res = run_device_only(sequences, images_dir, inference, cfg, telemetry, notifier=notifier, exp_id=exp_id, location=args.location, gps=getattr(cfg, "gps", ""))
        all_results["device_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "device_only"),
        }
        gc.collect()

    if "server_only" in methods:
        print(f"\n[{cfg.node_id}] === Server-Only ===")
        res = run_server_only(sequences, images_dir, inference, offloader, telemetry, notifier=notifier, exp_id=exp_id, location=args.location, gps=getattr(cfg, "gps", ""))
        all_results["server_only"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "server_only"),
        }
        gc.collect()

    if "static" in methods:
        print(f"\n[{cfg.node_id}] === Static Cooperative ===")
        res = run_static_cooperative(
            sequences, images_dir, inference, offloader, cfg, telemetry, notifier=notifier, exp_id=exp_id, location=args.location, gps=getattr(cfg, "gps", ""))
        all_results["static_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "static_cooperative"),
        }
        gc.collect()

    if "adaptive" in methods:
        print(f"\n[{cfg.node_id}] === Adaptive Cooperative ===")
        res = run_adaptive_cooperative(
            sequences, images_dir, inference, de, offloader, cfg, telemetry, notifier=notifier, exp_id=exp_id, location=args.location, gps=getattr(cfg, "gps", ""))
        all_results["adaptive_cooperative"] = {
            "frame_results": res,
            "hardware":      telemetry.summary(),
            "summary":       compute_summary(res, "adaptive_cooperative"),
        }
        gc.collect()

    offloader.close()

    # Simpan hasil
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{exp_id}_{cfg.node_id}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "node_id":       cfg.node_id,
                "device_type":   cfg.device_type,
                "experiment_id": exp_id,
                "location":      args.location,
                "operator":      args.operator,
                "server_url":    cfg.server_url,
                "timestamp":     datetime.now().isoformat(),
                "offloader_stats": offloader.stats,
            },
            "methods": all_results,
        }, f, indent=2)

    print_summary(cfg.node_id, args.location, args.operator, all_results)
    print(f"\nHasil → {out_path}")


if __name__ == "__main__":
    main()
