"""
quick_latency_check.py
=======================
Cek cepat breakdown latensi tanpa evaluasi penuh.
Ambil N frame acak dari images_dir, ukur semua komponen.

Gunakan untuk:
  1. Verifikasi server aktif sebelum evaluasi penuh
  2. Quick benchmark setelah ganti model/hardware
  3. Cek apakah server_inference_ms terbaca dengan benar

Run dari Pi:
    python3 scripts/quick_latency_check.py \
        --images_dir  data/full_test/images \
        --model_edge  models/best.onnx \
        --model_type  onnx \
        --server_url  http://192.168.100.154:8000 \
        --n_frames    30
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger

log = get_logger("quick_latency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--model_edge",  required=True)
    parser.add_argument("--model_type",  required=True,
                        choices=["onnx", "tflite"])
    parser.add_argument("--server_url",  default="http://localhost:8000")
    parser.add_argument("--n_frames",    type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    images_dir = Path(args.images_dir)

    # ── 1. Cek server ─────────────────────────────────────────────────────────
    import requests
    print(f"\n[1] Cek server: {args.server_url}")
    try:
        r = requests.get(f"{args.server_url}/health", timeout=5)
        data = r.json()
        print(f"    Status    : {data.get('status')}")
        print(f"    Device    : {data.get('device')}")
        print(f"    Thresholds: {'dari file ✓' if data.get('thresholds_from_file') else 'DEFAULT ⚠'}")
        print(f"    Uptime    : {data.get('uptime_s')} s")
        server_ok = data.get("status") == "ok"
    except Exception as e:
        print(f"    ✗ Server tidak aktif: {e}")
        print("    Jalankan server_inference.py di Ubuntu dulu!")
        server_ok = False

    # ── 2. Load edge model ────────────────────────────────────────────────────
    print(f"\n[2] Load edge model: {args.model_edge}")
    sys.path.insert(0, str(Path(__file__).parent))
    from phase_f_pi_evaluation import EdgeModel, offload_to_server_with_breakdown
    try:
        model = EdgeModel(args.model_edge, args.model_type)
        print(f"    ✓ Model loaded ({args.model_type})")
    except Exception as e:
        print(f"    ✗ Gagal load model: {e}")
        sys.exit(1)

    # ── 3. Pilih frame acak ───────────────────────────────────────────────────
    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not all_images:
        print(f"✗ Tidak ada gambar di {images_dir}")
        sys.exit(1)

    sample = random.sample(all_images, min(args.n_frames, len(all_images)))
    print(f"\n[3] Benchmark {len(sample)} frame acak dari {len(all_images)} gambar")

    # ── 4. Ukur latensi ───────────────────────────────────────────────────────
    session = requests.Session()
    results = []

    for i, img_path in enumerate(sample):
        print(f"    Frame {i+1:3d}/{len(sample)}", end="\r")

        t_total = time.perf_counter()

        # Edge inference dengan breakdown
        dets, bd = model.infer_with_breakdown(str(img_path))

        # Server request
        if server_ok:
            srv, net_bd = offload_to_server_with_breakdown(
                str(img_path), args.server_url, session
            )
        else:
            net_bd = {"network_total_ms": 0, "server_inference_ms": 0,
                      "network_overhead_ms": 0}

        total_ms = (time.perf_counter() - t_total) * 1000
        results.append({**bd, **net_bd, "total_ms": total_ms})

    print(f"    Selesai {len(results)} frame")

    # ── 5. Ringkasan ──────────────────────────────────────────────────────────
    def stats(key):
        vals = [r[key] for r in results if key in r]
        if not vals:
            return "N/A"
        return (f"avg={np.mean(vals):.1f}ms  "
                f"p50={np.percentile(vals,50):.1f}ms  "
                f"p95={np.percentile(vals,95):.1f}ms")

    print(f"\n{'='*60}")
    print(f"HASIL QUICK LATENCY CHECK ({args.model_type.upper()})")
    print(f"{'='*60}")
    print(f"  Disk Read        : {stats('disk_read_ms')}")
    print(f"  Preprocess       : {stats('preprocess_ms')}")
    print(f"  Edge Inference   : {stats('edge_inference_ms')}")
    print(f"  Edge Total       : {stats('edge_total_ms')}")
    if server_ok:
        print(f"  Network RT       : {stats('network_total_ms')}")
        print(f"  Server GPU Inf   : {stats('server_inference_ms')}")
        print(f"  Net Overhead     : {stats('network_overhead_ms')}")
    print(f"  TOTAL (edge+net) : {stats('total_ms')}")
    print(f"{'='*60}")

    if server_ok:
        srv_inf = [r["server_inference_ms"] for r in results]
        if all(v == 0 for v in srv_inf):
            print("\n  ⚠ server_inference_ms = 0 untuk semua frame!")
            print("    Pastikan server_inference.py sudah diupdate ke versi terbaru")
            print("    yang mengembalikan field 'server_inference_ms' terpisah.")
        else:
            print(f"\n  ✓ server_inference_ms terbaca dengan benar")

    print()


if __name__ == "__main__":
    main()
