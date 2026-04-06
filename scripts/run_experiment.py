#!/usr/bin/env python3
"""
scripts/run_experiment.py
==========================
Orchestrate eksperimen multi-node untuk RQ3 ARTEMIS v2.

Untuk RQ3, N node dijalankan secara paralel mengirim frame ke satu server.
Node bisa berupa:
  - Pi5 real (via SSH)
  - VPS simulasi Pi3/Pi4B (via SSH + tc-netem)
  - Multiple instance lokal (untuk quick test)

CARA PAKAI:

  # N=3: Pi5 real + VPS Pi3 + VPS Pi4B
  python3 scripts/run_experiment.py \
      --nodes pi5_real:pi5.local,pi3_vps:192.168.1.10,pi4b_vps:192.168.1.11 \
      --experiment_id rq3_n3 \
      --methods adaptive,server_only

  # Test lokal (N=3 simulasi dari satu mesin)
  python3 scripts/run_experiment.py \
      --local_n 3 \
      --experiment_id rq3_local_test

  # Pantau progress
  curl http://SERVER_IP:8000/status | python3 -m json.tool
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_node_ssh(node_id: str, host: str, device_type: str,
                 experiment_id: str, methods: str,
                 results: dict, idx: int):
    """Jalankan run_edge.py di remote host via SSH."""
    cmd = (
        f"ssh {host} "
        f"'cd ~/artemis-v2 && "
        f"source artemis-env/bin/activate && "
        f"python3 scripts/run_edge.py "
        f"--device {device_type} "
        f"--node_id {node_id} "
        f"--experiment_id {experiment_id} "
        f"--methods {methods}'"
    )
    print(f"  [Node {idx+1}] Starting: {node_id} @ {host}")
    t0     = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    results[node_id] = {
        "returncode": result.returncode,
        "elapsed_s":  round(elapsed, 1),
        "stdout":     result.stdout[-2000:] if result.stdout else "",
        "stderr":     result.stderr[-500:]  if result.stderr else "",
    }
    status = "✓" if result.returncode == 0 else "✗"
    print(f"  [Node {idx+1}] {status} {node_id} selesai dalam {elapsed:.0f}s "
          f"(rc={result.returncode})")


def run_node_local(node_id: str, device_type: str,
                   experiment_id: str, methods: str,
                   node_id_suffix: str,
                   results: dict, idx: int):
    """Jalankan run_edge.py lokal (untuk testing)."""
    cmd = [
        sys.executable, "scripts/run_edge.py",
        "--device",        device_type,
        "--node_id",       f"{node_id}_{node_id_suffix}",
        "--experiment_id", experiment_id,
        "--methods",       methods,
    ]
    print(f"  [Node {idx+1}] Starting local: {node_id}_{node_id_suffix}")
    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    results[f"{node_id}_{node_id_suffix}"] = {
        "returncode": result.returncode,
        "elapsed_s":  round(elapsed, 1),
    }
    status = "✓" if result.returncode == 0 else "✗"
    print(f"  [Node {idx+1}] {status} selesai dalam {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIS v2 — Multi-Node Experiment Orchestrator (RQ3)")
    parser.add_argument("--nodes", default=None,
                        help="node_id:host:device_type pasangan dipisah koma, "
                             "misal: pi5_real:pi5.local:pi5,pi3_vps:192.168.1.10:pi3")
    parser.add_argument("--local_n", type=int, default=0,
                        help="Jumlah node lokal untuk testing (tanpa SSH)")
    parser.add_argument("--local_device", default="pi5",
                        choices=["pi3", "pi4b", "pi5"])
    parser.add_argument("--experiment_id", required=True,
                        help="ID eksperimen, misal: rq3_n3_urban_4g")
    parser.add_argument("--methods", default="adaptive,server_only",
                        help="Methods yang dijalankan di setiap node")
    parser.add_argument("--server_status_url", default=None,
                        help="URL untuk polling /status selama eksperimen")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ARTEMIS v2 — RQ3 Multi-Node Experiment")
    print(f"{'='*60}")
    print(f"  Experiment ID : {args.experiment_id}")
    print(f"  Methods       : {args.methods}")
    print(f"{'='*60}\n")

    threads = []
    results = {}

    if args.local_n > 0:
        # Mode lokal — untuk quick test tanpa hardware
        print(f"Mode lokal: {args.local_n} node paralel ({args.local_device})")
        for i in range(args.local_n):
            t = threading.Thread(
                target=run_node_local,
                args=(args.local_device, args.local_device,
                      args.experiment_id, args.methods,
                      str(i+1), results, i),
                daemon=True,
            )
            threads.append(t)

    elif args.nodes:
        # Mode SSH — node di remote hosts
        node_specs = [s.strip() for s in args.nodes.split(",")]
        print(f"Mode SSH: {len(node_specs)} remote nodes")
        for i, spec in enumerate(node_specs):
            parts = spec.split(":")
            if len(parts) != 3:
                print(f"ERROR: format node harus node_id:host:device_type, dapat: {spec}")
                sys.exit(1)
            node_id, host, device_type = parts
            t = threading.Thread(
                target=run_node_ssh,
                args=(node_id, host, device_type,
                      args.experiment_id, args.methods,
                      results, i),
                daemon=True,
            )
            threads.append(t)
    else:
        print("ERROR: Gunakan --nodes atau --local_n")
        parser.print_help()
        sys.exit(1)

    # Start semua node secara paralel
    t_start = time.time()
    print(f"\nStarting {len(threads)} nodes secara paralel...")
    for t in threads:
        t.start()
        time.sleep(0.5)  # Stagger start sedikit

    # Poll server status jika diminta
    if args.server_status_url:
        import requests
        print(f"\nPolling server status setiap 30s...")
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            try:
                resp = requests.get(
                    f"{args.server_status_url}/status", timeout=5)
                data = resp.json()
                print(f"\n[Server Status] "
                      f"requests={data.get('total_requests',0)} | "
                      f"nodes={data.get('active_nodes',0)} | "
                      f"errors={data.get('total_errors',0)}")
            except Exception:
                pass

    # Tunggu semua selesai
    for t in threads:
        t.join()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"EKSPERIMEN SELESAI dalam {elapsed:.0f}s")
    print(f"{'='*60}")

    n_ok = sum(1 for r in results.values() if r["returncode"] == 0)
    n_fail = len(results) - n_ok
    print(f"  Berhasil : {n_ok}/{len(results)} nodes")
    if n_fail > 0:
        print(f"  Gagal    : {n_fail} nodes")
        for nid, r in results.items():
            if r["returncode"] != 0:
                print(f"    ✗ {nid}: rc={r['returncode']}")

    print(f"\nHasil per node tersimpan di results/ masing-masing")
    print(f"Cek server stats: curl SERVER_IP:8000/status | python3 -m json.tool")


if __name__ == "__main__":
    main()
