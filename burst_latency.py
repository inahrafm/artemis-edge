#!/usr/bin/env python3
"""
burst_latency.py — Artemis Field Measurement
Simulasi pengiriman frame ke server, ukur latency per transfer.

Usage:
    python3 burst_latency.py \
        --host artemis.domain.com \
        --size 150 \
        --repeat 200 \
        --interval 2 \
        --output burst_output.txt
"""

import argparse
import socket
import time
import os
import json
import statistics
import sys
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Artemis burst latency measurement")
    p.add_argument("--host",     required=True,       help="Server hostname/IP")
    p.add_argument("--port",     type=int, default=9999, help="Burst test port (default: 9999)")
    p.add_argument("--size",     type=int, default=150,  help="Payload size in KB (default: 150)")
    p.add_argument("--repeat",   type=int, default=200,  help="Number of transfers (default: 200)")
    p.add_argument("--interval", type=float, default=2.0, help="Interval between transfers in seconds")
    p.add_argument("--output",   default="burst.txt",   help="Output file path")
    p.add_argument("--timeout",  type=float, default=10.0, help="Socket timeout seconds")
    return p.parse_args()


def send_burst(host, port, payload_bytes, timeout):
    """Kirim payload ke server, ukur round-trip duration."""
    t_start = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Kirim header: ukuran payload (4 bytes big-endian)
            size = len(payload_bytes)
            sock.sendall(size.to_bytes(4, byteorder='big'))
            # Kirim payload
            sock.sendall(payload_bytes)
            # Tunggu ACK (1 byte) dari server
            ack = sock.recv(1)
            if ack != b'\x01':
                return None, "bad_ack"
        t_end = time.time()
        duration_ms = (t_end - t_start) * 1000
        return duration_ms, "ok"
    except socket.timeout:
        return None, "timeout"
    except ConnectionRefusedError:
        return None, "refused"
    except Exception as e:
        return None, str(e)


def percentile(data, p):
    """Hitung persentil ke-p dari list data."""
    if not data:
        return None
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def main():
    args = parse_args()

    payload = os.urandom(args.size * 1024)  # random bytes = simulate frame data
    results = []
    errors = {"timeout": 0, "refused": 0, "bad_ack": 0, "other": 0}

    print(f"\n{'='*55}")
    print(f"  Artemis Burst Latency Test")
    print(f"{'='*55}")
    print(f"  Host    : {args.host}:{args.port}")
    print(f"  Payload : {args.size} KB ({len(payload):,} bytes)")
    print(f"  Repeat  : {args.repeat}x @ {args.interval}s interval")
    print(f"  Output  : {args.output}")
    print(f"{'='*55}\n")

    # ── Check apakah port bisa dijangkau dulu ────────────────────────────────
    print("Checking server port... ", end="", flush=True)
    try:
        with socket.create_connection((args.host, args.port), timeout=5):
            print("OK\n")
    except Exception as e:
        print(f"GAGAL ({e})")
        print("\n⚠ Server burst listener belum jalan.")
        print("Jalankan di homeserver:")
        print(f"  python3 burst_server.py --port {args.port}\n")
        sys.exit(1)

    # ── Run transfers ────────────────────────────────────────────────────────
    for i in range(1, args.repeat + 1):
        duration_ms, status = send_burst(args.host, args.port, payload, args.timeout)

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if status == "ok":
            results.append(duration_ms)
            bar = "█" * min(int(duration_ms / 20), 30)
            print(f"  [{i:>3}/{args.repeat}] {ts}  {duration_ms:>8.1f} ms  {bar}")
        else:
            if status in errors:
                errors[status] += 1
            else:
                errors["other"] += 1
            print(f"  [{i:>3}/{args.repeat}] {ts}  {'TIMEOUT' if status=='timeout' else 'ERROR':>8}  ✗ ({status})")

        if i < args.repeat:
            time.sleep(args.interval)

    # ── Statistik ────────────────────────────────────────────────────────────
    n_ok    = len(results)
    n_fail  = args.repeat - n_ok
    loss_pct = (n_fail / args.repeat) * 100

    stats = {}
    if results:
        stats = {
            "min":    min(results),
            "max":    max(results),
            "avg":    statistics.mean(results),
            "median": statistics.median(results),
            "stdev":  statistics.stdev(results) if len(results) > 1 else 0,
            "p50":    percentile(results, 50),
            "p90":    percentile(results, 90),
            "p95":    percentile(results, 95),
            "p99":    percentile(results, 99),
        }

    summary_lines = [
        "",
        "=" * 55,
        "  BURST LATENCY SUMMARY",
        "=" * 55,
        f"  Total        : {args.repeat} transfers",
        f"  Sukses       : {n_ok}",
        f"  Gagal        : {n_fail}  (loss {loss_pct:.1f}%)",
        f"    timeout    : {errors['timeout']}",
        f"    refused    : {errors['refused']}",
        f"    other      : {errors['other']}",
    ]

    if stats:
        summary_lines += [
            "",
            f"  Min          : {stats['min']:.1f} ms",
            f"  Max          : {stats['max']:.1f} ms",
            f"  Avg          : {stats['avg']:.1f} ms",
            f"  Median (p50) : {stats['median']:.1f} ms",
            f"  Stdev        : {stats['stdev']:.1f} ms",
            f"  p90          : {stats['p90']:.1f} ms",
            f"  p95          : {stats['p95']:.1f} ms",
            f"  p99          : {stats['p99']:.1f} ms",
        ]

    summary_lines += ["=" * 55, ""]

    for line in summary_lines:
        print(line)

    # ── Tulis output file ────────────────────────────────────────────────────
    output_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "host": args.host,
            "port": args.port,
            "payload_kb": args.size,
            "repeat": args.repeat,
            "interval_s": args.interval,
        },
        "results": {
            "n_ok": n_ok,
            "n_fail": n_fail,
            "loss_pct": round(loss_pct, 2),
            "errors": errors,
            "stats": {k: round(v, 3) for k, v in stats.items()} if stats else {},
        },
        "raw_ms": [round(x, 3) for x in results],
    }

    with open(args.output, "w") as f:
        # Human-readable header
        f.write("\n".join(summary_lines) + "\n\n")
        # JSON untuk parsing otomatis
        f.write("=== JSON DATA ===\n")
        json.dump(output_data, f, indent=2)
        f.write("\n")

    print(f"Output disimpan → {args.output}")


if __name__ == "__main__":
    main()
