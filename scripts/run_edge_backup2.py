#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
scripts/run_edge.py — ARTEMIS v2 Edge Node Entry Point v2.1

Perubahan:
  - Checkpoint per method: save setelah setiap method selesai
  - --resume: skip method yang sudah ada di checkpoint
  - Async notifikasi: notif di background thread, tidak blocking frame loop
  - --offload_timeout: override timeout offloader
  - Auto-skip server_only jika RTT > --rtt_threshold (default 500ms)
  - Auto-reconnect: retry 5x sebelum abort
  - --no_notify: disable notifikasi alarm
"""

import argparse, gc, json, logging, threading, time
from datetime import datetime
from pathlib import Path

from edge.notifier        import AlarmNotifier
from edge.config          import load_config
from edge.decision_engine import DecisionEngine
from edge.inference       import EdgeInference
from edge.node import (
    HardwareTelemetry,
    run_adaptive_cooperative,
    run_device_only,
    run_server_only,
    run_static_cooperative,
)
from edge.offloader import Offloader

log = logging.getLogger("artemis.run_edge")


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Async notifikasi ──────────────────────────────────────────────────────────

def notify_async(notifier, **kwargs):
    """Kirim notifikasi di background thread — frame loop tidak pernah nunggu."""
    if notifier is None:
        return
    threading.Thread(target=notifier.notify, kwargs=kwargs, daemon=True).start()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _cp_path(out_dir, exp_id, method_key):
    return out_dir / f"{exp_id}_CHECKPOINT_{method_key}.json"


def save_checkpoint(out_dir, exp_id, method_key, data, cfg, args):
    cp = _cp_path(out_dir, exp_id, method_key)
    with open(cp, "w") as f:
        json.dump({
            "metadata": {
                "node_id": cfg.node_id, "device_type": cfg.device_type,
                "experiment_id": exp_id, "location": args.location,
                "operator": args.operator, "checkpoint": True,
                "method": method_key, "timestamp": datetime.now().isoformat(),
            },
            "methods": {method_key: data},
        }, f, indent=2)
    log.info(f"Checkpoint saved → {cp.name}")


def load_checkpoints(out_dir, exp_id):
    existing = {}
    for mk in ["device_only", "server_only",
                "static_cooperative", "adaptive_cooperative"]:
        cp = _cp_path(out_dir, exp_id, mk)
        if cp.exists():
            try:
                d = json.loads(cp.read_text())
                existing[mk] = d["methods"][mk]
                log.info(f"Resume: checkpoint ditemukan → {mk}")
            except Exception as e:
                log.warning(f"Gagal load checkpoint {cp.name}: {e}")
    return existing


def merge_final(out_dir, exp_id, all_results, cfg, args, offloader):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{exp_id}_{cfg.node_id}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "node_id": cfg.node_id, "device_type": cfg.device_type,
                "experiment_id": exp_id, "location": args.location,
                "operator": args.operator, "server_url": cfg.server_url,
                "timestamp": datetime.now().isoformat(),
                "offloader_stats": offloader.stats,
            },
            "methods": all_results,
        }, f, indent=2)
    # Hapus checkpoint
    for mk in ["device_only", "server_only",
                "static_cooperative", "adaptive_cooperative"]:
        cp = _cp_path(out_dir, exp_id, mk)
        if cp.exists():
            cp.unlink()
    return out_path


# ── Summary ───────────────────────────────────────────────────────────────────

def compute_summary(results, method_key):
    import numpy as np
    n = len(results)
    if n == 0:
        return {"n_frames": 0}

    steady = [r for r in results if not r.get("is_warmup")]
    warmup = [r for r in results if r.get("is_warmup")]

    def _s(lst, key):
        vals = [r[key] for r in lst if key in r and r[key] is not None]
        if not vals:
            return {"avg": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
        a = np.array(vals)
        return {"avg": round(float(a.mean()),3), "std": round(float(a.std()),3),
                "p50": round(float(np.percentile(a,50)),3),
                "p95": round(float(np.percentile(a,95)),3)}

    def _m(lst, key):
        vals = [r[key] for r in lst if key in r]
        return round(float(np.mean(vals)),3) if vals else 0.0

    dist  = {k: sum(1 for r in results if r.get("de_decision")==k)
             for k in ["LOCAL","OFFLOAD","DROP"]}
    n_off = dist.get("OFFLOAD", 0)

    keys  = ["disk_read_ms","preprocess_ms","edge_inference_ms","edge_total_ms",
             "de_ms","network_total_ms","server_inference_ms","network_overhead_ms",
             "notification_ms","total_ms"]
    bd    = {k: _s(steady, k) for k in keys}

    off_frames = [r for r in steady
                  if r.get("de_decision")=="OFFLOAD" or r.get("method")=="server_only"]
    if off_frames:
        bd["network_total_ms_offload_only"]    = _s(off_frames, "network_total_ms")
        bd["server_inference_ms_offload_only"] = _s(off_frames, "server_inference_ms")

    n_err = sum(1 for r in results if r.get("network_error", False))
    return {
        "n_frames": n, "n_warmup": len(warmup), "n_steady": len(steady),
        "avg_total_ms": _m(results, "total_ms"),
        "steady_avg_total_ms": _m(steady, "total_ms"),
        "alarm_count": sum(1 for r in results if r.get("alarm")!="NONE"),
        "offload_rate": round(n_off/n, 4),
        "decision_dist": dist,
        "avg_de_ms": _m(results, "de_ms"),
        "n_forced_offload": sum(1 for r in results if r.get("is_forced_offload")),
        "latency_breakdown": bd,
        "n_network_errors": n_err,
        "network_error_rate": round(n_err/n, 4) if n>0 else 0.0,
    }


def load_sequences(cfg, mode="sequences"):
    images_dir = Path(cfg.images_dir)
    if mode == "sequences" and Path(cfg.sequences).exists():
        with open(cfg.sequences) as f:
            data = json.load(f)
        seqs  = data.get("sequences", [])
        valid = []
        for s in seqs:
            vf = [fr for fr in s["frames"] if (images_dir/fr).exists()]
            if vf: valid.append({**s, "frames": vf})
        print(f"  Sequences: {len(valid)}/{len(seqs)} valid")
        return valid
    exts   = {".jpg",".jpeg",".png"}
    frames = sorted([f.name for f in images_dir.iterdir()
                     if f.suffix.lower() in exts])
    print(f"  Folder mode: {len(frames)} gambar")
    return [{"seq_id":"folder_all","seq_type":"continuous","frames":frames}]


def print_summary(node_id, location, operator, all_results):
    print(f"\n{'='*65}")
    print(f"RINGKASAN — {node_id.upper()} | {location} | {operator}")
    print(f"{'='*65}")
    print(f"  {'Method':<28} {'Avg(ms)':>8} {'P95(ms)':>8} {'Offload%':>9} {'NetErr':>7}")
    print(f"  {'-'*62}")
    for mk, md in all_results.items():
        s   = md["summary"]
        bd  = s.get("latency_breakdown", {})
        avg = bd.get("total_ms", {}).get("avg", s.get("steady_avg_total_ms", 0))
        p95 = bd.get("total_ms", {}).get("p95", 0)
        off = s.get("offload_rate", 0) * 100
        err = s.get("n_network_errors", 0)
        print(f"  {mk:<28} {avg:>7.1f}ms {p95:>7.1f}ms {off:>8.1f}% {err:>7}")
    print(f"{'='*65}")


# ── Auto-reconnect ────────────────────────────────────────────────────────────

def wait_for_server(offloader, max_retries=5, retry_delay=30):
    for attempt in range(1, max_retries+1):
        if offloader.health_check():
            return True
        if attempt < max_retries:
            log.warning(f"Server tidak respond ({attempt}/{max_retries}), "
                        f"retry dalam {retry_delay}s...")
            time.sleep(retry_delay)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARTEMIS v2 — Edge Node v2.1")
    parser.add_argument("--config",          default=None)
    parser.add_argument("--device",          default=None, choices=["pi3","pi4b","pi5"])
    parser.add_argument("--node_id",         default=None)
    parser.add_argument("--server",          default=None)
    parser.add_argument("--location",        default="unspecified")
    parser.add_argument("--operator",        default="unspecified")
    parser.add_argument("--experiment_id",   default=None)
    parser.add_argument("--methods",         default="all",
                        help="all | device_only,server_only,static,adaptive")
    parser.add_argument("--mode",            default="sequences",
                        choices=["sequences","folder"])
    parser.add_argument("--log_level",       default="INFO")
    parser.add_argument("--offload_timeout", default=None, type=float,
                        help="Override timeout offloader (detik)")
    parser.add_argument("--rtt_threshold",   default=500.0, type=float,
                        help="Auto-skip server_only jika RTT avg > threshold ms")
    parser.add_argument("--resume",          action="store_true",
                        help="Resume dari checkpoint yang sudah ada")
    parser.add_argument("--no_notify",       action="store_true",
                        help="Disable alarm notification ke dashboard")
    parser.add_argument("--init-config",     action="store_true")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.init_config:
        device = args.device or "pi5"
        import yaml
        print(yaml.dump({
            "node_id": f"{device}_node", "device_type": device,
            "server_url": "http://10.99.0.2:8000",
            "images_dir": "data/full_test/images",
            "sequences":  "sequences/sequence_list_v2.json",
            "thresholds": "thresholds_v2.json",
            "output_dir": "results",
            "request_timeout": {"pi3":30,"pi4b":20,"pi5":15}[device],
            "forced_offload_interval": 50,
        }, default_flow_style=False, sort_keys=False))
        return

    cfg = load_config(config_path=args.config, device_override=args.device)
    if args.node_id:         cfg.node_id       = args.node_id
    if args.server:          cfg.server_url    = args.server
    if args.offload_timeout: cfg.request_timeout = args.offload_timeout

    exp_id  = args.experiment_id or f"{cfg.device_type}_{args.location}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading edge model: {cfg.model_edge} ({cfg.model_type})")
    inference = EdgeInference(cfg.model_edge, cfg.model_type)
    print(f"Loading DE model: {cfg.model_de}")
    de = DecisionEngine(cfg.model_de)

    notifier = AlarmNotifier(
        alarm_url   = "https://weartemis.me/alarm",
        node_id     = cfg.node_id,
        device_type = cfg.device_type,
        enabled     = not args.no_notify,
    )

    offloader = Offloader(
        server_url    = cfg.server_url,
        node_id       = cfg.node_id,
        device_type   = cfg.device_type,
        timeout       = cfg.request_timeout,
        experiment_id = exp_id,
    )

    # Auto-reconnect
    print("\nChecking server...")
    server_ok = wait_for_server(offloader, max_retries=5, retry_delay=30)
    if not server_ok:
        log.warning("Server tidak bisa dijangkau setelah 5 percobaan.")

    sequences = load_sequences(cfg, mode=args.mode)
    if not sequences:
        print("ERROR: Tidak ada sequence valid.")
        sys.exit(1)

    # Tentukan methods
    methods = (["device_only","server_only","static","adaptive"]
               if args.methods == "all"
               else [m.strip() for m in args.methods.split(",")])

    if not server_ok:
        skipped = [m for m in methods if m != "device_only"]
        methods = ["device_only"] if "device_only" in methods else []
        if skipped:
            log.warning(f"Server tidak tersedia — skip: {skipped}")

    # Auto-skip server_only jika RTT terlalu tinggi
    if server_ok and "server_only" in methods:
        rtt = offloader.measure_rtt() if hasattr(offloader, "measure_rtt") else None
        if rtt and rtt > args.rtt_threshold:
            log.warning(f"RTT {rtt:.0f}ms > threshold {args.rtt_threshold:.0f}ms "
                        f"— auto-skip server_only")
            methods = [m for m in methods if m != "server_only"]
        elif rtt:
            log.info(f"RTT avg: {rtt:.0f}ms — OK")

    print(f"\nMethods: {methods}")
    print(f"Frames: {sum(len(s['frames']) for s in sequences)}")

    # Load checkpoints jika resume
    all_results = load_checkpoints(out_dir, exp_id) if args.resume else {}
    if all_results:
        print(f"  Resume: {list(all_results.keys())} sudah ada → di-skip")

    telemetry = HardwareTelemetry(interval_ms=300)
    gps       = getattr(cfg, "gps", "")
    notif_kw  = dict(notifier=notifier, notify_async_fn=notify_async,
                     exp_id=exp_id, location=args.location, gps=gps)

    def run_method(name, fn, *fn_args):
        mk = {"device_only":"device_only","server_only":"server_only",
              "static":"static_cooperative","adaptive":"adaptive_cooperative"}[name]
        if mk in all_results:
            print(f"\n[{cfg.node_id}] === {mk} === (SKIPPED — checkpoint ada)")
            return
        print(f"\n[{cfg.node_id}] === {mk} ===")
        res  = fn(*fn_args, **notif_kw)
        data = {"frame_results": res, "hardware": telemetry.summary(),
                "summary": compute_summary(res, mk)}
        all_results[mk] = data
        save_checkpoint(out_dir, exp_id, mk, data, cfg, args)
        gc.collect()

    images_dir = Path(cfg.images_dir)

    if "device_only" in methods:
        run_method("device_only", run_device_only,
                   sequences, images_dir, inference, cfg, telemetry)

    if "server_only" in methods:
        run_method("server_only", run_server_only,
                   sequences, images_dir, inference, offloader, telemetry)

    if "static" in methods:
        run_method("static", run_static_cooperative,
                   sequences, images_dir, inference, offloader, cfg, telemetry)

    if "adaptive" in methods:
        run_method("adaptive", run_adaptive_cooperative,
                   sequences, images_dir, inference, de, offloader, cfg, telemetry)

    if all_results:
        out_path = merge_final(out_dir, exp_id, all_results, cfg, args, offloader)
        offloader.close()
        print_summary(cfg.node_id, args.location, args.operator, all_results)
        print(f"\nHasil → {out_path}")
    else:
        print("Tidak ada hasil yang disimpan.")


if __name__ == "__main__":
    main()
