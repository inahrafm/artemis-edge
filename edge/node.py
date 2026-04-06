"""
edge/node.py — ARTEMIS v2 v2.1
Dengan async notifikasi: notif dikirim di background, tidak blocking frame loop.
notify_async_fn dipassing dari run_edge.py supaya tidak ada circular import.
"""

import logging, threading, time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import psutil

from edge.decision_engine import DecisionEngine
from edge.inference import EdgeInference
from edge.offloader import Offloader
from shared.config_schema import EdgeNodeConfig
from shared.features import extract_frame_features

log = logging.getLogger("artemis.edge.node")


class HardwareTelemetry:
    def __init__(self, interval_ms=500):
        self._interval = interval_ms / 1000
        self._data = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running = False
        self._thread = None

    def start(self):
        self._data = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            try:
                self._data["cpu"].append(psutil.cpu_percent(interval=None))
                self._data["ram_mb"].append(psutil.virtual_memory().used / 1024 / 1024)
                self._data["temp_c"].append(self._read_temp())
            except Exception:
                pass
            time.sleep(self._interval)

    @staticmethod
    def _read_temp():
        try:
            temps = psutil.sensors_temperatures()
            for k in ["cpu_thermal","cpu-thermal","thermal_zone0","coretemp","k10temp"]:
                if k in temps and temps[k]:
                    return temps[k][0].current
            p = Path("/sys/class/thermal/thermal_zone0/temp")
            if p.exists():
                return float(p.read_text().strip()) / 1000
        except Exception:
            pass
        return 0.0

    def summary(self):
        def s(lst):
            return ({"avg": round(float(np.mean(lst)),2), "max": round(float(np.max(lst)),2)}
                    if lst else {"avg": 0.0, "max": 0.0})
        return {"cpu": s(self._data["cpu"]), "ram_mb": s(self._data["ram_mb"]),
                "temp_c": s(self._data["temp_c"])}


def _progress(i, total, method, extra=""):
    if i % 100 == 0 or i == total:
        pct = i / total * 100
        bar = "█" * (i * 20 // total) + "░" * (20 - i * 20 // total)
        print(f"\r  [{bar}] {i:>4}/{total} ({pct:4.1f}%) {method}{extra}",
              end="", flush=True)
    if i == total:
        print()


def _determine_alarm(source, features=None, server_resp=None, edge_thresh=None):
    if source == "local" and features and edge_thresh:
        if features.get("confmax_fire",  0) >= edge_thresh["fire_local"]:  return "FIRE_CRITICAL"
        if features.get("confmax_smoke", 0) >= edge_thresh["smoke_local"]: return "SMOKE_EARLY_WARNING"
        return "NONE"
    elif source == "server" and server_resp:
        d = server_resp.get("decision", "NONE")
        if d == "FIRE":  return "FIRE_CRITICAL"
        if d == "SMOKE": return "SMOKE_EARLY_WARNING"
    return "NONE"


def _static_decision(feats, thresh):
    if (feats["confmax_fire"]  >= thresh["fire_local"] or
            feats["confmax_smoke"] >= thresh["smoke_local"]): return "LOCAL"
    if (feats["confmax_fire"]  <  thresh["fire_drop"] and
            feats["confmax_smoke"] <  thresh["smoke_drop"]):  return "DROP"
    return "OFFLOAD"


def _do_notify(notify_async_fn, notifier, alarm, method, feats,
               total_ms, seq_id, frame_index, exp_id, location, gps):
    """Kirim notifikasi async jika alarm triggered."""
    if notify_async_fn is None or notifier is None or alarm == "NONE":
        return
    notify_async_fn(
        notifier,
        alarm_type        = alarm,
        method            = method,
        location          = location,
        experiment_id     = exp_id,
        seq_id            = seq_id,
        frame_index       = frame_index,
        confmax_fire      = feats.get("confmax_fire",  0.0),
        confmax_smoke     = feats.get("confmax_smoke", 0.0),
        alarm_decision_ms = total_ms,
        frame_total_ms    = total_ms,
        gps               = gps,
    )


def run_device_only(sequences, images_dir, inference, cfg, telemetry,
                    notifier=None, notify_async_fn=None,
                    exp_id="", location="", gps=""):
    results = []
    total   = sum(len(s["frames"]) for s in sequences)
    i       = 0
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists(): continue
            t0       = time.perf_counter()
            dets, bd = inference.infer(str(img_path))
            feats    = extract_frame_features(dets)
            alarm    = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            total_ms = (time.perf_counter() - t0) * 1000
            i += 1
            _progress(i, total, "device_only", f"  {total_ms:6.1f}ms")
            _do_notify(notify_async_fn, notifier, alarm, "device_only",
                       feats, total_ms, seq["seq_id"], i, exp_id, location, gps)
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "device_only",
                "disk_read_ms": bd["disk_read_ms"], "preprocess_ms": bd["preprocess_ms"],
                "edge_inference_ms": bd["edge_inference_ms"], "edge_total_ms": bd["edge_total_ms"],
                "network_total_ms": 0.0, "server_inference_ms": 0.0,
                "network_overhead_ms": 0.0, "de_ms": 0.0, "notification_ms": 0.0,
                "total_ms": round(total_ms, 3), "alarm": alarm,
                "confmax_fire": round(feats["confmax_fire"],4),
                "confmax_smoke": round(feats["confmax_smoke"],4),
                "is_warmup": False, "de_decision": "LOCAL",
            })
    telemetry.stop()
    return results


def run_server_only(sequences, images_dir, inference, offloader, telemetry,
                    notifier=None, notify_async_fn=None,
                    exp_id="", location="", gps=""):
    results = []
    total   = sum(len(s["frames"]) for s in sequences)
    i       = 0
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists(): continue
            t0          = time.perf_counter()
            _, disk_ms  = inference.read_raw(str(img_path))
            srv, net_bd = offloader.offload(str(img_path))
            total_ms    = (time.perf_counter() - t0) * 1000
            alarm       = _determine_alarm("server", server_resp=srv)
            i += 1
            err_flag = " ✗" if srv.get("network_error") else ""
            _progress(i, total, "server_only",
                      f"  {net_bd['network_total_ms']:6.1f}ms{err_flag}")
            feats_approx = {"confmax_fire": srv.get("confmax_fire",0.0),
                            "confmax_smoke": srv.get("confmax_smoke",0.0)}
            _do_notify(notify_async_fn, notifier, alarm, "server_only",
                       feats_approx, total_ms, seq["seq_id"], i, exp_id, location, gps)
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "server_only",
                "disk_read_ms": round(disk_ms,3),
                "preprocess_ms": 0.0, "edge_inference_ms": 0.0, "edge_total_ms": 0.0,
                "network_total_ms": net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms": 0.0, "notification_ms": 0.0,
                "total_ms": round(total_ms,3), "alarm": alarm,
                "confmax_fire": srv.get("confmax_fire",0.0),
                "confmax_smoke": srv.get("confmax_smoke",0.0),
                "is_warmup": False, "de_decision": "OFFLOAD",
                "network_error": bool(srv.get("network_error",False)),
            })
    telemetry.stop()
    return results


def run_static_cooperative(sequences, images_dir, inference, offloader, cfg,
                            telemetry, notifier=None, notify_async_fn=None,
                            exp_id="", location="", gps=""):
    results = []
    total   = sum(len(s["frames"]) for s in sequences)
    i       = 0
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists(): continue
            t0       = time.perf_counter()
            dets, bd = inference.infer(str(img_path))
            feats    = extract_frame_features(dets)
            decision = _static_decision(feats, cfg.edge_thresh)
            alarm    = "NONE"
            net_bd   = {"network_total_ms":0.0,"server_inference_ms":0.0,"network_overhead_ms":0.0}
            if decision == "LOCAL":
                alarm = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            elif decision == "OFFLOAD":
                srv, net_bd = offloader.offload(str(img_path))
                alarm = _determine_alarm("server", server_resp=srv)
            total_ms = (time.perf_counter() - t0) * 1000
            i += 1
            _progress(i, total, "static", f"  {decision:<7} {total_ms:6.1f}ms")
            _do_notify(notify_async_fn, notifier, alarm, "static_cooperative",
                       feats, total_ms, seq["seq_id"], i, exp_id, location, gps)
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "static_cooperative",
                "de_decision": decision,
                "disk_read_ms": bd["disk_read_ms"], "preprocess_ms": bd["preprocess_ms"],
                "edge_inference_ms": bd["edge_inference_ms"], "edge_total_ms": bd["edge_total_ms"],
                "network_total_ms": net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms": 0.0, "notification_ms": 0.0,
                "total_ms": round(total_ms,3), "alarm": alarm,
                "confmax_fire": round(feats["confmax_fire"],4),
                "confmax_smoke": round(feats["confmax_smoke"],4),
                "is_warmup": False,
            })
    telemetry.stop()
    n_off = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    log.info(f"[static] offload_rate={n_off/len(results):.1%}")
    return results


def run_adaptive_cooperative(sequences, images_dir, inference, de, offloader,
                              cfg, telemetry, notifier=None, notify_async_fn=None,
                              exp_id="", location="", gps=""):
    results          = []
    global_frame_idx = 0
    total            = sum(len(s["frames"]) for s in sequences)
    i                = 0
    telemetry.start()

    for seq in sequences:
        prev_feats = None
        for local_idx, fname in enumerate(seq["frames"]):
            img_path = images_dir / fname
            if not img_path.exists():
                global_frame_idx += 1
                continue
            t0       = time.perf_counter()
            dets, bd = inference.infer(str(img_path))
            feats    = extract_frame_features(dets, prev_feats)
            prev_feats = feats
            decision, de_ms, is_warmup = de.predict(
                feats, seq_id=seq["seq_id"],
                forced_interval=cfg.forced_offload_interval,
                global_frame_idx=global_frame_idx)
            alarm  = "NONE"
            net_bd = {"network_total_ms":0.0,"server_inference_ms":0.0,"network_overhead_ms":0.0}
            if decision == "LOCAL":
                alarm = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            elif decision == "OFFLOAD":
                srv, net_bd = offloader.offload(str(img_path))
                alarm = _determine_alarm("server", server_resp=srv)
            total_ms = (time.perf_counter() - t0) * 1000
            i += 1
            _progress(i, total, "adaptive", f"  {decision:<7} {total_ms:6.1f}ms")
            _do_notify(notify_async_fn, notifier, alarm, "adaptive_cooperative",
                       feats, total_ms, seq["seq_id"], local_idx, exp_id, location, gps)
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "adaptive_cooperative",
                "local_frame_idx": local_idx, "global_frame_idx": global_frame_idx,
                "de_decision": decision,
                "is_forced_offload": (cfg.forced_offload_interval > 0 and
                                      global_frame_idx % cfg.forced_offload_interval == 0),
                "is_warmup": is_warmup,
                "disk_read_ms": bd["disk_read_ms"], "preprocess_ms": bd["preprocess_ms"],
                "edge_inference_ms": bd["edge_inference_ms"], "edge_total_ms": bd["edge_total_ms"],
                "de_ms": de_ms,
                "network_total_ms": net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "notification_ms": 0.0,
                "total_ms": round(total_ms,3), "alarm": alarm,
                "confmax_fire": round(feats["confmax_fire"],4),
                "confmax_smoke": round(feats["confmax_smoke"],4),
                "recent_offload_rate": round(de.recent_offload_rate,4),
            })
            global_frame_idx += 1

    telemetry.stop()
    n_off = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    log.info(f"[adaptive] offload_rate={n_off/len(results):.1%}")
    return results
