#!/usr/bin/env python3
"""
scripts/run_demo.py — ARTEMIS v2 Demo Mode
==========================================
Inferensi dari video file, webcam, atau RTSP stream.

Contoh penggunaan:

  # Video file, tampilkan di layar + kirim alarm
  python3 scripts/run_demo.py --source video.mp4 --method server_only --show --send_alarm

  # Webcam
  python3 scripts/run_demo.py --source 0 --method adaptive --show

  # RTSP stream, kirim alarm + frame ke dashboard
  python3 scripts/run_demo.py --source rtsp://192.168.1.100/stream --method server_only \
      --send_alarm --send_frame

  # Tanpa tampilan layar, alarm saja
  python3 scripts/run_demo.py --source video.mp4 --method server_only --send_alarm
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import base64
import logging
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from edge.config import load_config
from edge.decision_engine import DecisionEngine
from edge.inference import EdgeInference
from edge.offloader import Offloader
from shared.constants import IMG_SIZE, CLASS_FIRE, CLASS_SMOKE
import json

from shared.features import extract_frame_features

log = logging.getLogger("artemis.run_demo")

TMP_FRAME = "/tmp/artemis_demo_frame.jpg"


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Dashboard sender ──────────────────────────────────────────────────────────
# HAPUS seluruh fungsi send_alarm() lama, ganti dengan ini:

def send_alarm(alarm_url, node_id, device_type, alarm_type, location,
               method, routing_decision, inference_ms,
               confmax_fire=0.0, confmax_smoke=0.0,
               frame_idx=-1, seq_id="", experiment_id="",
               frame_bgr=None, send_frame=False, frame_quality=60):
    frame_b64 = None
    if send_frame and frame_bgr is not None:
        try:
            ok, buf = cv2.imencode('.jpg', frame_bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), frame_quality])
            if ok:
                frame_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        except Exception as e:
            log.warning(f"Encode frame gagal: {e}")
    try:
        payload = {
            "node_id":           node_id,
            "device_type":       device_type,
            "alarm_type":        alarm_type,
            "location":          location,
            "experiment_id":     experiment_id,
            "method":            method,
            "routing_decision":  routing_decision,
            "alarm_decision_ms": round(inference_ms, 3),
            "confmax_fire":      round(confmax_fire,  4),
            "confmax_smoke":     round(confmax_smoke, 4),
            "frame_idx":         frame_idx,
            "seq_id":            seq_id,
            "frame_b64":         frame_b64,
            "sent_at":           time.time(),
        }
        resp = requests.post(alarm_url, json=payload, timeout=5)
        log.info(f"Alarm sent [{alarm_type}] routing={routing_decision} → {resp.status_code}")
    except Exception as e:
        log.warning(f"Alarm gagal: {e}")

def send_frame(frame_url, node_id, frame, decision, inference_ms):
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "node_id":      node_id,
            "timestamp":    datetime.now().isoformat(),
            "decision":     decision,
            "inference_ms": inference_ms,
            "frame_b64":    b64,
        }
        resp = requests.post(frame_url, json=payload, timeout=5)
        log.debug(f"Frame sent → {resp.status_code}")
    except Exception as e:
        log.warning(f"Frame gagal: {e}")


def notify_async(fn, **kwargs):
    threading.Thread(target=fn, kwargs=kwargs, daemon=True).start()


# ── Overlay ───────────────────────────────────────────────────────────────────

COLORS = {
    "FIRE":       (0,   0,   255),
    "SMOKE":      (0,   165, 255),
    "FIRE_SMOKE": (0,   0,   200),
    "NONE":       (0,   255, 0  ),
}

CLASS_LABEL = {CLASS_FIRE: "Fire", CLASS_SMOKE: "Smoke"}


def draw_overlay(frame: np.ndarray, boxes: list, alarm: str,
                 decision: str, inference_ms: float,
                 fps: float, method: str) -> np.ndarray:
    h, w = frame.shape[:2]

    # Gambar bounding boxes
    for box in boxes:
        cls_id = box["class_id"]
        conf   = box["confidence"]
        x1 = int(box["x1"] * w)
        y1 = int(box["y1"] * h)
        x2 = int(box["x2"] * w)
        y2 = int(box["y2"] * h)
        color = COLORS.get("FIRE") if cls_id == CLASS_FIRE else COLORS.get("SMOKE")
        label = f"{CLASS_LABEL.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = COLORS.get(alarm, (255, 255, 255))
    cv2.putText(frame, f"ARTEMIS v2 | {method.upper()}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame,
                f"Decision: {decision} | Alarm: {alarm} | {inference_ms:.0f}ms | {fps:.1f}fps",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    if alarm != "NONE":
        cv2.circle(frame, (w - 20, 20), 12, color, -1)

    return frame


# ── Helper ────────────────────────────────────────────────────────────────────

def frame_to_tmp(frame: np.ndarray, path: str = TMP_FRAME) -> str:
    """Simpan frame numpy ke file JPEG sementara, return path."""
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def boxes_to_summary(boxes: list) -> dict:
    """Hitung confmax/confavg/count dari boxes untuk DE dan alarm."""
    fire_confs  = [b["confidence"] for b in boxes if b["class_id"] == CLASS_FIRE]
    smoke_confs = [b["confidence"] for b in boxes if b["class_id"] == CLASS_SMOKE]
    return {
        "confmax_fire":  max(fire_confs,  default=0.0),
        "confavg_fire":  float(np.mean(fire_confs))  if fire_confs  else 0.0,
        "count_fire":    len(fire_confs),
        "confmax_smoke": max(smoke_confs, default=0.0),
        "confavg_smoke": float(np.mean(smoke_confs)) if smoke_confs else 0.0,
        "count_smoke":   len(smoke_confs),
    }


def determine_alarm(summary: dict, cfg) -> str:
    fire  = summary.get("confmax_fire",  0.0)
    smoke = summary.get("confmax_smoke", 0.0)
    thresh   = getattr(cfg, "edge_thresh", {})
    fire_th  = thresh.get("fire_local",  0.5)
    smoke_th = thresh.get("smoke_local", 0.5)
    has_fire  = fire  >= fire_th
    has_smoke = smoke >= smoke_th
    if has_fire and has_smoke: return "FIRE_SMOKE"
    if has_fire:               return "FIRE"
    if has_smoke:              return "SMOKE"
    return "NONE"


def build_de_features(summary: dict) -> np.ndarray:
    return np.array([
        summary["confmax_fire"],  summary["confavg_fire"],  summary["count_fire"],
        summary["confmax_smoke"], summary["confavg_smoke"], summary["count_smoke"],
    ], dtype=np.float32)


# ── Inferensi satu frame ──────────────────────────────────────────────────────

def infer_frame(frame: np.ndarray, method: str,
                inference: EdgeInference, de, offloader, cfg,
                prev_feats=None, global_frame_idx=0, seq_id="video") -> dict:
    t0  = time.time()
    tmp = frame_to_tmp(frame)

    boxes       = []
    de_decision = "LOCAL"
    server_ms   = 0.0
    summary     = {}
    feats       = None

    if method == "device_only":
        boxes, _ = inference.infer_with_boxes(tmp)
        summary  = boxes_to_summary(boxes)

    elif method == "server_only":
        srv, _      = offloader.offload(tmp)
        summary     = {k: srv.get(k, 0.0) for k in
                       ["confmax_fire","confavg_fire","count_fire",
                        "confmax_smoke","confavg_smoke","count_smoke"]}
        server_ms   = srv.get("server_inference_ms", 0.0)
        de_decision = "OFFLOAD"
        boxes       = []

    elif method == "adaptive":
        boxes, _ = inference.infer_with_boxes(tmp)
        summary  = boxes_to_summary(boxes)
        dets  = [{"class_id": b["class_id"], "confidence": b["confidence"]}
                 for b in boxes]
        feats = extract_frame_features(dets, prev_feats)
        de_decision, _, _ = de.predict(
            feats,
            seq_id=seq_id,
            forced_interval=cfg.forced_offload_interval,
            global_frame_idx=global_frame_idx,
        )
        if de_decision == "OFFLOAD":
            srv, _  = offloader.offload(tmp)
            summary = {k: srv.get(k, 0.0) for k in
                       ["confmax_fire","confavg_fire","count_fire",
                        "confmax_smoke","confavg_smoke","count_smoke"]}
            server_ms = srv.get("server_inference_ms", 0.0)
            boxes   = []

    alarm    = determine_alarm(summary, cfg)
    total_ms = (time.time() - t0) * 1000

    return {
        "decision":      de_decision,
        "alarm":         alarm,
        "boxes":         boxes,
        "inference_ms":  total_ms,
        "server_ms":     server_ms,
        "confmax_fire":  summary.get("confmax_fire",  0.0),
        "confmax_smoke": summary.get("confmax_smoke", 0.0),
        "feats":         feats,
    }


# ── Sequence mode ─────────────────────────────────────────────────────────────

def run_sequence_mode(args, cfg, inference, de, offloader):
    """
    Jalankan inferensi dari sequence_list JSON (seperti run_edge.py).
    Replikasi persis logika run_adaptive_cooperative di edge/node.py:
    - prev_feats di-track per sequence untuk delta features
    - de.predict() dipanggil dengan seq_id, forced_interval, global_frame_idx
    """
    seq_path   = args.source
    images_dir = Path(cfg.images_dir)

    with open(seq_path) as f:
        data = json.load(f)

    sequences = data.get("sequences", [])
    log.info(f"Sequences: {len(sequences)} | images_dir: {images_dir}")

    global_frame_idx = 0

    for seq in sequences:
        seq_id     = seq.get("seq_id", "unknown")
        frames     = seq.get("frames", [])
        prev_feats = None  # reset per sequence

        for local_idx, fname in enumerate(frames):
            img_path = images_dir / fname
            if not img_path.exists():
                global_frame_idx += 1
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                global_frame_idx += 1
                continue

            try:
                t0 = time.time()
                tmp = frame_to_tmp(frame)

                boxes, bd = inference.infer_with_boxes(tmp)

                # Konversi ke format detections untuk extract_frame_features
                dets  = [{"class_id": b["class_id"], "confidence": b["confidence"]}
                         for b in boxes]

                # Feature extraction dengan prev_feats (sama seperti run_edge)
                feats      = extract_frame_features(dets, prev_feats)
                prev_feats = feats

                if args.method == "adaptive" and de is not None:
                    # Panggil DE persis seperti run_edge
                    decision, de_ms, is_warmup = de.predict(
                        feats,
                        seq_id=seq_id,
                        forced_interval=cfg.forced_offload_interval,
                        global_frame_idx=global_frame_idx,
                    )
                elif args.method == "server_only":
                    decision = "OFFLOAD"
                else:
                    decision = "LOCAL"

                # Eksekusi berdasarkan decision
                summary   = boxes_to_summary(boxes)
                server_ms = 0.0

                if decision == "OFFLOAD" and offloader:
                    srv, _    = offloader.offload(tmp)
                    summary   = {
                        "confmax_fire":  srv.get("confmax_fire",  0.0),
                        "confavg_fire":  srv.get("confavg_fire",  0.0),
                        "count_fire":    srv.get("count_fire",    0),
                        "confmax_smoke": srv.get("confmax_smoke", 0.0),
                        "confavg_smoke": srv.get("confavg_smoke", 0.0),
                        "count_smoke":   srv.get("count_smoke",   0),
                    }
                    server_ms = srv.get("server_inference_ms", 0.0)
                    boxes     = []

                alarm    = determine_alarm(summary, cfg)
                total_ms = (time.time() - t0) * 1000

                print(
                    f"frame {global_frame_idx:4d} [{seq_id}]: "
                    f"{decision:8s} alarm={alarm:20s} "
                    f"fire={feats['confmax_fire']:.3f} "
                    f"smoke={feats['confmax_smoke']:.3f} "
                    f"({total_ms:.0f}ms)"
                )

                if args.send_alarm and alarm != "NONE":
                    t = threading.Thread(
                        target=send_alarm,
                        kwargs=dict(
                            alarm_url=args.alarm_url,
                            node_id=cfg.node_id,
                            device_type=cfg.device_type,
                            alarm_type=alarm,
                            location=args.location,
                            method=args.method,
                            routing_decision=result["decision"],
                            inference_ms=inf_ms,
                            confmax_fire=result["confmax_fire"],
                            confmax_smoke=result["confmax_smoke"],
                            frame_idx=global_frame_idx,
                            seq_id=seq_id,
                            experiment_id=getattr(args, "experiment_id", ""),
                            frame_bgr=frame.copy() if args.send_frame else None,
                            send_frame=args.send_frame,
                        ),
                        daemon=False,
                    )
                    t.start()
                    alarm_threads.append(t)

                if args.show:
                    display = draw_overlay(
                        frame.copy(), boxes, alarm, decision,
                        total_ms, 0.0, args.method,
                    )
                    h, w = display.shape[:2]
                    new_w = args.display_w
                    new_h = int(h * new_w / w)
                    display = cv2.resize(display, (new_w, new_h))
                    cv2.imshow("ARTEMIS v2 — Sequence", display)
                    if cv2.waitKey(args.seq_delay) & 0xFF == ord("q"):
                        return

            except Exception as e:
                log.warning(f"frame {global_frame_idx} gagal: {e}")

            global_frame_idx += 1

    if args.show:
        cv2.destroyAllWindows()
    log.info(f"Sequence selesai — total {global_frame_idx} frame diproses.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARTEMIS v2 — Demo Mode")
    parser.add_argument("--source",     required=True,
                        help="Video file, webcam index (0), atau RTSP URL")
    parser.add_argument("--method",     default="server_only",
                        choices=["device_only", "server_only", "adaptive"])
    parser.add_argument("--fps",        type=float, default=5.0,
                        help="Max FPS processing (default: 5)")
    parser.add_argument("--show",       action="store_true",
                        help="Tampilkan overlay di layar")
    parser.add_argument("--send_alarm", action="store_true",
                        help="Kirim alarm ke dashboard")
    parser.add_argument("--send_frame", action="store_true",
                        help="Kirim frame ke dashboard")
    parser.add_argument("--alarm_url",  default="https://weartemis.me/alarm")
    parser.add_argument("--frame_url",  default="https://weartemis.me/frame")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--device",     default=None, choices=["pi3", "pi4b", "pi5"])
    parser.add_argument("--location",   default="demo")
    parser.add_argument("--server",     default=None, help="Override server URL")
    parser.add_argument("--display_w",  type=int, default=960,
                        help="Lebar window display (default: 960)")
    parser.add_argument("--seq_delay",  type=int, default=100,
                        help="Delay antar frame di sequence mode dalam ms (default: 100)")
    parser.add_argument("--log_level",  default="INFO")
    args = parser.parse_args()
    alarm_threads = []

    setup_logging(args.log_level)

    cfg = load_config(config_path=args.config, device_override=args.device)
    if args.server:
        cfg.server_url = args.server

    log.info(f"Loading edge model: {cfg.model_edge} ({cfg.model_type})")
    inference = EdgeInference(cfg.model_edge, cfg.model_type)

    de = None
    if args.method == "adaptive":
        log.info(f"Loading DE model: {cfg.model_de}")
        de = DecisionEngine(cfg.model_de)

    offloader = None
    if args.method in ("server_only", "adaptive"):
        offloader = Offloader(
            server_url=cfg.server_url,
            node_id=cfg.node_id,
            device_type=cfg.device_type,
            timeout=cfg.request_timeout,
        )
        log.info(f"Server: {cfg.server_url}")

    # Sequence mode — source adalah path ke .json sequence list
    if args.source.endswith(".json"):
        log.info(f"Sequence mode: {args.source}")
        run_sequence_mode(args, cfg, inference, de, offloader)
        if offloader:
            offloader.close()
        return

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Tidak bisa membuka source: {args.source}")
        sys.exit(1)

    log.info(f"Source: {args.source} | Method: {args.method} | Max FPS: {args.fps}")
    log.info(f"show={args.show} | send_alarm={args.send_alarm} | send_frame={args.send_frame}")

    interval         = 1.0 / args.fps
    last_infer       = 0.0
    fps_counter      = 0
    fps_timer        = time.time()
    current_fps      = 0.0
    prev_feats       = None
    global_frame_idx = 0
    seq_id           = args.source  # pakai nama file/source sebagai seq_id
    last_result      = {"decision": "-", "alarm": "NONE",
                        "inference_ms": 0.0, "boxes": [], "feats": None}

    print("\nTekan 'q' untuk keluar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("Stream selesai.")
                break

            now = time.time()

            if now - last_infer >= interval:
                last_infer = now
                try:
                    result      = infer_frame(frame, args.method,
                                              inference, de, offloader, cfg,
                                              prev_feats=prev_feats,
                                              global_frame_idx=global_frame_idx,
                                              seq_id=seq_id)
                    last_result  = result
                    prev_feats   = result.get("feats") or prev_feats
                    global_frame_idx += 1
                    alarm        = result["alarm"]
                    inf_ms       = result["inference_ms"]

                    log.info(
                        f"[{result['decision']}] alarm={alarm} "
                        f"fire={result['confmax_fire']:.3f} "
                        f"smoke={result['confmax_smoke']:.3f} "
                        f"({inf_ms:.0f}ms)"
                    )

                    if args.send_alarm and alarm != "NONE":
                        t = threading.Thread(
                            target=send_alarm,
                            kwargs=dict(
                                alarm_url=args.alarm_url,
                                node_id=cfg.node_id,
                                device_type=cfg.device_type,
                                alarm_type=alarm,
                                location=args.location,
                                method=args.method,
                                routing_decision=result["decision"],
                                inference_ms=inf_ms,
                                confmax_fire=result["confmax_fire"],
                                confmax_smoke=result["confmax_smoke"],
                                frame_idx=global_frame_idx,
                                seq_id=seq_id,
                                experiment_id=getattr(args, "experiment_id", ""),
                                frame_bgr=frame.copy() if args.send_frame else None,
                                send_frame=args.send_frame,
                            ),
                            daemon=False,
                        )
                        t.start()
                        alarm_threads.append(t)

                except Exception as e:
                    log.warning(f"Inferensi gagal: {e}")

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            if args.show:
                display = draw_overlay(
                    frame.copy(),
                    last_result.get("boxes", []),
                    last_result["alarm"],
                    last_result["decision"],
                    last_result["inference_ms"],
                    current_fps,
                    args.method,
                )
                h, w = display.shape[:2]
                new_w = args.display_w
                new_h = int(h * new_w / w)
                display = cv2.resize(display, (new_w, new_h))
                cv2.imshow("ARTEMIS v2", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Dihentikan.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if offloader:
            offloader.close()
        if alarm_threads:
            log.info(f"Menunggu {len(alarm_threads)} alarm thread selesai...")
            for t in alarm_threads:
                t.join(timeout=5)
            log.info("Semua alarm terkirim.")


if __name__ == "__main__":
    main()
