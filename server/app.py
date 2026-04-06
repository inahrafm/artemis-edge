"""
server/app.py
=============
FastAPI async inference server untuk ARTEMIS v2 Topik 3.

Upgrade dari Flask (server_inference.py) ke FastAPI:
- Async endpoint handling — lebih efisien untuk concurrent N Pi (RQ3)
- X-Node-ID header tracking built-in
- /status endpoint dengan per-node stats
- Backward compatible dengan format request/response existing
- Gunicorn-ready dengan uvicorn worker

CARA MENJALANKAN:

  # Development
  python3 scripts/run_server.py --model best.pt --port 8000

  # Production (Gunicorn + uvicorn worker — untuk RQ3 multi-node)
  ARTEMIS_MODEL=/path/to/best.pt ARTEMIS_DEVICE=cuda \
  gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
           -b 0.0.0.0:8000 --timeout 60 \
           "server.app:create_app()"

ENDPOINTS:
  GET  /health     → status server (backward compatible)
  GET  /status     → per-node stats untuk monitoring RQ3
  GET  /nodes      → daftar node aktif
  POST /infer      → terima gambar, return deteksi
  GET  /thresholds
  POST /reload_thresholds
  POST /reload_model
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from server import inference as server_inference
from server.logger import RequestLogger
from shared.constants import DEFAULT_THRESHOLDS

log = logging.getLogger("artemis.server.app")

# Global state
_logger:     Optional[RequestLogger] = None
_req_counter = 0
_start_time  = time.time()
_thresholds_path = "thresholds_v2.json"


def _load_thresholds(path: str) -> bool:
    global _thresholds_path
    _thresholds_path = path
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            thresh = data.get("server_model", {})
            if thresh:
                server_inference.set_thresholds(thresh)
                log.info(f"Thresholds dimuat: {thresh}")
                return True
        except Exception as e:
            log.error(f"Gagal baca thresholds: {e}")
    server_inference.set_thresholds(dict(DEFAULT_THRESHOLDS["server_model"]))
    log.warning(f"Thresholds file tidak ditemukan atau kosong: {path}, pakai defaults")
    return False


def create_app(model_path: str = None,
               device: str = "cuda",
               thresholds_path: str = "thresholds_v2.json",
               log_dir: str = "logs") -> FastAPI:
    """
    FastAPI app factory.
    Dipanggil oleh run_server.py atau Gunicorn.
    """
    global _logger, _thresholds_path

    app = FastAPI(
        title="ARTEMIS v2 Inference Server",
        description="YOLOv26x GPU inference untuk deteksi kebakaran/asap",
        version="2.0-topik3",
    )

    _logger = RequestLogger(log_dir=log_dir)

    # Load model jika path diberikan (bisa juga dari env var)
    mp = model_path or os.environ.get("ARTEMIS_MODEL")
    if mp:
        mp = str(Path(mp).expanduser())
        _load_thresholds(thresholds_path)
        server_inference.load_model(mp, device)

    # ── /health ───────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        import torch
        thresh    = server_inference.get_thresholds()
        from_file = thresh != DEFAULT_THRESHOLDS["server_model"]
        status    = _logger.get_status() if _logger else {}
        return {
            "status":               "ok" if server_inference.is_loaded() else "model_not_loaded",
            "model":                str(server_inference._model_path),
            "device":               server_inference._device,
            "cuda_available":       torch.cuda.is_available(),
            "uptime_s":             round(time.time() - _start_time, 1),
            "requests_ok":          status.get("total_requests", 0),
            "requests_err":         status.get("total_errors", 0),
            "thresholds":           thresh,
            "thresholds_from_file": from_file,
            "active_nodes":         status.get("active_nodes", 0),
        }

    # ── /status ───────────────────────────────────────────────────────────────
    @app.get("/status")
    async def status():
        """Per-node stats untuk monitoring RQ3."""
        if not _logger:
            return {"error": "logger not initialized"}
        return _logger.get_status()

    # ── /nodes ────────────────────────────────────────────────────────────────
    @app.get("/nodes")
    async def nodes(active_last_s: float = 300):
        """Daftar node yang aktif dalam N detik terakhir."""
        if not _logger:
            return {"nodes": {}, "count": 0}
        full   = _logger.get_status()
        now    = time.time()
        result = {
            nid: info for nid, info in full["nodes"].items()
            if info["last_seen_ago_s"] <= active_last_s
        }
        return {"nodes": result, "count": len(result)}

    # ── /infer ────────────────────────────────────────────────────────────────
    @app.post("/infer")
    async def infer(
        file:          UploadFile = File(...),
        x_node_id:     Optional[str] = Header(None, alias="X-Node-ID"),
        x_device_type: Optional[str] = Header(None, alias="X-Device-Type"),
        x_experiment_id: Optional[str] = Header(None, alias="X-Experiment-ID"),
    ):
        global _req_counter

        if not server_inference.is_loaded():
            raise HTTPException(status_code=503, detail="model not loaded")

        node_id     = x_node_id     or "unknown"
        device_type = x_device_type or "unknown"
        exp_id      = x_experiment_id

        t_total = time.time()
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="empty file")

        try:
            result = server_inference.run_inference(image_bytes)
            _req_counter += 1
            req_num  = _req_counter
            total_ms = (time.time() - t_total) * 1000

            if _logger:
                _logger.log_request(
                    request_id          = req_num,
                    node_id             = node_id,
                    device_type         = device_type,
                    experiment_id       = exp_id,
                    server_inference_ms = result["server_inference_ms"],
                    server_total_ms     = total_ms,
                    result              = result,
                )

            result["server_total_ms"] = round(total_ms, 2)
            result["node_id"]         = node_id
            result["request_id"]      = req_num
            return JSONResponse(content=result)

        except Exception as e:
            log.error(f"Inference error (node={node_id}): {e}", exc_info=True)
            if _logger:
                _logger.log_request(
                    req_num if '_req_counter' in dir() else 0,
                    node_id, device_type, exp_id, 0.0, 0.0, {}, is_error=True)
            raise HTTPException(status_code=500, detail=str(e))

    # ── /thresholds ───────────────────────────────────────────────────────────
    @app.get("/thresholds")
    async def get_thresholds():
        thresh    = server_inference.get_thresholds()
        from_file = thresh != DEFAULT_THRESHOLDS["server_model"]
        return {
            **thresh,
            "_source": _thresholds_path if from_file else "hardcoded_defaults",
        }

    @app.post("/reload_thresholds")
    async def reload_thresholds():
        ok = _load_thresholds(_thresholds_path)
        return {
            "status":     "ok" if ok else "using_defaults",
            "thresholds": server_inference.get_thresholds(),
            "from_file":  ok,
        }

    @app.post("/reload_model")
    async def reload_model():
        try:
            server_inference.load_model(
                server_inference._model_path,
                server_inference._device)
            return {"status": "reloaded",
                    "model":  str(server_inference._model_path)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
