"""
edge/offloader.py
=================
HTTP client untuk frame offload dari edge ke server ARTEMIS v2.

Mengelola:
- POST multipart/form-data ke /infer endpoint
- X-Node-ID dan X-Device-Type headers untuk server-side tracking (RQ3)
- Timeout yang configurable per kondisi jaringan
- Error tracking: timeout, connection refused, dll
- Breakdown latensi: network_total, server_inference, network_overhead
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import requests

log = logging.getLogger("artemis.edge.offloader")


class Offloader:
    """
    HTTP offloader untuk mengirim frame ke inference server.

    Satu instance per session eksperimen — session di-reuse untuk
    connection pooling dan header persistence.
    """

    def __init__(self,
                 server_url: str,
                 node_id: str = "unknown",
                 device_type: str = "unknown",
                 timeout: float = 15.0,
                 experiment_id: str = None):
        """
        Args:
            server_url:    base URL server, misal 'http://10.99.0.2:8000'
            node_id:       identifier node untuk X-Node-ID header (RQ3 tracking)
            device_type:   tipe device untuk X-Device-Type header
            timeout:       request timeout dalam detik
            experiment_id: label eksperimen untuk X-Experiment-ID header
        """
        self.server_url    = server_url.rstrip("/")
        self.timeout       = timeout
        self._session      = requests.Session()
        self._n_success    = 0
        self._n_errors     = 0
        self._error_types  = {"timeout": 0, "connection": 0, "other": 0}

        # Set headers untuk node tracking — dikirim di setiap request
        self._session.headers.update({
            "X-Node-ID":       node_id,
            "X-Device-Type":   device_type,
        })
        if experiment_id:
            self._session.headers["X-Experiment-ID"] = experiment_id

        log.info(f"Offloader init: {server_url} | node={node_id} | "
                 f"device={device_type} | timeout={timeout}s")

    # ── Public interface ──────────────────────────────────────────────────────

    def offload(self, image_path: str) -> Tuple[Dict, Dict]:
        """
        Kirim frame ke server dan ukur breakdown latensi network.

        Args:
            image_path: path ke file JPEG yang akan dikirim

        Returns:
            (server_response, latency_breakdown)

            server_response: dict dari JSON response server
                { confmax_fire, confmax_smoke, decision,
                  server_inference_ms, server_total_ms, ... }
                Jika error: { decision: "NONE", network_error: True, error: str }

            latency_breakdown: {
                "network_total_ms":    float,  # total round-trip
                "server_inference_ms": float,  # GPU inference di server
                "network_overhead_ms": float,  # network_total - server_inference
            }
        """
        t0 = time.perf_counter()
        try:
            with open(image_path, "rb") as f:
                resp = self._session.post(
                    f"{self.server_url}/infer",
                    files={"file": (Path(image_path).name, f, "image/jpeg")},
                    timeout=self.timeout,
                )
            data = resp.json()
            self._n_success += 1

        except requests.Timeout:
            self._n_errors  += 1
            self._error_types["timeout"] += 1
            log.warning(f"Timeout ({self.timeout}s): {image_path}")
            data = {"decision": "NONE", "server_inference_ms": 0.0,
                    "server_total_ms": 0.0, "confmax_fire": 0.0,
                    "confmax_smoke": 0.0, "network_error": True,
                    "error": "timeout"}

        except requests.ConnectionError as e:
            self._n_errors  += 1
            self._error_types["connection"] += 1
            log.warning(f"Connection error: {e}")
            data = {"decision": "NONE", "server_inference_ms": 0.0,
                    "server_total_ms": 0.0, "confmax_fire": 0.0,
                    "confmax_smoke": 0.0, "network_error": True,
                    "error": f"connection_error: {e}"}

        except Exception as e:
            self._n_errors  += 1
            self._error_types["other"] += 1
            log.warning(f"Request error: {e}")
            data = {"decision": "NONE", "server_inference_ms": 0.0,
                    "server_total_ms": 0.0, "confmax_fire": 0.0,
                    "confmax_smoke": 0.0, "network_error": True,
                    "error": str(e)}

        network_total_ms    = (time.perf_counter() - t0) * 1000
        server_inference_ms = float(
            data.get("server_inference_ms") or
            data.get("server_total_ms") or 0.0
        )
        network_overhead_ms = max(0.0, network_total_ms - server_inference_ms)

        return data, {
            "network_total_ms":    round(network_total_ms,    3),
            "server_inference_ms": round(server_inference_ms, 3),
            "network_overhead_ms": round(network_overhead_ms, 3),
        }

    def health_check(self, timeout: float = 5.0) -> bool:
        """Verifikasi server bisa dijangkau. Return True jika OK."""
        try:
            resp = self._session.get(
                f"{self.server_url}/health", timeout=timeout)
            data = resp.json()
            if data.get("status") == "ok":
                log.info(f"Server OK — model: "
                         f"{Path(data.get('model','?')).name}, "
                         f"device: {data.get('device','?')}, "
                         f"uptime: {data.get('uptime_s',0):.0f}s")
                return True
            return False
        except Exception as e:
            log.warning(f"Health check gagal: {e}")
            return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        total = self._n_success + self._n_errors
        return {
            "n_success":   self._n_success,
            "n_errors":    self._n_errors,
            "error_rate":  round(self._n_errors / total, 4) if total > 0 else 0.0,
            "error_types": dict(self._error_types),
        }

    def close(self):
        self._session.close()
