"""
server/logger.py
================
Per-request structured logging untuk ARTEMIS v2 server.
Mencatat setiap request ke CSV untuk analisis RQ3 (multi-node degradation).
"""

import csv
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("artemis.server.logger")


class RequestLogger:
    """
    Thread-safe logger untuk per-request metrics.

    Mencatat ke:
    - In-memory per-node stats (untuk /status endpoint)
    - CSV file (untuk analisis RQ3 offline)
    """

    CSV_FIELDS = [
        "timestamp", "request_id", "node_id", "device_type",
        "experiment_id", "server_inference_ms", "server_total_ms",
        "decision", "confmax_fire", "confmax_smoke", "is_error",
    ]

    def __init__(self, log_dir: str = "logs"):
        self._lock       = threading.Lock()
        self._log_dir    = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path   = None
        self._csv_file   = None
        self._csv_writer = None
        self._node_stats = defaultdict(lambda: {
            "count": 0, "errors": 0,
            "latencies_ms": deque(maxlen=1000),
            "last_seen": 0.0,
            "device_type": "unknown",
            "experiment_id": None,
        })
        self._total_requests = 0
        self._total_errors   = 0
        self._start_time     = time.time()
        self._init_csv()

    def _init_csv(self):
        ts             = time.strftime("%Y%m%d_%H%M%S")
        self._csv_path = self._log_dir / f"server_requests_{ts}.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()
        log.info(f"Request log: {self._csv_path}")

    def log_request(self,
                    request_id: int,
                    node_id: str,
                    device_type: str,
                    experiment_id: Optional[str],
                    server_inference_ms: float,
                    server_total_ms: float,
                    result: Dict,
                    is_error: bool = False):
        """Catat satu request ke memory dan CSV."""
        ts = time.time()
        with self._lock:
            self._total_requests += 1
            if is_error:
                self._total_errors += 1

            # Update per-node stats
            s = self._node_stats[node_id]
            s["count"]       += 1
            s["last_seen"]    = ts
            s["device_type"]  = device_type
            if experiment_id:
                s["experiment_id"] = experiment_id
            if not is_error:
                s["latencies_ms"].append(server_total_ms)
            if is_error:
                s["errors"] += 1

            # Write CSV
            if self._csv_writer:
                self._csv_writer.writerow({
                    "timestamp":           round(ts, 3),
                    "request_id":          request_id,
                    "node_id":             node_id,
                    "device_type":         device_type,
                    "experiment_id":       experiment_id or "",
                    "server_inference_ms": round(server_inference_ms, 3),
                    "server_total_ms":     round(server_total_ms, 3),
                    "decision":            result.get("decision", "NONE"),
                    "confmax_fire":        result.get("confmax_fire", 0.0),
                    "confmax_smoke":       result.get("confmax_smoke", 0.0),
                    "is_error":            int(is_error),
                })
                self._csv_file.flush()

    def get_status(self) -> Dict:
        """Ringkasan status server + per-node stats untuk /status endpoint."""
        now = time.time()
        with self._lock:
            nodes = {}
            for nid, s in self._node_stats.items():
                lats = list(s["latencies_ms"])
                nodes[nid] = {
                    "device_type":    s["device_type"],
                    "experiment_id":  s["experiment_id"],
                    "request_count":  s["count"],
                    "error_count":    s["errors"],
                    "last_seen_ago_s": round(now - s["last_seen"], 1),
                    "latency_avg_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
                    "latency_p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else 0.0,
                    "latency_p99_ms": round(float(np.percentile(lats, 99)), 2) if lats else 0.0,
                }
            return {
                "uptime_s":       round(now - self._start_time, 1),
                "total_requests": self._total_requests,
                "total_errors":   self._total_errors,
                "active_nodes":   len(self._node_stats),
                "nodes":          nodes,
                "csv_log":        str(self._csv_path),
            }

    def close(self):
        if self._csv_file:
            self._csv_file.close()
