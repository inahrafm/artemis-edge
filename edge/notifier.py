"""
edge/notifier.py — ARTEMIS v2 Alarm Notifier
Kirim notifikasi alarm ke VPS alarm server, ukur notification latency.
"""
import logging, time
from typing import Optional, Dict
import requests

log = logging.getLogger("artemis.edge.notifier")

class AlarmNotifier:
    def __init__(self, alarm_url: str, node_id: str = "unknown",
                 device_type: str = "unknown", timeout: float = 5.0,
                 enabled: bool = True):
        self.alarm_url   = alarm_url
        self.node_id     = node_id
        self.device_type = device_type
        self.timeout     = min(timeout, 1.5)
        self.enabled     = enabled
        self._session    = requests.Session()
        self._n_sent     = 0
        self._n_failed   = 0
        log.info(f"AlarmNotifier: {alarm_url} | enabled={enabled}")

    def notify(self, alarm_type: str, method: str = "", location: str = "",
               experiment_id: str = "", seq_id: str = "", frame_index: int = 0,
               confmax_fire: float = 0.0, confmax_smoke: float = 0.0,
               alarm_decision_ms: float = 0.0, frame_total_ms: float = 0.0,
               gps: str = "") -> Optional[Dict]:
        if not self.enabled:
            return None
        t_sent = time.time()
        try:
            resp = self._session.post(self.alarm_url, stream=False, json={
                "node_id":           self.node_id,
                "device_type":       self.device_type,
                "alarm_type":        alarm_type,
                "method":            method,
                "location":          location,
                "gps":               gps,
                "experiment_id":     experiment_id,
                "seq_id":            seq_id,
                "frame_index":       frame_index,
                "confmax_fire":      round(confmax_fire,  4),
                "confmax_smoke":     round(confmax_smoke, 4),
                "alarm_decision_ms": round(alarm_decision_ms, 3),
                "frame_total_ms":    round(frame_total_ms,    3),
                "sent_at":           t_sent,
                "t_alarm_sent_ms":   t_sent * 1000,
            }, timeout=self.timeout)
            data = resp.json()
            self._n_sent += 1
            log.info(f"Alarm sent [{alarm_type}] notif={data.get('notification_ms',0):.1f}ms")
            return data
        except Exception as e:
            self._n_failed += 1
            log.warning(f"Alarm notification failed: {e}")
            return None

    @property
    def stats(self) -> Dict:
        return {"n_sent": self._n_sent, "n_failed": self._n_failed,
                "enabled": self.enabled}

    def close(self):
        self._session.close()
