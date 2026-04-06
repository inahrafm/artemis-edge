"""
edge/decision_engine.py
========================
LightGBM Decision Engine wrapper untuk ARTEMIS v2.

Mengelola:
- Load model LightGBM dari file
- Sliding window N=10 frame dengan per-sequence reset
- Feature vector building dan prediksi LOCAL/OFFLOAD/DROP
- Recent offload rate tracking

State window di-reset setiap sequence boundary (seq_id berubah).
"""

import logging
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from shared.constants import LABEL_NAMES, WINDOW_SIZE
from shared.features import build_feature_vector

log = logging.getLogger("artemis.edge.decision_engine")


class DecisionEngine:
    """
    LightGBM-based routing Decision Engine.

    Routing decisions: LOCAL | OFFLOAD | DROP
    Window state di-reset otomatis saat seq_id berubah.
    """

    def __init__(self, model_path: str, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._model      = self._load(model_path)

        # State temporal — di-reset per sequence
        self._window          = deque(maxlen=window_size)
        self._recent_offloads = deque(maxlen=window_size)
        self._current_seq_id  = None
        self._local_frame_idx = 0

    def _load(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"DE model tidak ditemukan: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.info(f"DE model loaded: {path}")
        return model

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self,
                features: Dict,
                seq_id: Optional[str] = None,
                forced_interval: int = 50,
                global_frame_idx: int = 0) -> Tuple[str, float, bool]:
        """
        Prediksi routing decision untuk satu frame.

        Args:
            features:         output dari extract_frame_features()
            seq_id:           ID sequence saat ini (untuk window reset)
            forced_interval:  setiap N frame di-OFFLOAD paksa (0 = disabled)
            global_frame_idx: index frame global untuk forced offload

        Returns:
            (decision, de_ms, is_warmup)
            decision:  'LOCAL' | 'OFFLOAD' | 'DROP'
            de_ms:     waktu prediksi DE dalam ms
            is_warmup: True jika frame masih dalam warmup period
        """
        # Reset window jika sequence berubah
        if seq_id is not None and seq_id != self._current_seq_id:
            self._reset_state(seq_id)

        is_warmup = self._local_frame_idx < self.window_size - 1

        # Warmup: isi window dengan frame dummy (fitur nol) di awal
        if len(self._window) == 0:
            for _ in range(self.window_size - 1):
                self._window.append(features)

        self._window.append(features)

        ror = (sum(self._recent_offloads) / len(self._recent_offloads)
               if self._recent_offloads else 0.0)

        # Build feature vector dan prediksi
        t0     = time.perf_counter()
        fv     = build_feature_vector(list(self._window), ror)
        label  = LABEL_NAMES[int(self._model.predict(fv.reshape(1, -1))[0])]
        de_ms  = (time.perf_counter() - t0) * 1000

        # Forced OFFLOAD override
        is_forced = (forced_interval > 0
                     and global_frame_idx % forced_interval == 0)
        if is_forced:
            label = "OFFLOAD"

        # Update state
        self._recent_offloads.append(1 if label == "OFFLOAD" else 0)
        self._local_frame_idx += 1

        return label, round(de_ms, 4), is_warmup

    def reset(self, seq_id: Optional[str] = None):
        """Manual reset — untuk awal sequence baru atau re-use instance."""
        self._reset_state(seq_id)

    @property
    def recent_offload_rate(self) -> float:
        if not self._recent_offloads:
            return 0.0
        return sum(self._recent_offloads) / len(self._recent_offloads)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_state(self, seq_id: Optional[str]):
        self._window.clear()
        self._recent_offloads.clear()
        self._current_seq_id  = seq_id
        self._local_frame_idx = 0
        if seq_id:
            log.debug(f"Window reset untuk seq_id={seq_id}")
