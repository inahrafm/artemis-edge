"""
utils.py — Shared constants, logging setup, and utilities for Artemis v2.
Used by all Phase A–F scripts.

FEATURE DIMENSIONALITY CLARIFICATION (fixes audit Blocker 4):
  Config C for window size N=10:
    History frames  : 16 features × (N-1) = 16 × 9 = 144
    Delta (current) : 6 features
    Global stats    : 6 features
    TOTAL           : 156 features
  The 16 features per history frame are:
    [0]  confmax_fire      [1]  confavg_fire      [2]  count_fire
    [3]  confmax_smoke     [4]  confavg_smoke     [5]  count_smoke
    [6]  Δconfmax_fire     [7]  Δconfmax_smoke    [8]  Δcount_fire
    [9]  Δcount_smoke      [10] confmax_all       [11] confstd_all
    [12] confmin_all       [13] range_all         [14] smoke_fire_ratio
    [15] total_count
  NOTE: The thesis description stating "20 features per frame" is incorrect.
        The correct number is 16 per history frame, as implemented here.
"""

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Global constants ──────────────────────────────────────────────────────────
CLASS_SMOKE = 0
CLASS_FIRE  = 1
CLASS_NAMES = {CLASS_SMOKE: "smoke", CLASS_FIRE: "fire"}
IMG_SIZE    = 640
SEED        = 42
# WINDOW_SIZE=10: motivated by empirical evidence that temporal context in
# the range N=5–10 outperforms single-frame baselines for lightweight models
# (Quan et al. 2025, arXiv; Yurchuk & Semenchenko 2025, Advanced IT).
# The specific value N=10 is validated empirically for ARTEMIS by
# experiment_temporal_anchoring.py, which compares frames-to-alarm across
# N ∈ {3,5,8,10,15} on gradual escalation sequences.
WINDOW_SIZE = 10
#WINDOW_SIZE = 3
LABEL_MAP   = {"LOCAL": 0, "OFFLOAD": 1, "DROP": 2}
LABEL_NAMES = ["LOCAL", "OFFLOAD", "DROP"]
SEEDS       = [42, 123, 7, 2024, 99]

# Features per history frame (16, NOT 20 as incorrectly stated in earlier thesis drafts)
FEATURES_PER_HISTORY_FRAME = 16
FEATURES_DELTA_CURRENT     = 6
FEATURES_GLOBAL            = 6

def expected_n_features(window_size: int) -> int:
    """Return expected feature vector length for given window size."""
    return (FEATURES_PER_HISTORY_FRAME * (window_size - 1)
            + FEATURES_DELTA_CURRENT
            + FEATURES_GLOBAL)

# Default thresholds (edge model, will be overridden by thresholds_v2.json)
DEFAULT_THRESHOLDS = {
    "edge_model": {
        "fire_local":  0.695, "fire_drop":   0.151,
        "smoke_local": 0.797, "smoke_drop":  0.128,
    },
    "server_model": {
        "fire_local":  0.28,  "fire_drop":   0.05,
        "smoke_local": 0.35,  "smoke_drop":  0.04,
    }
}


# ── Logging setup ─────────────────────────────────────────────────────────────
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ── Threshold loading ─────────────────────────────────────────────────────────
def load_thresholds(path: str, model_type: str = "edge_model") -> Dict[str, float]:
    """Load calibrated thresholds from thresholds_v2.json."""
    p = Path(path)
    if not p.exists():
        logging.warning(f"thresholds file not found: {path}, using defaults")
        return DEFAULT_THRESHOLDS[model_type]
    with open(p) as f:
        data = json.load(f)
    return data.get(model_type, DEFAULT_THRESHOLDS[model_type])


# ── Statistical helpers ───────────────────────────────────────────────────────
def wilson_ci(successes: int, total: int,
               confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Used for DROP precision/recall where n may be small (prevents the
    symmetric Wald interval from producing [0,0] or [1,1] on extreme counts).

    Args:
        successes : number of positive outcomes
        total     : total number of trials
        confidence: confidence level (default 0.95 → 95% CI)

    Returns:
        (lower, upper) confidence interval bounds
    """
    if total == 0:
        return (0.0, 0.0)
    from scipy import stats as scipy_stats
    z   = scipy_stats.norm.ppf((1 + confidence) / 2)
    p   = successes / total
    n   = total
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def wilson_ci_str(successes: int, total: int,
                   confidence: float = 0.95) -> str:
    """Return formatted Wilson CI string, e.g. '0.857 [0.607, 0.970]'."""
    if total == 0:
        return "N/A (n=0)"
    lo, hi = wilson_ci(successes, total, confidence)
    p = successes / total
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]  (n={total})"


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_frame_features(detections: List[Dict],
                            prev: Optional[Dict] = None) -> Dict:
    """
    Extract scalar features from one frame's YOLO detection outputs.

    Returns a dict with keys that match what build_feature_vector() expects.
    The 'd_*' delta fields are stored here but build_feature_vector()
    recomputes them internally from the window — they are used only for
    reference / debugging.
    """
    fire_c  = [d["confidence"] for d in detections if d["class_id"] == CLASS_FIRE]
    smoke_c = [d["confidence"] for d in detections if d["class_id"] == CLASS_SMOKE]

    cmf = float(max(fire_c))        if fire_c  else 0.0
    caf = float(np.mean(fire_c))    if fire_c  else 0.0
    cf  = len(fire_c)
    cms = float(max(smoke_c))       if smoke_c else 0.0
    cas = float(np.mean(smoke_c))   if smoke_c else 0.0
    cs  = len(smoke_c)

    if prev:
        d_cmf = cmf - prev.get("confmax_fire",  0.0)
        d_caf = caf - prev.get("confavg_fire",  0.0)
        d_cf  = float(cf - prev.get("count_fire",  0))
        d_cms = cms - prev.get("confmax_smoke", 0.0)
        d_cas = cas - prev.get("confavg_smoke", 0.0)
        d_cs  = float(cs - prev.get("count_smoke", 0))
    else:
        d_cmf = d_caf = d_cf = d_cms = d_cas = d_cs = 0.0

    all_c    = [c for c in [cmf, cms] if c > 0]
    cmax_all = float(max(all_c)) if all_c else 0.0
    cmin_all = float(min(all_c)) if all_c else 0.0
    cstd_all = float(np.std(all_c)) if len(all_c) > 1 else 0.0
    rng_all  = cmax_all - cmin_all
    tot      = cf + cs
    # Key name: smoke_fire_ratio (consistent with Frame.to_features())
    sfr      = cs / tot if tot > 0 else 0.5

    return {
        "confmax_fire":    cmf, "confavg_fire":    caf, "count_fire":    cf,
        "d_confmax_fire":  d_cmf, "d_confavg_fire": d_caf, "d_count_fire": d_cf,
        "confmax_smoke":   cms, "confavg_smoke":   cas, "count_smoke":   cs,
        "d_confmax_smoke": d_cms, "d_confavg_smoke": d_cas, "d_count_smoke": d_cs,
        "confmax_all":     cmax_all, "confstd_all": cstd_all,
        "confmin_all":     cmin_all, "range_all":   rng_all,
        "smoke_fire_ratio":    sfr,   # consistent key name
        "recent_offload_rate": 0.0,   # updated externally before build_feature_vector
        "_cmf_raw": cmf, "_cms_raw": cms,
    }


def build_feature_vector(window: List[Dict],
                          recent_offload_rate: float = 0.0) -> np.ndarray:
    """
    Build feature vector from sliding window of N frames.

    Structure (for N=10, total=156 features):
      History frames 0..N-2: 16 features each  →  16 × 9 = 144
      Delta for current frame (N-1 vs N-2)     →  6
      Window-level global statistics            →  6
                                                   ─────
      TOTAL                                     →  156

    NOTE: Raw confmax of the current frame is intentionally EXCLUDED
    (anti-circular design). Only deltas relative to the previous frame
    are included for the current frame, reducing feature-label correlation
    from 0.90 to 0.22.

    The 'smoke_fire_ratio' key must exist in each window dict. Both
    extract_frame_features() and Frame.to_features() provide this key.
    """
    N        = len(window)
    features = []

    # ── History frames (0 to N-2): 16 features each ──────────────────────────
    for t in range(N - 1):
        fi   = window[t]
        prev = window[t - 1] if t > 0 else window[t]   # t=0: self-delta = 0
        features.extend([
            # Per-class base (6)
            fi["confmax_fire"],  fi["confavg_fire"],  float(fi["count_fire"]),
            fi["confmax_smoke"], fi["confavg_smoke"], float(fi["count_smoke"]),
            # Per-class delta (4)
            fi["confmax_fire"]  - prev["confmax_fire"],
            fi["confmax_smoke"] - prev["confmax_smoke"],
            float(fi["count_fire"]  - prev["count_fire"]),
            float(fi["count_smoke"] - prev["count_smoke"]),
            # Aggregate (4)
            fi["confmax_all"], fi["confstd_all"], fi["confmin_all"], fi["range_all"],
            # Temporal ratio + total count (2)
            fi.get("smoke_fire_ratio", 0.5),
            float(fi["count_fire"] + fi["count_smoke"]),
        ])  # 16 per frame

    # ── Delta features for current frame (N-1 vs N-2): 6 ─────────────────────
    cur  = window[-1]
    prev = window[-2] if N >= 2 else window[-1]
    features.extend([
        cur["confmax_fire"]  - prev["confmax_fire"],
        cur["confmax_smoke"] - prev["confmax_smoke"],
        float(cur["count_fire"]  - prev["count_fire"]),
        float(cur["count_smoke"] - prev["count_smoke"]),
        cur["confavg_fire"]  - prev["confavg_fire"],
        cur["confavg_smoke"] - prev["confavg_smoke"],
    ])

    # ── Window-level global statistics: 6 ────────────────────────────────────
    fire_w  = [f["confmax_fire"]  for f in window]
    smoke_w = [f["confmax_smoke"] for f in window]
    xs = np.arange(N, dtype=np.float32)
    trend_f = float(np.polyfit(xs, fire_w,  1)[0]) if N > 1 else 0.0
    trend_s = float(np.polyfit(xs, smoke_w, 1)[0]) if N > 1 else 0.0
    features.extend([
        float(np.std(fire_w)),
        float(np.std(smoke_w)),
        trend_f,
        trend_s,
        float(np.mean([f.get("smoke_fire_ratio", 0.5) for f in window])),
        recent_offload_rate,
    ])

    fv = np.array(features, dtype=np.float32)
    assert len(fv) == expected_n_features(N), (
        f"Feature vector length mismatch: got {len(fv)}, "
        f"expected {expected_n_features(N)} for N={N}"
    )
    return fv


# ── Timer context manager ─────────────────────────────────────────────────────
class Timer:
    def __init__(self): self.elapsed_ms = 0.0
    def __enter__(self): self._t = time.perf_counter(); return self
    def __exit__(self, *_): self.elapsed_ms = (time.perf_counter() - self._t) * 1000
