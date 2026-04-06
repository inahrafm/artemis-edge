"""
shared/features.py
==================
Feature extraction untuk Decision Engine ARTEMIS v2.
Dipindahkan dari scripts/utils.py — tidak ada perubahan logika.
"""

from typing import Dict, List, Optional
import numpy as np

from shared.constants import (
    CLASS_FIRE, CLASS_SMOKE,
    FEATURES_PER_HISTORY_FRAME,
    expected_n_features,
)


def extract_frame_features(detections: List[Dict],
                            prev: Optional[Dict] = None) -> Dict:
    """
    Extract scalar features dari satu frame YOLO detection output.

    Args:
        detections: list of {"class_id": int, "confidence": float}
        prev:       features dari frame sebelumnya (untuk delta features)

    Returns:
        dict dengan semua keys yang dibutuhkan build_feature_vector()
    """
    fire_c  = [d["confidence"] for d in detections if d["class_id"] == CLASS_FIRE]
    smoke_c = [d["confidence"] for d in detections if d["class_id"] == CLASS_SMOKE]

    cmf = float(max(fire_c))      if fire_c  else 0.0
    caf = float(np.mean(fire_c))  if fire_c  else 0.0
    cf  = len(fire_c)
    cms = float(max(smoke_c))     if smoke_c else 0.0
    cas = float(np.mean(smoke_c)) if smoke_c else 0.0
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
    cmax_all = float(max(all_c))  if all_c          else 0.0
    cmin_all = float(min(all_c))  if all_c          else 0.0
    cstd_all = float(np.std(all_c)) if len(all_c) > 1 else 0.0
    rng_all  = cmax_all - cmin_all
    tot      = cf + cs
    sfr      = cs / tot if tot > 0 else 0.5

    return {
        "confmax_fire":     cmf,   "confavg_fire":    caf,   "count_fire":    cf,
        "d_confmax_fire":   d_cmf, "d_confavg_fire":  d_caf, "d_count_fire":  d_cf,
        "confmax_smoke":    cms,   "confavg_smoke":   cas,   "count_smoke":   cs,
        "d_confmax_smoke":  d_cms, "d_confavg_smoke": d_cas, "d_count_smoke": d_cs,
        "confmax_all":      cmax_all, "confstd_all":  cstd_all,
        "confmin_all":      cmin_all, "range_all":    rng_all,
        "smoke_fire_ratio":    sfr,
        "recent_offload_rate": 0.0,
        "_cmf_raw": cmf, "_cms_raw": cms,
    }


def build_feature_vector(window: List[Dict],
                          recent_offload_rate: float = 0.0) -> np.ndarray:
    """
    Build 156-dim feature vector dari sliding window N=10 frames.

    Structure:
      History frames 0..N-2 : 16 features each  →  16 × 9 = 144
      Delta current (N-1 vs N-2)                →  6
      Window-level global stats                  →  6
                                                    ─────
      TOTAL                                      →  156

    NOTE: Raw confmax current frame sengaja TIDAK dimasukkan (anti-circular).
    """
    N        = len(window)
    features = []

    # ── History frames (0 to N-2) ─────────────────────────────────────────────
    for t in range(N - 1):
        fi   = window[t]
        prev = window[t - 1] if t > 0 else window[t]
        features.extend([
            fi["confmax_fire"],  fi["confavg_fire"],  float(fi["count_fire"]),
            fi["confmax_smoke"], fi["confavg_smoke"], float(fi["count_smoke"]),
            fi["confmax_fire"]  - prev["confmax_fire"],
            fi["confmax_smoke"] - prev["confmax_smoke"],
            float(fi["count_fire"]  - prev["count_fire"]),
            float(fi["count_smoke"] - prev["count_smoke"]),
            fi["confmax_all"], fi["confstd_all"], fi["confmin_all"], fi["range_all"],
            fi.get("smoke_fire_ratio", 0.5),
            float(fi["count_fire"] + fi["count_smoke"]),
        ])

    # ── Delta current frame ───────────────────────────────────────────────────
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

    # ── Window-level global stats ─────────────────────────────────────────────
    fire_w  = [f["confmax_fire"]  for f in window]
    smoke_w = [f["confmax_smoke"] for f in window]
    xs      = np.arange(N, dtype=np.float32)
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
