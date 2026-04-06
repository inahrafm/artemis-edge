"""
shared/constants.py
===================
Konstanta global ARTEMIS v2. Digunakan oleh semua modul edge dan server.
Dipindahkan dari scripts/utils.py untuk arsitektur modular Topik 3.
"""

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_SMOKE = 0
CLASS_FIRE  = 1
CLASS_NAMES = {CLASS_SMOKE: "smoke", CLASS_FIRE: "fire"}

# ── Model input ───────────────────────────────────────────────────────────────
IMG_SIZE = 640

# ── Decision Engine ───────────────────────────────────────────────────────────
WINDOW_SIZE = 10          # N=10 validated empirically (experiment_temporal_anchoring.py)
LABEL_MAP   = {"LOCAL": 0, "OFFLOAD": 1, "DROP": 2}
LABEL_NAMES = ["LOCAL", "OFFLOAD", "DROP"]

# ── Feature vector dimensionality ─────────────────────────────────────────────
# Config C (N=10): 16 × (N-1) + 6 delta + 6 global = 156
# NOTE: 16 features per history frame, NOT 20 as in earlier thesis drafts.
FEATURES_PER_HISTORY_FRAME = 16
FEATURES_DELTA_CURRENT     = 6
FEATURES_GLOBAL            = 6

def expected_n_features(window_size: int) -> int:
    """Return expected feature vector length for given window size."""
    return (FEATURES_PER_HISTORY_FRAME * (window_size - 1)
            + FEATURES_DELTA_CURRENT
            + FEATURES_GLOBAL)

# ── Default thresholds (fallback jika thresholds_v2.json tidak ada) ───────────
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

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED  = 42
SEEDS = [42, 123, 7, 2024, 99]
