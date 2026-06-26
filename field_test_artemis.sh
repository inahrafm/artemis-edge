#!/bin/bash
# =============================================================================
# field_test_artemis.sh
# Jalankan setelah collect_measurement.sh selesai di lokasi yang sama.
# Kirim frame nyata ke server via WireGuard 4G, ukur end-to-end latency.
#
# Tujuan: validasi tc simulation — bandingkan network_total_ms real-world
#         dengan hasil simulasi tc di kondisi yang equivalent.
#
# Usage:
#   bash field_test_artemis.sh --session SESSION_ID
#
# Contoh:
#   bash field_test_artemis.sh --session jayagiri_hutan_rendah_telkomsel_20260410_091523
#
# Output masuk ke folder sesi collect_measurement yang sama:
#   /home/pi/measurements/<SESSION_ID>/artemis_field/
# =============================================================================

set -euo pipefail

# ─── DETEKSI PERANGKAT & AUTO-ACTIVATE VIRTUALENV ────────────────────────────
HOSTNAME_NOW=$(hostname)
if [[ "$HOSTNAME_NOW" == "artemis-5" ]]; then
    VENV_PATH="/home/pi/artemis-pi5/artemis-env"
    PI_ID="pi5_field"
    MODEL_EDGE_DEFAULT="models/best.onnx"
    MODEL_TYPE_DEFAULT="onnx"
elif [[ "$HOSTNAME_NOW" == "artemis-3" ]]; then
    VENV_PATH="/home/pi/dfire-artemis-pi3/artemis-env"
    PI_ID="pi3_field"
    MODEL_EDGE_DEFAULT="models/best_float32.tflite"
    MODEL_TYPE_DEFAULT="tflite"
else
    VENV_PATH="/home/pi/artemis-pi5/artemis-env"
    PI_ID="pi_field"
    MODEL_EDGE_DEFAULT="models/best.onnx"
    MODEL_TYPE_DEFAULT="onnx"
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        source "${VENV_PATH}/bin/activate"
    else
        echo "[WARN] Virtualenv tidak ditemukan di ${VENV_PATH}, lanjut tanpa activate"
    fi
fi

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
ARTEMIS_DIR="/home/pi/artemis-v2"
SERVER_URL="http://10.99.0.2:8000"          # Homeserver via WireGuard
MODEL_EDGE="${MODEL_EDGE_DEFAULT}"
MODEL_TYPE="${MODEL_TYPE_DEFAULT}"
MODEL_DE="models/lightgbm_de_v2.pkl"        # N=10, k=1 — model official
THRESHOLDS="thresholds_v2.json"
SEQ_FIELD="sequences/sequence_list_field.json"
MEASUREMENTS_BASE="/home/pi/measurements"

# Warna
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
sep()  { echo -e "${CYAN}$(printf '─%.0s' {1..60})${NC}"; }

# ─── Parse argumen ────────────────────────────────────────────────────────────
SESSION_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) SESSION_ID="$2"; shift 2 ;;
        *) err "Unknown argument: $1" ;;
    esac
done

# Kalau tidak ada --session, tanya interaktif
if [[ -z "$SESSION_ID" ]]; then
    sep
    echo -e "${BOLD}=== ARTEMIS FIELD TEST ===${NC}"
    echo ""
    echo "Daftar sesi terbaru di ${MEASUREMENTS_BASE}:"
    ls -t "$MEASUREMENTS_BASE" 2>/dev/null | head -5 | nl -w2 -s'. '
    echo ""
    echo -n "Masukkan SESSION_ID (copy dari output collect_measurement.sh): "
    read -r SESSION_ID
fi

[[ -z "$SESSION_ID" ]] && err "SESSION_ID tidak boleh kosong."

OUTPUT_DIR="${MEASUREMENTS_BASE}/${SESSION_ID}/artemis_field"
MEASUREMENT_DIR="${MEASUREMENTS_BASE}/${SESSION_ID}"

[[ -d "$MEASUREMENT_DIR" ]] || err "Folder sesi tidak ditemukan: ${MEASUREMENT_DIR}"

mkdir -p "$OUTPUT_DIR"

# ─── Validasi setup ───────────────────────────────────────────────────────────
sep
echo -e "${BOLD}=== ARTEMIS FIELD TEST — Validasi Setup ===${NC}"
echo -e "Session : ${CYAN}${SESSION_ID}${NC}"
echo -e "Output  : ${CYAN}${OUTPUT_DIR}${NC}"
sep

cd "$ARTEMIS_DIR"

# Cek WINDOW_SIZE
WS=$(python3 -c "
import sys; sys.path.insert(0,'scripts')
from utils import WINDOW_SIZE
print(WINDOW_SIZE)
")
if [[ "$WS" != "10" ]]; then
    err "WINDOW_SIZE di utils.py = ${WS}, harus 10! Edit scripts/utils.py dulu."
fi
log "✓ WINDOW_SIZE = ${WS}"

# Cek model files
[[ -f "$MODEL_EDGE" ]]   || err "Model edge tidak ditemukan: ${MODEL_EDGE}"
[[ -f "$MODEL_DE" ]]     || err "Model DE tidak ditemukan: ${MODEL_DE}"
[[ -f "$THRESHOLDS" ]]   || err "Thresholds tidak ditemukan: ${THRESHOLDS}"
log "✓ Model files OK"

# Cek sequence list field — generate kalau belum ada
if [[ ! -f "$SEQ_FIELD" ]]; then
    warn "sequence_list_field.json belum ada. Generating..."
    python3 "$(dirname "$0")/gen_field_sequences.py" \
        --sequences sequences/sequence_list_v2.json \
        --images_dir data/full_test/images \
        --output "$SEQ_FIELD" \
        --n_sequences 10 \
        --frames_per_seq 15
    log "✓ Sequence list field generated"
else
    N_SEQ=$(python3 -c "import json; d=json.load(open('${SEQ_FIELD}')); print(d['n_sequences'])")
    N_FR=$(python3 -c "import json; d=json.load(open('${SEQ_FIELD}')); print(d['total_frames'])")
    log "✓ Sequence list field: ${N_SEQ} sequences, ${N_FR} total frames"
fi

# Cek WireGuard aktif
if ! sudo wg show wg0 &>/dev/null; then
    err "WireGuard wg0 tidak aktif! Jalankan: sudo wg-quick up wg0"
fi
log "✓ WireGuard aktif"

# Cek server health
sep
log "Mengecek server health di ${SERVER_URL}..."
HEALTH=$(curl -sf --max-time 10 "${SERVER_URL}/health" || echo "FAILED")
if [[ "$HEALTH" == "FAILED" ]]; then
    err "Server tidak bisa dijangkau di ${SERVER_URL}\nPastikan:\n  1. WireGuard aktif di homeserver\n  2. server_inference.py jalan di homeserver"
fi

# Tampilkan info server
python3 -c "
import json, sys
try:
    d = json.loads('${HEALTH}')
    print(f\"  Status  : {d.get('status','?')}\")
    print(f\"  Device  : {d.get('device','?')}\")
    print(f\"  Uptime  : {d.get('uptime_s','?')}s\")
    print(f\"  Req OK  : {d.get('requests_ok','?')}\")
except:
    print('  (tidak bisa parse response)')
" 2>/dev/null || true
log "✓ Server OK"

# ─── Estimasi waktu ───────────────────────────────────────────────────────────
sep
N_FRAMES=$(python3 -c "import json; d=json.load(open('${SEQ_FIELD}')); print(d['total_frames'])")
# server_only: ~300ms/frame di 4G, adaptive: ~500ms/frame (edge + kadang network)
EST_SERVER=$(( N_FRAMES * 300 / 1000 / 60 + 1 ))
EST_ADAPTIVE=$(( N_FRAMES * 500 / 1000 / 60 + 1 ))
EST_TOTAL=$(( EST_SERVER + EST_ADAPTIVE ))

echo -e "${BOLD}Estimasi waktu:${NC}"
echo -e "  server_only  : ~${EST_SERVER} menit  (${N_FRAMES} frames × ~300ms)"
echo -e "  adaptive     : ~${EST_ADAPTIVE} menit  (${N_FRAMES} frames × ~500ms)"
echo -e "  Total        : ~${EST_TOTAL} menit"
echo ""
echo -n "Lanjutkan? [y/N] "
read -r CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Dibatalkan."; exit 0; }

# ─── Run server_only ─────────────────────────────────────────────────────────
sep
log "STEP 1/2 — server_only (Raspi → Server via 4G, tanpa edge inference)..."
log "  Tujuan: ukur network_total_ms murni untuk validasi tc simulation"

python3 scripts/phase_f_pi_evaluation.py \
    --pi_id       "${PI_ID}_serveronly" \
    --model_edge  "$MODEL_EDGE" \
    --model_type  "$MODEL_TYPE" \
    --model_de    "$MODEL_DE" \
    --sequences   "$SEQ_FIELD" \
    --images_dir  data/full_test/images \
    --server_url  "$SERVER_URL" \
    --thresholds  "$THRESHOLDS" \
    --output_dir  "$OUTPUT_DIR" \
    --forced_offload_interval 0 \
    --methods     server_only \
    2>&1 | tee "${OUTPUT_DIR}/log_server_only.txt"

log "server_only selesai → ${OUTPUT_DIR}/results_${PI_ID}_serveronly.json"

# ─── Run adaptive ─────────────────────────────────────────────────────────────
sep
log "STEP 2/2 — adaptive cooperative (edge + selective offload via 4G)..."
log "  Tujuan: ukur end-to-end latency sistem ARTEMIS di kondisi jaringan nyata"

python3 scripts/phase_f_pi_evaluation.py \
    --pi_id       "${PI_ID}_adaptive" \
    --model_edge  "$MODEL_EDGE" \
    --model_type  "$MODEL_TYPE" \
    --model_de    "$MODEL_DE" \
    --sequences   "$SEQ_FIELD" \
    --images_dir  data/full_test/images \
    --server_url  "$SERVER_URL" \
    --thresholds  "$THRESHOLDS" \
    --output_dir  "$OUTPUT_DIR" \
    --forced_offload_interval 50 \
    --methods     adaptive \
    2>&1 | tee "${OUTPUT_DIR}/log_adaptive.txt"

log "adaptive selesai → ${OUTPUT_DIR}/results_${PI_ID}_adaptive.json"

# ─── Quick summary ────────────────────────────────────────────────────────────
sep
echo -e "${BOLD}=== QUICK SUMMARY — ARTEMIS FIELD TEST ===${NC}"

export OUTPUT_DIR_PY="$OUTPUT_DIR"
export PI_ID_PY="$PI_ID"
python3 - <<'PYEOF'
import json, os, sys

output_dir = os.environ.get("OUTPUT_DIR_PY", "")
if not output_dir:
    print("  (set OUTPUT_DIR_PY untuk summary otomatis)")
    sys.exit(0)

def load(fname):
    p = os.path.join(output_dir, fname)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

so = load(f"results_{os.environ.get('PI_ID_PY','pi_field')}_serveronly.json")
ad = load(f"results_{os.environ.get('PI_ID_PY','pi_field')}_adaptive.json")

print(f"  {'Method':<22} {'network_total_ms avg':>22} {'server_inf_ms avg':>18} {'offload%':>9}")
print(f"  {'-'*73}")

for label, data, method_key in [
    ("server_only",  so, "server_only"),
    ("adaptive",     ad, "adaptive_cooperative"),
]:
    if not data:
        print(f"  {label:<22}  (data tidak tersedia)")
        continue
    methods = data.get("methods", {})
    m = methods.get(method_key, {})
    s = m.get("summary", {})
    bd = s.get("latency_breakdown", {})

    net_avg  = bd.get("network_total_ms_offload_only", bd.get("network_total_ms", {})).get("avg", 0)
    srv_avg  = bd.get("server_inference_ms_offload_only", bd.get("server_inference_ms", {})).get("avg", 0)
    off_pct  = s.get("offload_rate", 0) * 100
    print(f"  {label:<22} {net_avg:>21.1f}ms {srv_avg:>17.1f}ms {off_pct:>8.1f}%")

# Baca ping dari collect_measurement untuk perbandingan
ping_file = os.path.join(output_dir, "../../ping_homeserver.txt")
ping_file = os.path.normpath(ping_file)
if os.path.exists(ping_file):
    with open(ping_file) as f:
        for line in f:
            if "rtt min/avg/max/mdev" in line:
                print(f"\n  Ping RTT (collect_measurement): {line.strip()}")
                break

PYEOF

# ─── Backup info ─────────────────────────────────────────────────────────────
sep
echo -e "${YELLOW}⚠ JANGAN LUPA:${NC}"
echo "  □ Backup folder artemis_field ke HP/cloud"
echo "  □ Catat apakah ada frame yang gagal (cek log_*.txt)"
echo "  □ Screenshot kondisi sinyal saat test berjalan"
sep
echo -e "Output lengkap: ${CYAN}${OUTPUT_DIR}${NC}"
