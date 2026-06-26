#!/bin/bash
source /home/pi/artemis-pi5/artemis-env/bin/activate 2>/dev/null || true
# =============================================================================
# field_session.sh — ARTEMIS v2 Topik 3
# Integrated Field Session: Network Measurement + System Evaluation
# + Telegram notifications
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTEMIS_DIR="${SCRIPT_DIR}"
VPS_IP="10.99.0.1"
HOME_IP="10.99.0.2"
SERVER_IPERF_PORT=5201
BURST_PORT=9999
BURST_PAYLOAD_KB=150
BURST_REPEAT=200
BURST_INTERVAL=2
OUTPUT_BASE="/home/pi/artemis-v2/measurements"
RESULTS_DIR="${ARTEMIS_DIR}/results"
BASELINE_VPS_AVG_MS=21.44
BASELINE_HOME_AVG_MS=41.13
BASELINE_VPS_RELAY_MS=19.69
EDGE_METHODS="all"
EDGE_MODE="sequences"

# Telegram
TG_BOT_TOKEN="${TG_BOT_TOKEN:-7764464886:AAGJIA4yli0N3dhHCnGk9VNCf1KO4U_sOMQ}"
TG_CHAT_ID="${TG_CHAT_ID:-1046756964}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()    { echo -e "${RED}[ERROR]${NC} $1"; tg_notify "❌ ERROR di ${SESSION_ID:-unknown}: $1"; exit 1; }
sep()    { echo -e "${CYAN}$(printf '─%.0s' {1..65})${NC}"; }
header() { sep; echo -e "${BOLD}$1${NC}"; sep; }

# ── Telegram ──────────────────────────────────────────────────────────────────
tg_notify() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TG_CHAT_ID}" \
        -d text="${msg}" \
        --max-time 10 > /dev/null 2>&1 || true
}

tg_notify_result() {
    local method="$1" avg="$2" p95="$3" offload="$4" err="$5"
    tg_notify "📊 *${method}*
- Avg: ${avg}ms | P95: ${p95}ms
- Offload: ${offload}% | NetErr: ${err}"
}

# ── Parse mode ────────────────────────────────────────────────────────────────
EVAL_ONLY=false
if [[ "${1:-}" == "--eval-only" ]]; then
    EVAL_ONLY=true
    if [[ -z "${SESSION_ID:-}" ]]; then
        err "Mode --eval-only membutuhkan SESSION_ID env var"
    fi
fi

# ── Cek dependencies ──────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in ping mtr iperf3 python3 curl; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    [[ ! -f "${SCRIPT_DIR}/burst_latency.py" ]] && \
        [[ ! -f "${ARTEMIS_DIR}/scripts/burst_latency.py" ]] && \
        missing+=("burst_latency.py")
    [[ ! -f "${ARTEMIS_DIR}/scripts/run_edge.py" ]] && \
        missing+=("scripts/run_edge.py")
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Dependency tidak ditemukan: ${missing[*]}"
    fi
    if ! sudo wg show wg0 &>/dev/null; then
        err "WireGuard wg0 tidak aktif! Jalankan: sudo wg-quick up wg0"
    fi
    log "✓ Semua dependency OK, WireGuard aktif"
}

# ── Input metadata ────────────────────────────────────────────────────────────
collect_metadata() {
    header "=== ARTEMIS FIELD SESSION — Input Metadata ==="

    if [[ -n "${LOCATION:-}" ]]; then
        LOCATION_NAME=$(echo "$LOCATION" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
        log "Mode non-interaktif: LOCATION=${LOCATION_NAME}"
    else
        echo -n "Nama lokasi (misal: jayagiri_hutan_rendah): "
        read -r LOCATION_NAME
        LOCATION_NAME=$(echo "$LOCATION_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    fi

    if [[ -n "${CATEGORY:-}" ]]; then CAT_NUM="${CATEGORY}"
    else
        echo "Kategori lokasi:"
        echo "  1) urban_dense        4) hutan_menengah"
        echo "  2) suburban_baseline  5) hutan_tinggi"
        echo "  3) hutan_rendah       6) rural_hutan"
        echo -n "Pilih (1-6): "; read -r CAT_NUM
    fi
    case $CAT_NUM in
        1) CATEGORY_NAME="urban_dense" ;;
        2) CATEGORY_NAME="suburban_baseline" ;;
        3) CATEGORY_NAME="hutan_rendah" ;;
        4) CATEGORY_NAME="hutan_menengah" ;;
        5) CATEGORY_NAME="hutan_tinggi" ;;
        6) CATEGORY_NAME="rural_hutan" ;;
        *) CATEGORY_NAME="unknown" ;;
    esac

    GPS_COORD="${GPS:-}"
    [[ -z "$GPS_COORD" ]] && { echo -n "Koordinat GPS (lat,lon): "; read -r GPS_COORD; }
    ELEVATION="${ELEVATION:-}"
    [[ -z "$ELEVATION" ]] && { echo -n "Elevasi (meter): "; read -r ELEVATION; }
    OPERATOR="${OPERATOR:-}"
    [[ -z "$OPERATOR" ]] && { echo -n "Operator (telkomsel/xl/indosat): "; read -r OPERATOR; }
    OPERATOR=$(echo "$OPERATOR" | tr '[:upper:]' '[:lower:]')
    RSRP="${RSRP:-}"
    [[ -z "$RSRP" ]] && { echo -n "RSRP (dBm, misal: -85): "; read -r RSRP; }
    LTE_BAND="${BAND:-}"
    [[ -z "$LTE_BAND" ]] && { echo -n "Band LTE (misal: B3): "; read -r LTE_BAND; }
    ENB_ID="${ENB:-}"
    [[ -z "$ENB_ID" ]] && { echo -n "eNB ID: "; read -r ENB_ID; }
    WEATHER="${WEATHER:-}"
    [[ -z "$WEATHER" ]] && { echo -n "Cuaca (cerah/berawan/hujan): "; read -r WEATHER; }
    SIGNAL_BARS="${BARS:-}"
    [[ -z "$SIGNAL_BARS" ]] && { echo -n "Sinyal bar (1-5): "; read -r SIGNAL_BARS; }

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SESSION_ID=$(echo "${LOCATION_NAME}_${OPERATOR}_${TIMESTAMP}" \
                 | tr ' ' '_' | tr -s '_' | tr -cd '[:alnum:]_-')
    MEAS_DIR="${OUTPUT_BASE}/${SESSION_ID}"
    mkdir -p "$MEAS_DIR"

    cat > "${MEAS_DIR}/metadata.txt" << METAEOF
=== ARTEMIS FIELD SESSION METADATA ===
timestamp              : $(date '+%Y-%m-%d %H:%M:%S')
session_id             : ${SESSION_ID}
location_name          : ${LOCATION_NAME}
category               : ${CATEGORY_NAME}
gps_coord              : ${GPS_COORD}
elevation_m            : ${ELEVATION}
operator               : ${OPERATOR}
rsrp_dbm               : ${RSRP}
lte_band               : ${LTE_BAND}
enb_id                 : ${ENB_ID}
weather                : ${WEATHER}
signal_bars            : ${SIGNAL_BARS}
raspi_hostname         : $(hostname)
raspi_os               : $(grep PRETTY_NAME /etc/os-release | cut -d'"' -f2)

=== ARSITEKTUR SERVER ===
vps_relay_ip           : ${VPS_IP}
homeserver_gpu_ip      : ${HOME_IP}
infra_note             : Raspi → VPS relay → Homeserver GPU backend

=== BASELINE LAB (terkontrol) ===
baseline_vps_avg_ms    : ${BASELINE_VPS_AVG_MS}
baseline_home_avg_ms   : ${BASELINE_HOME_AVG_MS}
baseline_vps_relay_ms  : ${BASELINE_VPS_RELAY_MS}

=== SESSION CONFIG ===
phase1_network_meas    : ping 500x, mtr 200 cycles, iperf3 60s up+down, burst 200x
phase2_system_eval     : run_edge.py methods=${EDGE_METHODS} mode=${EDGE_MODE}
METAEOF

    log "Session ID: ${SESSION_ID}"
    log "Metadata → ${MEAS_DIR}/metadata.txt"

    tg_notify "🚀 *ARTEMIS Session Dimulai*
📍 Lokasi: ${LOCATION_NAME}
📡 Operator: ${OPERATOR}
📶 RSRP: ${RSRP} dBm | Band: ${LTE_BAND}
🆔 Session: \`${SESSION_ID}\`"
}

# ── Cek koneksi ───────────────────────────────────────────────────────────────
check_connectivity() {
    sep
    log "Mengecek koneksi WireGuard..."
    ping -c 2 -W 5 "$VPS_IP" &>/dev/null && \
        log "✓ VPS relay (${VPS_IP}) OK" || \
        err "VPS tidak dapat dijangkau. Cek: sudo wg show"
    ping -c 2 -W 5 "$HOME_IP" &>/dev/null && \
        log "✓ Homeserver GPU (${HOME_IP}) OK" || \
        err "Homeserver tidak dapat dijangkau."

    if python3 -c "
import urllib.request, sys, json
try:
    r = urllib.request.urlopen('http://${HOME_IP}:8000/health', timeout=5)
    d = json.loads(r.read())
    print(f'  ARTEMIS server OK — device: {d.get(\"device\",\"?\")}')
    sys.exit(0)
except Exception as e:
    print(f'  ARTEMIS server tidak bisa dijangkau: {e}')
    sys.exit(1)
" 2>&1; then
        log "✓ ARTEMIS inference server OK"
    else
        warn "ARTEMIS server tidak tersedia"
        tg_notify "⚠️ *WARNING* — ARTEMIS inference server tidak respond di ${SESSION_ID}"
    fi
}

# ── Phase 1: Network Measurement ──────────────────────────────────────────────
run_phase1() {
    header "=== PHASE 1: NETWORK MEASUREMENT ==="
    echo -e "  Estimasi waktu: ~25 menit"
    tg_notify "📡 *Phase 1 dimulai* — Network Measurement
📍 ${LOCATION_NAME} | ${OPERATOR}
⏱ Estimasi: ~25 menit"

    local BURST_SCRIPT
    if [[ -f "${SCRIPT_DIR}/burst_latency.py" ]]; then
        BURST_SCRIPT="${SCRIPT_DIR}/burst_latency.py"
    else
        BURST_SCRIPT="${ARTEMIS_DIR}/scripts/burst_latency.py"
    fi

    log "  [1/5] Ping 500x → VPS relay (${VPS_IP})..."
    ping -c 500 -i 0.2 "$VPS_IP" 2>&1 | tee "${MEAS_DIR}/ping_vps.txt"

    log "  [2/5] Ping 500x → Homeserver (${HOME_IP})..."
    ping -c 500 -i 0.2 "$HOME_IP" 2>&1 | tee "${MEAS_DIR}/ping_homeserver.txt"

    log "  [3/5] MTR 200 cycles → Homeserver..."
    mtr --report --report-cycles 200 --no-dns "$HOME_IP" 2>&1 \
        | tee "${MEAS_DIR}/mtr_homeserver.txt"

    log "  [4/5] iperf3 uplink 60s..."
    if iperf3 -c "$HOME_IP" -p "$SERVER_IPERF_PORT" -t 60 -i 10 2>&1 \
        | tee "${MEAS_DIR}/bw_uplink.txt"; then
        log "  Uplink OK"
    else
        warn "  iperf3 uplink gagal"
        echo "FAILED" > "${MEAS_DIR}/bw_uplink.txt"
    fi

    log "  [4b/5] iperf3 downlink 60s..."
    if iperf3 -c "$HOME_IP" -p "$SERVER_IPERF_PORT" -t 60 -R -i 10 2>&1 \
        | tee "${MEAS_DIR}/bw_downlink.txt"; then
        log "  Downlink OK"
    else
        warn "  iperf3 downlink gagal"
        echo "FAILED" > "${MEAS_DIR}/bw_downlink.txt"
    fi

    log "  [5/5] Burst latency ${BURST_PAYLOAD_KB}KB × ${BURST_REPEAT}x..."
    python3 "$BURST_SCRIPT" \
        --host "$HOME_IP" \
        --port "$BURST_PORT" \
        --size "$BURST_PAYLOAD_KB" \
        --repeat "$BURST_REPEAT" \
        --interval "$BURST_INTERVAL" \
        --output "${MEAS_DIR}/burst.txt"

    RTT_HOME=$(grep "rtt min/avg/max/mdev" "${MEAS_DIR}/ping_homeserver.txt" \
               | head -1 | awk '{print $4}' | cut -d'/' -f2)
    RTT_VPS=$(grep "rtt min/avg/max/mdev" "${MEAS_DIR}/ping_vps.txt" \
              | head -1 | awk '{print $4}' | cut -d'/' -f2)
    BURST_AVG=$(python3 -c "
import json, re
with open('${MEAS_DIR}/burst.txt') as f:
    txt = f.read()
m = re.search(r'=== JSON DATA ===\s*(\{.*)', txt, re.DOTALL)
if m:
    d = json.loads(m.group(1))
    print(d['results']['stats'].get('avg', 'N/A'))
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    cat > "${MEAS_DIR}/network_summary.txt" << SUMEOF
=== NETWORK MEASUREMENT SUMMARY ===
session_id      : ${SESSION_ID}
timestamp       : $(date '+%Y-%m-%d %H:%M:%S')
rtt_vps_avg_ms  : ${RTT_VPS}
rtt_home_avg_ms : ${RTT_HOME}
burst_avg_ms    : ${BURST_AVG}
baseline_vps    : ${BASELINE_VPS_AVG_MS} ms
baseline_home   : ${BASELINE_HOME_AVG_MS} ms
SUMEOF

    log "Phase 1 selesai. RTT home: ${RTT_HOME}ms | VPS: ${RTT_VPS}ms | Burst: ${BURST_AVG}ms"
    tg_notify "✅ *Phase 1 Selesai* — Network Measurement
📍 ${LOCATION_NAME} | ${OPERATOR}
📊 RTT VPS: ${RTT_VPS}ms | RTT Home: ${RTT_HOME}ms
💾 Burst avg: ${BURST_AVG}ms
🔁 Baseline: ${BASELINE_HOME_AVG_MS}ms"
}

# ── Phase 2: System Evaluation ────────────────────────────────────────────────
run_phase2() {
    header "=== PHASE 2: SYSTEM EVALUATION ==="
    echo -e "  Methods: ${EDGE_METHODS}"
    tg_notify "🤖 *Phase 2 dimulai* — System Evaluation
📍 ${LOCATION_NAME} | ${OPERATOR}
⚙️ Methods: ${EDGE_METHODS}
⏱ Estimasi: ~40-80 menit"

    mkdir -p "$RESULTS_DIR"
    cd "$ARTEMIS_DIR"

    PYTHONPATH="${ARTEMIS_DIR}" python3 scripts/run_edge.py \
        --location  "${LOCATION_NAME:-${SESSION_ID}}" \
        --operator  "${OPERATOR:-unknown}" \
        --experiment_id "${SESSION_ID}" \
        --methods   "${EDGE_METHODS}" \
        --mode      "${EDGE_MODE}" 2>&1 | tee /tmp/run_edge_output.txt

    # Parse ringkasan dari output
    local summary_text
    summary_text=$(grep -A20 "RINGKASAN" /tmp/run_edge_output.txt | \
                   grep -E "device_only|server_only|static|adaptive" | \
                   awk '{printf "• %s: %s avg, %s err\n", $1, $2, $5}' || echo "")

    log "Phase 2 selesai."
    tg_notify "✅ *Phase 2 Selesai* — System Evaluation
📍 ${LOCATION_NAME} | ${OPERATOR}
${summary_text}
🆔 Session: \`${SESSION_ID}\`"
}

# ── Link results ──────────────────────────────────────────────────────────────
link_results() {
    sep
    log "Linking results ke measurement folder..."
    local result_files
    result_files=$(find "$RESULTS_DIR" -name "*${SESSION_ID}*" 2>/dev/null || true)
    if [[ -n "$result_files" ]]; then
        for f in $result_files; do
            local bname
            bname=$(basename "$f")
            ln -sf "$f" "${MEAS_DIR}/${bname}" 2>/dev/null || \
                cp "$f" "${MEAS_DIR}/${bname}"
            log "  → ${MEAS_DIR}/${bname}"
        done
    else
        warn "Tidak ada results file dengan session_id: ${SESSION_ID}"
    fi
    echo "" >> "${MEAS_DIR}/metadata.txt"
    echo "=== POST-SESSION INFO ===" >> "${MEAS_DIR}/metadata.txt"
    echo "phase2_completed   : $(date '+%Y-%m-%d %H:%M:%S')" >> "${MEAS_DIR}/metadata.txt"
}

# ── Final summary ─────────────────────────────────────────────────────────────
print_final_summary() {
    header "=== SESSION SELESAI ==="

    if [[ -f "${MEAS_DIR}/network_summary.txt" ]]; then
        echo -e "${BOLD}Network Measurement:${NC}"
        grep -E "rtt_|burst_" "${MEAS_DIR}/network_summary.txt" | \
            while IFS= read -r line; do echo "  $line"; done
    fi

    echo ""
    echo -e "${BOLD}System Evaluation:${NC}"
    local result_files
    result_files=$(find "$RESULTS_DIR" -name "*${SESSION_ID}*" 2>/dev/null || true)
    if [[ -n "$result_files" ]]; then
        for f in $result_files; do echo "  → $(basename $f)"; done
    fi

    sep
    echo -e "${BOLD}Output:${NC}"
    echo "  Network measurement  : ${MEAS_DIR}/"
    echo "  System evaluation    : ${RESULTS_DIR}/*${SESSION_ID}*"
    echo ""
    echo -e "${BOLD}Analisis (di homeserver):${NC}"
    echo "  scp pi@10.99.0.3:${RESULTS_DIR}/*${SESSION_ID}* \\"
    echo "      wakata@10.99.0.2:~/jupyter/artemis-v2/results/"
    echo ""
    echo "  python3 scripts/analyze_results.py \\"
    echo "      --results_dir results/ \\"
    echo "      --gt_file data/ground_truth.json \\"
    echo "      --experiment_id ${SESSION_ID} \\"
    echo "      --output_dir analysis/${SESSION_ID}"
    sep

    echo -e "${YELLOW}⚠ JANGAN LUPA:${NC}"
    echo "  □ Foto lokasi + vegetasi (dengan timestamp HP)"
    echo "  □ Screenshot Network Cell Info (RSRP, Band, eNB)"
    echo "  □ Catat anomali sinyal selama session"
    echo "  □ Backup ke HP / cloud sebelum pindah lokasi"
    echo "  □ Di homeserver: jalankan burst_server.py dan iperf3 -s sebelum sesi berikutnya"
    sep

    tg_notify "🏁 *Session Selesai!*
📍 ${LOCATION_NAME} | ${OPERATOR}
🆔 \`${SESSION_ID}\`

*Analisis di homeserver:*
\`\`\`
python3 scripts/analyze_results.py \\
  --experiment_id ${SESSION_ID} \\
  --output_dir analysis/${SESSION_ID}
\`\`\`

⚠️ Jangan lupa foto lokasi & screenshot Network Cell Info!"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    check_deps

    if [[ "$EVAL_ONLY" == "true" ]]; then
        MEAS_DIR="${OUTPUT_BASE}/${SESSION_ID}"
        [[ ! -d "$MEAS_DIR" ]] && err "Measurement folder tidak ditemukan: ${MEAS_DIR}"
        LOCATION_NAME=$(grep "location_name" "${MEAS_DIR}/metadata.txt" 2>/dev/null \
                        | awk '{print $3}' || echo "unknown")
        OPERATOR=$(grep "^operator" "${MEAS_DIR}/metadata.txt" 2>/dev/null \
                   | awk '{print $3}' || echo "unknown")
        log "Mode --eval-only: SESSION_ID=${SESSION_ID}"
        tg_notify "🔄 *Resume Phase 2* — ${SESSION_ID}\n📍 ${LOCATION_NAME} | ${OPERATOR}"
        run_phase2
        link_results
        print_final_summary
    else
        collect_metadata
        check_connectivity

        header "=== RENCANA SESSION ==="
        echo "  Phase 1 — Network Measurement  : ~25 menit"
        echo "  Phase 2 — System Evaluation    : ~40-80 menit"
        echo "  Session ID                     : ${SESSION_ID}"
        sep
        echo -n "Lanjutkan? (y/n): "
        read -r CONFIRM
        [[ "$CONFIRM" != "y" ]] && { log "Dibatalkan."; exit 0; }

        run_phase1
        run_phase2
    fi

    link_results
    print_final_summary
}

main "$@"
