# ARTEMIS v2 — Edge Node

Sistem deteksi kebakaran berbasis kamera untuk perangkat edge (Raspberry Pi). Frame diproses secara lokal menggunakan model ringan, lalu diteruskan ke server inferensi jika diperlukan berdasarkan keputusan adaptive routing.

## Persyaratan

- Raspberry Pi 3, 4B, atau 5
- Python 3.10+
- Model edge dan decision engine (lihat bagian Download Model)

## Instalasi

```bash
git clone https://github.com/inahrafm/artemis-edge.git
cd artemis-edge
pip install -r requirements.txt
```

## Download Model

```bash
mkdir -p models

# Pi 4B dan Pi 5 (TFLite FP32)
curl https://model.weartemis.me/best_float32.tflite -o models/best_float32.tflite

# Pi 3 (ONNX FP32)
curl https://model.weartemis.me/best.onnx -o models/best.onnx

# Decision Engine (semua perangkat)
curl https://model.weartemis.me/lightgbm_de_v2.pkl -o models/lightgbm_de_v2.pkl
```

## Konfigurasi

Edit file config sesuai perangkat yang digunakan:

```bash
# Pi 5
nano config/pi5.yaml

# Pi 4B
nano config/pi4b.yaml

# Pi 3
nano config/pi3.yaml
```

Ganti `YOUR_SERVER_IP` dengan IP server inferensi ARTEMIS:

```yaml
server_url: http://YOUR_SERVER_IP:8000
```

## Menjalankan

Sistem mendeteksi tipe perangkat secara otomatis — tidak perlu menentukan config secara manual.

```bash
PYTHONPATH=/home/pi/artemis-edge python3 scripts/run_edge.py \
    --location nama_lokasi \
    --experiment_id id_eksperimen \
    --methods server_only
```

Pilihan `--methods`:
- `server_only` — semua frame dikirim ke server
- `adaptive` — routing otomatis LOCAL/OFFLOAD/DROP
- `device_only` — inferensi lokal saja
- `static_cooperative` — threshold statis

## Verifikasi Server

Pastikan server inferensi aktif sebelum menjalankan edge:

```bash
curl http://YOUR_SERVER_IP:8000/health
```

## Auto-Configuration

Saat pertama dijalankan, sistem otomatis:
1. Mendeteksi tipe Raspberry Pi dari hardware (`/proc/device-tree/model`)
2. Memuat config dan format model yang optimal per perangkat:
   - Pi 3 → ONNX FP32
   - Pi 4B → TFLite FP32
   - Pi 5 → TFLite FP32
3. Fallback ke inferensi lokal jika server tidak dapat diakses
