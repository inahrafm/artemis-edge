# ARTEMIS — Edge Demo

Sistem deteksi kebakaran berbasis kamera untuk perangkat edge (Raspberry Pi).
Inferensi dijalankan secara lokal, adaptif, atau diteruskan ke server — bisa dari video file, webcam, atau RTSP stream.

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

Edit file config sesuai perangkat:

```bash
nano config/pi5.yaml   # Pi 5
nano config/pi4b.yaml  # Pi 4B
nano config/pi3.yaml   # Pi 3
```

Ganti `YOUR_SERVER_IP` dengan IP server inferensi ARTEMIS:

```yaml
server_url: http://YOUR_SERVER_IP:8000
```

## Menjalankan Demo

Sistem mendeteksi tipe perangkat secara otomatis.

```bash
# Video file — tampilkan overlay di layar
python3 scripts/run_demo.py --source video3.mp4 --method server_only --show

# Webcam
python3 scripts/run_demo.py --source 0 --method adaptive --show

# RTSP stream — kirim alarm ke dashboard
python3 scripts/run_demo.py --source rtsp://192.168.1.100/stream --method server_only --send_alarm

# Tanpa tampilan, alarm saja
python3 scripts/run_demo.py --source video3.mp4 --method server_only --send_alarm
```

Pilihan `--method`:

- `server_only` — semua frame dikirim ke server
- `adaptive` — routing otomatis LOCAL/OFFLOAD/DROP berdasarkan decision engine
- `device_only` — inferensi lokal saja, tanpa server

## Auto-Configuration

Saat pertama dijalankan, sistem otomatis:

1. Mendeteksi tipe Raspberry Pi dari hardware (`/proc/device-tree/model`)
2. Memuat config dan format model yang optimal per perangkat:
   - Pi 3 → ONNX FP32
   - Pi 4B → TFLite FP32
   - Pi 5 → TFLite FP32
3. Fallback ke inferensi lokal jika server tidak dapat diakses
