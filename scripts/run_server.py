#!/usr/bin/env python3
"""
scripts/run_server.py
======================
Entry point ARTEMIS v2 inference server.

CARA PAKAI:

  # Development (Flask-style, single threaded)
  python3 scripts/run_server.py --model best.pt --port 8000

  # Production (Gunicorn + uvicorn — untuk RQ3 multi-node)
  python3 scripts/run_server.py --model best.pt --port 8000 --gunicorn

  # Print perintah Gunicorn saja
  python3 scripts/run_server.py --model best.pt --gunicorn --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIS v2 — Inference Server Entry Point")
    parser.add_argument("--model",      required=True,
                        help="Path ke best.pt (YOLOv26x)")
    parser.add_argument("--thresholds", default="thresholds_v2.json")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--log_level",  default="INFO")
    parser.add_argument("--gunicorn",   action="store_true",
                        help="Jalankan dengan Gunicorn (production mode)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print perintah Gunicorn tanpa menjalankan")
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("artemis.run_server")

    model_abs = str(Path(args.model).expanduser().resolve())
    if not Path(model_abs).exists():
        log.error(f"Model tidak ditemukan: {model_abs}")
        sys.exit(1)

    if args.gunicorn or args.dry_run:
        # Cek uvicorn tersedia
        try:
            import uvicorn
        except ImportError:
            log.error("uvicorn tidak terinstall. Install dengan:")
            log.error("  pip install uvicorn --break-system-packages")
            sys.exit(1)

        gunicorn_cmd = (
            f"ARTEMIS_MODEL={model_abs} "
            f"ARTEMIS_DEVICE={args.device} "
            f"ARTEMIS_THRESHOLDS={args.thresholds} "
            f"ARTEMIS_LOG_DIR={args.log_dir} "
            f"gunicorn -w 1 -k uvicorn.workers.UvicornWorker "
            f"-b {args.host}:{args.port} "
            f"--timeout 60 "
            f"'server.app:create_app()'"
        )

        print("\n# Perintah Gunicorn untuk production:")
        print(gunicorn_cmd)
        print()

        if args.dry_run:
            return

        log.info("Memulai server dengan Gunicorn...")
        os.execvp("bash", ["bash", "-c", gunicorn_cmd])
        return

    # Development mode — uvicorn langsung
    try:
        import uvicorn
    except ImportError:
        # Fallback ke Flask-style jika uvicorn tidak ada
        log.warning("uvicorn tidak ada, fallback ke Flask dev server")
        _run_flask(model_abs, args)
        return

    # Set env vars untuk create_app()
    os.environ["ARTEMIS_MODEL"]      = model_abs
    os.environ["ARTEMIS_DEVICE"]     = args.device
    os.environ["ARTEMIS_THRESHOLDS"] = args.thresholds
    os.environ["ARTEMIS_LOG_DIR"]    = args.log_dir

    log.info("=" * 60)
    log.info("ARTEMIS v2 Inference Server — Topik 3")
    log.info(f"  Model    : {model_abs}")
    log.info(f"  Device   : {args.device}")
    log.info(f"  Endpoint : http://{args.host}:{args.port}")
    log.info(f"  Docs     : http://{args.host}:{args.port}/docs")
    log.info("=" * 60)

    from server.app import create_app
    app = create_app(model_abs, args.device, args.thresholds, args.log_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


def _run_flask(model_path: str, args):
    """Fallback Flask dev server jika uvicorn tidak tersedia."""
    log = logging.getLogger("artemis.run_server")
    log.warning("Mode Flask dev — untuk multi-node install uvicorn + gunicorn")

    # Import server_inference_v2 sebagai fallback
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from server_inference_v2 import create_app as create_flask_app
        app = create_flask_app(model_path, args.device, args.thresholds)
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except ImportError:
        log.error("Tidak bisa import server. Install uvicorn atau pastikan "
                  "server_inference_v2.py ada di scripts/")
        sys.exit(1)


if __name__ == "__main__":
    main()
