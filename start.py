"""
start.py — Launch Everything with One Command
===============================================
Starts both the background data collector AND the FastAPI backend
in parallel so you only need to run one command.

Usage:
  cd D:\\nlp\\mental_health_app
  python start.py

What it starts:
  1. Data collector  → runs silently in background thread
  2. FastAPI backend → http://localhost:8000
  3. Opens browser   → http://localhost:3000 (React frontend)

Stop everything with Ctrl+C
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


def run_collector():
    """Run the data collector in a background thread."""
    sys.path.insert(0, os.path.join(ROOT, 'collector'))
    try:
        from data_collector import DataCollector
        collector = DataCollector()
        collector.start()
    except Exception as e:
        log.error(f"[Collector] Failed to start: {e}")


def run_backend():
    """Run the FastAPI backend."""
    import uvicorn
    sys.path.insert(0, os.path.join(ROOT, 'backend'))
    sys.path.insert(0, os.path.join(ROOT, 'collector'))
    sys.path.insert(0, os.path.join(ROOT, 'model'))

    uvicorn.run(
        "backend.main:app",
        host    = "127.0.0.1",
        port    = 8000,
        reload  = False,
        workers = 1,
        log_level = "info",
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  Mental Health Tracker - Starting...")
    print("=" * 55)
    print("  Collector : background data logging")
    print("  Backend   : http://localhost:8000")
    print("  Frontend  : http://localhost:3000")
    print("  API Docs  : http://localhost:8000/docs")
    print("  Stop with : Ctrl+C")
    print("=" * 55)

    # Start collector in background thread
    collector_thread = threading.Thread(target=run_collector, daemon=True)
    collector_thread.start()
    log.info("[Start] Collector thread started.")

    # Wait a moment then open browser
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:3000")
    threading.Thread(target=open_browser, daemon=True).start()

    # Run backend in main thread (blocking)
    try:
        run_backend()
    except KeyboardInterrupt:
        log.info("[Start] Shutting down...")
        print("\nGoodbye! Your data is saved locally.")
