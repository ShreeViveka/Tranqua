"""
bundle_python.py — Bundle Python Backend into Electron App
============================================================
This script uses PyInstaller to package the Python backend
into a standalone executable that ships INSIDE the Electron app.

This means users do NOT need Python installed.
They just install Tranqua.exe and everything works.

Run this ONCE before building the Electron installer:
  python bundle_python.py

Output: python_dist/ folder (Electron picks this up automatically)

Requirements:
  pip install pyinstaller
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)


def bundle():
    print("=" * 55)
    print("  Tranqua — Python Backend Bundler")
    print("=" * 55)

    # Install PyInstaller
    run("pip install pyinstaller -q")

    # Create a launcher script that starts the backend
    launcher = ROOT / "electron" / "_backend_launcher.py"
    launcher.write_text("""
import sys
import os

# Add the bundled app to Python path
app_root = os.path.dirname(sys.executable)
sys.path.insert(0, app_root)
sys.path.insert(0, os.path.join(app_root, 'collector'))
sys.path.insert(0, os.path.join(app_root, 'model'))

import uvicorn

if __name__ == '__main__':
    # Get port from command line args (default 8000)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(
        'backend.main:app',
        host='127.0.0.1',
        port=port,
        log_level='info',
    )
""")

    # Also create collector launcher
    collector_launcher = ROOT / "electron" / "_collector_launcher.py"
    collector_launcher.write_text("""
import sys
import os

app_root = os.path.dirname(sys.executable)
sys.path.insert(0, app_root)
sys.path.insert(0, os.path.join(app_root, 'collector'))

from data_collector import DataCollector

if __name__ == '__main__':
    collector = DataCollector()
    collector.start()
""")

    # Clean previous build
    for d in ['build', 'dist', 'python_dist']:
        if (ROOT / d).exists():
            shutil.rmtree(ROOT / d)
            print(f"Cleaned: {d}/")

    # Bundle backend with PyInstaller
    print("\nBundling Python backend...")
    run(f"""pyinstaller \
        --name tranqua_backend \
        --onedir \
        --noconsole \
        --distpath python_dist \
        --workpath build/pyinstaller \
        --add-data "backend{os.pathsep}backend" \
        --add-data "collector{os.pathsep}collector" \
        --add-data "model{os.pathsep}model" \
        --add-data "config.env{os.pathsep}." \
        --hidden-import uvicorn \
        --hidden-import uvicorn.logging \
        --hidden-import uvicorn.protocols \
        --hidden-import uvicorn.protocols.http \
        --hidden-import uvicorn.protocols.http.h11_impl \
        --hidden-import uvicorn.lifespan \
        --hidden-import uvicorn.lifespan.on \
        --hidden-import fastapi \
        --hidden-import pydantic \
        --hidden-import spacy \
        --hidden-import torch \
        --hidden-import gensim \
        --hidden-import nltk \
        --hidden-import sklearn \
        --hidden-import mysql.connector \
        --hidden-import sqlite3 \
        --hidden-import psutil \
        electron/_backend_launcher.py""")

    # Bundle collector with PyInstaller
    print("\nBundling data collector...")
    run(f"""pyinstaller \
        --name tranqua_collector \
        --onefile \
        --noconsole \
        --distpath python_dist/tranqua_backend \
        --workpath build/pyinstaller_collector \
        --add-data "collector{os.pathsep}collector" \
        --hidden-import psutil \
        --hidden-import pynput \
        --hidden-import pynput.keyboard \
        --hidden-import pynput.mouse \
        --hidden-import sqlite3 \
        --hidden-import mysql.connector \
        electron/_collector_launcher.py""")

    print("\n" + "=" * 55)
    print("  Bundle complete!")
    print(f"  Output: {ROOT / 'python_dist'}")
    print("")
    print("  Now build the Electron app:")
    print("    npm run dist")
    print("=" * 55)


if __name__ == '__main__':
    bundle()
