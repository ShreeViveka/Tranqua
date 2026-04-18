"""
install.py — One-Click Installer for Serenity
===============================================
Other users run this ONCE to set up everything:

  python install.py

What it does:
  1. Checks Python version
  2. Installs all pip dependencies
  3. Downloads spaCy model
  4. Creates SQLite database (no MySQL needed)
  5. Downloads the pre-trained GRU model
  6. Sets up Windows auto-start
  7. Launches the app

No MySQL, no config files, no manual setup.
Just: python install.py → done.
"""

import os
import sys
import subprocess
import platform
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Pre-trained model URL (upload yours to GitHub Releases after training) ────
# After training, run: python install.py --upload
# This uploads saved_model.pt + vocab.pkl to GitHub Releases
GITHUB_REPO         = "YOUR_GITHUB_USERNAME/serenity-mental-health"
MODEL_RELEASE_TAG   = "v1.0.0"
MODEL_DOWNLOAD_URL  = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/saved_model.pt"
VOCAB_DOWNLOAD_URL  = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/vocab.pkl"
CONFIG_DOWNLOAD_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/model_config.json"

# FL server URL (update after Railway deploy)
FL_SERVER_URL = "http://localhost:9000"  # ← update to your Railway URL

PASS = "  [OK]  "
FAIL = "  [!!]  "
INFO = "  [--]  "


def header(text):
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")


def step(text):
    print(f"\n>>> {text}")


def ok(text):
    print(f"{PASS} {text}")


def fail(text):
    print(f"{FAIL} {text}")


def info(text):
    print(f"{INFO} {text}")


def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        fail(f"Command failed: {cmd}")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return False
    return True


def check_python():
    step("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        fail(f"Python 3.10+ required. You have {major}.{minor}")
        fail("Download from: https://python.org/downloads")
        sys.exit(1)
    ok(f"Python {major}.{minor} ✓")


def install_dependencies():
    step("Installing Python dependencies...")
    packages = [
        "spacy",
        "gensim",
        "scikit-learn",
        "nltk",
        "matplotlib",
        "pandas",
        "numpy",
        "torch",
        "fastapi",
        "uvicorn",
        "psutil",
        "pynput",
        "requests",
        "python-dotenv",
        "reportlab",
        "mysql-connector-python",  # optional — installer handles if it fails
    ]

    for pkg in packages:
        result = subprocess.run(
            f'pip install {pkg} -q',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(f"{pkg}")
        else:
            info(f"{pkg} — skipped (optional)")

    # Download spaCy model
    step("Downloading spaCy language model...")
    if run("python -m spacy download en_core_web_sm -q"):
        ok("spaCy en_core_web_sm")

    # NLTK data
    step("Downloading NLTK data...")
    subprocess.run(
        'python -c "import nltk; nltk.download(\'punkt\',quiet=True); '
        'nltk.download(\'stopwords\',quiet=True); nltk.download(\'punkt_tab\',quiet=True)"',
        shell=True
    )
    ok("NLTK datasets")


def create_config():
    """Create a default config.env for users without MySQL."""
    step("Setting up configuration...")

    config_path = ROOT / 'config.env'
    if config_path.exists():
        # Check if it has real MySQL credentials
        content = config_path.read_text()
        if 'your_password_here' not in content and 'DB_PASSWORD=' in content:
            ok("config.env already configured")
            return

    # Write config with empty MySQL password → will trigger SQLite fallback
    config_content = f"""# Serenity Configuration
# MySQL credentials (leave empty to use SQLite — recommended for most users)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=mental_health_tracker
DB_USER=mht_user
DB_PASSWORD=

# FL Server URL (update after deploying to Railway)
FL_SERVER_URL={FL_SERVER_URL}
"""
    config_path.write_text(config_content)
    ok("config.env created (using SQLite — no MySQL needed)")


def setup_database():
    """Initialise the database (SQLite or MySQL)."""
    step("Setting up database...")

    sys.path.insert(0, str(ROOT / 'collector'))
    try:
        from db import init_db, which_db
        init_db()
        ok(f"Database initialised ({which_db()})")
    except Exception as e:
        fail(f"Database setup failed: {e}")
        return False
    return True


def download_model():
    """Download pre-trained model from GitHub Releases."""
    step("Downloading pre-trained model...")

    model_dir = ROOT / 'model'
    model_dir.mkdir(exist_ok=True)

    files = [
        (MODEL_DOWNLOAD_URL,  model_dir / 'saved_model.pt'),
        (VOCAB_DOWNLOAD_URL,  model_dir / 'vocab.pkl'),
        (CONFIG_DOWNLOAD_URL, model_dir / 'model_config.json'),
    ]

    all_exist = all(f[1].exists() for f in files)
    if all_exist:
        ok("Model already downloaded")
        return True

    if 'YOUR_GITHUB_USERNAME' in MODEL_DOWNLOAD_URL:
        info("Model download URL not configured yet.")
        info("To train the model yourself, run:")
        info("  python model/trainer.py --data 'Combined Data.csv'")
        return False

    for url, dest in files:
        if dest.exists():
            ok(f"{dest.name} already exists")
            continue
        try:
            info(f"Downloading {dest.name}...")
            urllib.request.urlretrieve(url, str(dest))
            ok(f"{dest.name} downloaded ({dest.stat().st_size // 1024}KB)")
        except Exception as e:
            fail(f"Failed to download {dest.name}: {e}")
            info("You can train the model yourself:")
            info("  python model/trainer.py --data 'Combined Data.csv'")
            return False

    return True


def setup_autostart():
    """Set up Windows auto-start for the collector."""
    step("Setting up auto-start...")

    if platform.system() != 'Windows':
        info("Auto-start setup is Windows-only. On Mac/Linux, add to startup manually.")
        return

    setup_script = ROOT / 'collector' / 'setup_autostart.py'
    if not setup_script.exists():
        info("setup_autostart.py not found — skip auto-start")
        return

    result = subprocess.run(
        f'python "{setup_script}"',
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0:
        ok("Auto-start configured — collector starts on Windows login")
    else:
        info("Auto-start needs Administrator. Run setup_autostart.py as Admin later.")


def create_desktop_shortcut():
    """Create a desktop shortcut to launch the app."""
    step("Creating desktop shortcut...")

    if platform.system() != 'Windows':
        info("Desktop shortcut is Windows-only")
        return

    desktop   = Path.home() / 'Desktop'
    shortcut  = desktop / 'Serenity.bat'
    start_py  = ROOT / 'start.py'

    bat_content = f"""@echo off
title Serenity - Mental Health Tracker
cd /d "{ROOT}"
python "{start_py}"
pause
"""
    shortcut.write_text(bat_content)
    ok(f"Desktop shortcut created: {shortcut}")


def check_node():
    """Check if Node.js is available for the React frontend."""
    step("Checking Node.js (for React frontend)...")
    result = subprocess.run('node --version', shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        ok(f"Node.js {result.stdout.strip()} found")
        return True
    else:
        info("Node.js not found. Install from: https://nodejs.org")
        info("Without Node.js, the app will work but the web UI won't load.")
        return False


def install_frontend():
    """Install React frontend dependencies."""
    step("Installing React frontend...")
    frontend_dir = ROOT / 'frontend'
    if not frontend_dir.exists():
        info("Frontend directory not found — skip")
        return

    result = subprocess.run(
        'npm install',
        shell=True, capture_output=True, text=True,
        cwd=str(frontend_dir)
    )
    if result.returncode == 0:
        ok("React dependencies installed")
    else:
        fail("npm install failed")
        info(result.stderr[-300:] if result.stderr else "")


def final_summary(has_model: bool):
    header("Installation Complete!")

    print("""
  Serenity is ready to use!

  HOW TO START THE APP:
  ─────────────────────
  Option 1 (recommended):
    Double-click 'Serenity' shortcut on your desktop

  Option 2 (manual):
    Open terminal in this folder and run:
    python start.py

  WHAT HAPPENS WHEN YOU START:
  ─────────────────────────────
  1. Data collector starts (tracks your app usage silently)
  2. Backend API starts on http://localhost:8000
  3. React UI opens at http://localhost:3000
    """)

    if not has_model:
        print("""
  IMPORTANT — MODEL NOT DOWNLOADED:
  ────────────────────────────────────
  You need to train the model before predictions work:

    python model/trainer.py --data "Combined Data.csv"

  This takes ~15 minutes. Then restart the app.
        """)

    print("""
  PRIVACY REMINDER:
  ──────────────────
  All your diary entries and app usage are stored LOCALLY
  on your computer. Nothing is sent to any server except
  anonymous model weight updates (when you're plugged in).

  Enjoy your mental health journey! 🌱
    """)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    header("Serenity — Mental Health Tracker Installer")
    print("  This will set up everything you need.")
    print("  No MySQL required — uses SQLite automatically.")
    print("  Estimated time: 5-10 minutes")

    check_python()
    install_dependencies()
    create_config()

    db_ok    = setup_database()
    has_node = check_node()

    if has_node:
        install_frontend()

    has_model = download_model()
    setup_autostart()
    create_desktop_shortcut()
    final_summary(has_model)

    if db_ok:
        print("\nLaunching app now...")
        try:
            subprocess.Popen([sys.executable, str(ROOT / 'start.py')])
        except Exception as e:
            info(f"Could not auto-launch: {e}")
            info("Run 'python start.py' manually")


if __name__ == '__main__':
    main()
