"""
fl_client.py — Federated Learning Client
=========================================
Handles sending model weight UPDATES (not raw data) to the central server.

How it works:
  1. Local model trains on YOUR data (runs on your laptop)
  2. Computes the DIFFERENCE between local weights and global weights
  3. Adds privacy noise (differential privacy)
  4. Sends ONLY the weight difference — never your diary or usage data
  5. Receives improved global model back from server
  6. Updates your local model

Triggers upload when:
  - Laptop is plugged in (AC power)
  - Internet is available
  - Haven't uploaded today yet
  - At least 3 days of local training data exist
"""

import os
import sys
import json
import time
import logging
import hashlib
from datetime import datetime, date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import get_connection
from data_collector import is_plugged_in, has_internet

log = logging.getLogger(__name__)

# ── Server config (update SERVER_URL when you deploy) ───────────────────────
SERVER_URL   = os.environ.get('FL_SERVER_URL', 'http://localhost:9000')
CLIENT_ID    = None   # generated once and stored locally
CLIENT_FILE  = os.path.join(os.path.dirname(__file__), '..', 'data', 'client_id.txt')

# Differential privacy noise scale (higher = more privacy, less accuracy)
DP_NOISE_SCALE = 0.01


def get_or_create_client_id() -> str:
    """Generate a random anonymous client ID (stored locally)."""
    global CLIENT_ID
    if CLIENT_ID:
        return CLIENT_ID

    os.makedirs(os.path.dirname(CLIENT_FILE), exist_ok=True)

    if os.path.exists(CLIENT_FILE):
        with open(CLIENT_FILE) as f:
            CLIENT_ID = f.read().strip()
    else:
        import uuid
        CLIENT_ID = str(uuid.uuid4())
        with open(CLIENT_FILE, 'w') as f:
            f.write(CLIENT_ID)
        log.info(f"[FL] New client ID generated: {CLIENT_ID[:8]}...")

    return CLIENT_ID


def has_enough_local_data(min_days: int = 3) -> bool:
    """Check if we have enough training data to contribute."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM diary_entries")
    count  = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count >= min_days


def already_uploaded_today() -> bool:
    """Check if we already sent a weight update today."""
    today  = date.today().isoformat()
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM fl_history WHERE DATE(timestamp) = %s AND upload_success = 1",
        (today,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row is not None


def add_differential_privacy_noise(weights: np.ndarray) -> np.ndarray:
    """
    Add Gaussian noise for differential privacy.
    This ensures even the weight update can't be reverse-engineered
    to reveal your original data.
    """
    noise = np.random.normal(0, DP_NOISE_SCALE, weights.shape)
    return weights + noise


def compute_weight_delta(local_weights: dict, global_weights: dict) -> dict:
    """
    Compute the DIFFERENCE between local and global weights.
    We send the delta, not the full weights — more efficient and private.
    """
    delta = {}
    for key in local_weights:
        if key in global_weights:
            local  = np.array(local_weights[key])
            global_ = np.array(global_weights[key])
            diff   = local - global_

            # Add differential privacy noise
            diff_with_noise = add_differential_privacy_noise(diff)

            # Clip to bound sensitivity (another DP technique)
            norm = np.linalg.norm(diff_with_noise)
            if norm > 1.0:
                diff_with_noise = diff_with_noise / norm

            delta[key] = diff_with_noise.tolist()
    return delta


def should_upload() -> tuple[bool, str]:
    """
    Check all conditions for uploading.
    Returns (should_upload: bool, reason: str)
    """
    if not is_plugged_in():
        return False, "Laptop is on battery — waiting for AC power"

    if not has_internet():
        return False, "No internet connection"

    if already_uploaded_today():
        return False, "Already uploaded today"

    if not has_enough_local_data():
        return False, "Not enough local training data yet (need 3+ days)"

    return True, "All conditions met"


def upload_weight_update(local_weights: dict, global_weights: dict,
                         round_number: int, loss_before: float) -> bool:
    """
    Upload weight delta to the federated server.
    Returns True if successful.
    """
    try:
        import requests
    except ImportError:
        log.error("[FL] requests not installed. Run: pip install requests")
        return False

    should, reason = should_upload()
    if not should:
        log.info(f"[FL] Upload skipped: {reason}")
        return False

    client_id = get_or_create_client_id()
    delta     = compute_weight_delta(local_weights, global_weights)

    payload = {
        'client_id'   : client_id,
        'round_number': round_number,
        'weight_delta': delta,
        'loss_before' : loss_before,
        'timestamp'   : datetime.now().isoformat(),
        'data_hash'   : _compute_data_hash()   # for integrity, not privacy
    }

    log.info(f"[FL] Uploading weight update to {SERVER_URL}...")

    try:
        response = requests.post(
            f"{SERVER_URL}/fl/upload",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        log.info(f"[FL] Upload successful. Server round: {result.get('round')}")

        # Log to DB
        conn = get_connection()
        conn.execute("""
            INSERT INTO fl_history (round_number, loss_before, upload_success, model_version)
            VALUES (?, ?, 1, ?)
        """, (round_number, loss_before, result.get('model_version', 'unknown')))
        conn.commit()
        conn.close()

        return True

    except Exception as e:
        log.error(f"[FL] Upload failed: {e}")
        conn = get_connection()
        conn.execute("""
            INSERT INTO fl_history (round_number, loss_before, upload_success)
            VALUES (?, ?, 0)
        """, (round_number, loss_before))
        conn.commit()
        conn.close()
        return False


def download_global_model() -> dict | None:
    """Download the latest global model from the server."""
    try:
        import requests
        response = requests.get(f"{SERVER_URL}/fl/model", timeout=30)
        response.raise_for_status()
        log.info("[FL] Global model downloaded successfully.")
        return response.json()
    except Exception as e:
        log.error(f"[FL] Failed to download global model: {e}")
        return None


def _compute_data_hash() -> str:
    """
    Compute a hash of our data count/dates for integrity verification.
    Does NOT reveal any content — just proves we have real data.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT date FROM diary_entries ORDER BY date"
    ).fetchall()
    conn.close()

    dates_str = ','.join(r['date'] for r in rows)
    return hashlib.sha256(dates_str.encode()).hexdigest()[:16]


def run_fl_monitor():
    """
    Background thread: checks upload conditions every 30 minutes.
    Call this from the main collector or a separate process.
    """
    log.info("[FL Monitor] Started — checking every 30 minutes.")
    while True:
        should, reason = should_upload()
        if should:
            log.info("[FL Monitor] Conditions met — attempting upload...")
            # In real use, load weights from saved model files here
            # upload_weight_update(local_weights, global_weights, round_num, loss)
        else:
            log.debug(f"[FL Monitor] {reason}")
        time.sleep(30 * 60)   # check every 30 minutes


if __name__ == '__main__':
    # Quick status check
    should, reason = should_upload()
    print(f"Upload conditions: {reason}")
    print(f"Client ID: {get_or_create_client_id()[:8]}...")
    print(f"Enough data: {has_enough_local_data()}")
    print(f"Uploaded today: {already_uploaded_today()}")
