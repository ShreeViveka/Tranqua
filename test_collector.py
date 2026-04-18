"""
test_collector.py — Verify everything is working before running for real
=========================================================================
Run this first:
    python test_collector.py

It checks:
  1. All dependencies are installed
  2. Database can be created
  3. Active window detection works
  4. Battery/charging detection works
  5. Internet check works
  6. Input tracking works (keyboard/mouse)
  7. Category classification works
  8. Writes a test record to the database

Expected output: All checks show [OK]
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collector'))

PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"

print("=" * 55)
print("  Mental Health Tracker — Setup Verification")
print("=" * 55)

errors = []

# ── 1. Check psutil ──────────────────────────────────────────────────────────
try:
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    print(f"{PASS} psutil installed  (CPU: {cpu}%)")
except ImportError:
    print(f"{FAIL} psutil NOT installed → run: pip install psutil")
    errors.append("psutil")

# ── 2. Check pynput ─────────────────────────────────────────────────────────
try:
    from pynput import mouse, keyboard
    print(f"{PASS} pynput installed  (keyboard/mouse tracking ready)")
except ImportError:
    print(f"{WARN} pynput NOT installed → keyboard tracking disabled")
    print(f"       Install with: pip install pynput")

# ── 3. Check database ────────────────────────────────────────────────────────
try:
    from db import init_db, get_daily_summary, save_diary_entry, get_diary_entry
    init_db()
    print(f"{PASS} Database initialised  (data/tracker.db)")
except Exception as e:
    print(f"{FAIL} Database error: {e}")
    errors.append("database")

# ── 4. Check active window detection ────────────────────────────────────────
try:
    from data_collector import get_active_window
    app, title = get_active_window()
    print(f"{PASS} Active window detected  ({app} | {title[:40] if title else 'no title'})")
except Exception as e:
    print(f"{FAIL} Active window detection failed: {e}")
    errors.append("window_detection")

# ── 5. Check battery status ──────────────────────────────────────────────────
try:
    from data_collector import is_plugged_in
    plugged = is_plugged_in()
    battery = psutil.sensors_battery()
    pct     = f"{battery.percent:.0f}%" if battery else "N/A (desktop)"
    status  = "Plugged in" if plugged else "On battery"
    print(f"{PASS} Battery status  ({status}, {pct})")
except Exception as e:
    print(f"{FAIL} Battery check failed: {e}")

# ── 6. Check internet ────────────────────────────────────────────────────────
try:
    from data_collector import has_internet
    internet = has_internet()
    print(f"{PASS} Internet check  ({'Connected' if internet else 'No connection'})")
except Exception as e:
    print(f"{FAIL} Internet check failed: {e}")

# ── 7. Check category classifier ────────────────────────────────────────────
try:
    from categories import classify_app
    tests = [
        ("chrome.exe",    "YouTube",    "entertainment"),
        ("chrome.exe",    "Instagram",  "social_media"),
        ("code.exe",      "VSCode",     "work"),
        ("vlc.exe",       "",           "entertainment"),
        ("WINWORD.EXE",   "Document",   "work"),
    ]
    all_ok = True
    for app, title, expected in tests:
        result = classify_app(app, title)
        if result != expected:
            print(f"{FAIL} Category: {app}+{title} → {result} (expected {expected})")
            all_ok = False
    if all_ok:
        print(f"{PASS} Category classifier  (all {len(tests)} test cases correct)")
except Exception as e:
    print(f"{FAIL} Category classifier error: {e}")

# ── 8. Write test diary entry ────────────────────────────────────────────────
try:
    from datetime import date
    from db import save_diary_entry, get_diary_entry
    test_date  = "2099-01-01"   # far future so it doesn't interfere
    test_entry = "This is a test diary entry to verify the database works correctly."
    save_diary_entry(test_date, test_entry)
    retrieved  = get_diary_entry(test_date)
    assert retrieved['entry_text'] == test_entry
    print(f"{PASS} Diary write/read  (database working correctly)")
except Exception as e:
    print(f"{FAIL} Diary write/read failed: {e}")
    errors.append("diary")

# ── 9. Check FL client ────────────────────────────────────────────────────────
try:
    from fl_client import get_or_create_client_id, should_upload
    client_id = get_or_create_client_id()
    ok, reason = should_upload()
    print(f"{PASS} FL client  (ID: {client_id[:8]}...)")
    print(f"       Upload status: {reason}")
except Exception as e:
    print(f"{FAIL} FL client error: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 55)
if not errors:
    print("  ✅ All checks passed! You're ready to start collecting.")
    print()
    print("  To start the collector now:")
    print("    python collector/data_collector.py")
    print()
    print("  To set up auto-start on Windows boot:")
    print("    python collector/setup_autostart.py")
else:
    print(f"  ❌ {len(errors)} issue(s) found: {', '.join(errors)}")
    print("  Fix the issues above, then run this test again.")
print("=" * 55)
