"""
data_collector.py — Background Data Collector
=============================================
Runs silently in the background and collects:
  - Active app / window every 5 seconds
  - Keyboard & mouse activity level
  - Idle time detection
  - Battery / charging status
  - Summarises everything into SQLite at end of each hour

HOW TO RUN:
  python data_collector.py

  It will run forever until you Ctrl+C or close the terminal.
  For auto-start on Windows boot, run setup_autostart.py

PRIVACY:
  - All data stored locally in data/tracker.db
  - Window titles are stored but never sent to any server
  - Only model weight updates (NOT raw data) ever leave your laptop
"""

import time
import threading
import logging
import sys
import os
from datetime import datetime, date
from collections import defaultdict

# ── Third-party imports ──────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    print("[ERROR] psutil not installed. Run: pip install psutil")
    sys.exit(1)

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    print("[WARNING] pynput not installed. Keyboard/mouse tracking disabled.")
    print("          Run: pip install pynput")
    PYNPUT_AVAILABLE = False

# Windows-specific: get active window title
try:
    import ctypes
    import ctypes.wintypes
    WINDOWS = True
except ImportError:
    WINDOWS = False

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import init_db, insert_app_usage, upsert_daily_summary
from categories import classify_app, is_late_night

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'collector.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECS   = 5      # check active window every 5 seconds
SUMMARY_INTERVAL_SECS = 300   # write summary to DB every 5 minutes
IDLE_THRESHOLD_SECS  = 60     # no input for 60s = idle
BREAK_THRESHOLD_SECS = 600    # idle for 10min = a "break"


# ════════════════════════════════════════════════════════════════════════════
# ACTIVE WINDOW DETECTION
# ════════════════════════════════════════════════════════════════════════════

def get_active_window_windows():
    """Get the active window app name and title on Windows."""
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()

        # Get process ID from window handle
        pid = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        # Get process name
        try:
            proc     = psutil.Process(pid.value)
            app_name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            app_name = "unknown"

        # Get window title
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
        buf    = ctypes.create_unicode_buffer(length)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length)
        window_title = buf.value or ""

        return app_name, window_title
    except Exception:
        return "unknown", ""


def get_active_window_fallback():
    """Fallback: get the top CPU-using process as proxy for active app."""
    try:
        procs = sorted(
            psutil.process_iter(['name', 'cpu_percent']),
            key=lambda p: p.info['cpu_percent'] or 0,
            reverse=True
        )
        for p in procs[:3]:
            name = p.info.get('name', '')
            if name and name.lower() not in ('system idle process', 'system', 'idle'):
                return name, ""
    except Exception:
        pass
    return "unknown", ""


def get_active_window():
    """Platform-aware active window getter."""
    if WINDOWS:
        return get_active_window_windows()
    return get_active_window_fallback()


# ════════════════════════════════════════════════════════════════════════════
# BATTERY / CHARGING STATUS
# ════════════════════════════════════════════════════════════════════════════

def is_plugged_in() -> bool:
    """Returns True if laptop is plugged into power."""
    battery = psutil.sensors_battery()
    if battery is None:
        return True   # desktop PC — always "plugged in"
    return battery.power_plugged


def has_internet() -> bool:
    """Quick check for internet connectivity."""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# ════════════════════════════════════════════════════════════════════════════
# INPUT ACTIVITY TRACKER
# ════════════════════════════════════════════════════════════════════════════

class ActivityTracker:
    """
    Tracks keyboard and mouse activity using pynput listeners.
    Runs in background threads — never blocks the main collector.
    """

    def __init__(self):
        self.keystrokes       = 0
        self.mouse_clicks     = 0
        self.mouse_distance   = 0.0
        self._last_mouse_pos  = None
        self._last_input_time = time.time()
        self._lock            = threading.Lock()

    def start(self):
        if not PYNPUT_AVAILABLE:
            return

        # Keyboard listener
        self._kb_listener = keyboard.Listener(on_press=self._on_key)
        self._kb_listener.daemon = True
        self._kb_listener.start()

        # Mouse listener
        self._ms_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click
        )
        self._ms_listener.daemon = True
        self._ms_listener.start()
        log.info("[Activity] Input listeners started.")

    def _on_key(self, key):
        with self._lock:
            self.keystrokes      += 1
            self._last_input_time = time.time()

    def _on_click(self, x, y, button, pressed):
        if pressed:
            with self._lock:
                self.mouse_clicks    += 1
                self._last_input_time = time.time()

    def _on_move(self, x, y):
        with self._lock:
            if self._last_mouse_pos is not None:
                dx = x - self._last_mouse_pos[0]
                dy = y - self._last_mouse_pos[1]
                self.mouse_distance  += (dx**2 + dy**2) ** 0.5
            self._last_mouse_pos  = (x, y)
            self._last_input_time = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_input_time

    @property
    def is_idle(self) -> bool:
        return self.idle_seconds > IDLE_THRESHOLD_SECS

    def snapshot_and_reset(self) -> dict:
        """Return current counts and reset for next interval."""
        with self._lock:
            snap = {
                'keystrokes'     : self.keystrokes,
                'mouse_clicks'   : self.mouse_clicks,
                'mouse_distance' : round(self.mouse_distance, 1),
                'idle_seconds'   : round(self.idle_seconds, 1),
                'is_idle'        : self.is_idle
            }
            self.keystrokes     = 0
            self.mouse_clicks   = 0
            self.mouse_distance = 0.0
        return snap


# ════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR
# ════════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    The main background collector.
    Polls the active window every POLL_INTERVAL_SECS seconds,
    accumulates data, and writes to DB every SUMMARY_INTERVAL_SECS seconds.
    """

    def __init__(self):
        self.activity     = ActivityTracker()
        self._running     = False

        # In-memory accumulators (reset every summary interval)
        self._app_seconds : dict = defaultdict(int)   # app_name -> seconds
        self._app_titles  : dict = {}                  # app_name -> last title
        self._idle_seconds: int  = 0
        self._active_seconds: int = 0
        self._late_night_seconds: int = 0
        self._break_count : int  = 0
        self._plugged_mins: float = 0.0
        self._session_start: str  = None
        self._last_break_start: float = None
        self._total_keystrokes: int = 0
        self._total_mouse_dist: float = 0.0
        self._interval_start = time.time()
        self._day_start      = datetime.now().strftime('%Y-%m-%d')

    def start(self):
        """Start the collector. Blocks forever."""
        log.info("=" * 55)
        log.info("  Mental Health Tracker — Data Collector Started")
        log.info("=" * 55)
        log.info(f"  Poll interval  : every {POLL_INTERVAL_SECS}s")
        log.info(f"  Summary write  : every {SUMMARY_INTERVAL_SECS}s")
        log.info(f"  Idle threshold : {IDLE_THRESHOLD_SECS}s")
        log.info(f"  Data saved to  : data/tracker.db")
        log.info("  Press Ctrl+C to stop.")
        log.info("=" * 55)

        init_db()
        self.activity.start()
        self._running     = True
        self._session_start = datetime.now().isoformat()

        try:
            last_summary_time = time.time()

            while self._running:
                self._poll()

                # Write summary every SUMMARY_INTERVAL_SECS
                if time.time() - last_summary_time >= SUMMARY_INTERVAL_SECS:
                    self._flush_to_db()
                    last_summary_time = time.time()

                # Check if day rolled over
                today = datetime.now().strftime('%Y-%m-%d')
                if today != self._day_start:
                    log.info(f"[Collector] New day detected: {today}")
                    self._flush_to_db(end_of_day=True)
                    self._reset_day()
                    self._day_start = today

                time.sleep(POLL_INTERVAL_SECS)

        except KeyboardInterrupt:
            log.info("[Collector] Stopping — flushing final data...")
            self._flush_to_db(end_of_day=True)
            log.info("[Collector] Stopped cleanly.")

    def _poll(self):
        """Single poll — called every POLL_INTERVAL_SECS."""
        now  = datetime.now()
        hour = now.hour

        # Get active window
        app_name, window_title = get_active_window()

        # Update app usage accumulators
        self._app_seconds[app_name]  += POLL_INTERVAL_SECS
        self._app_titles[app_name]    = window_title

        # Track idle vs active
        activity_snap = self.activity.snapshot_and_reset()
        self._total_keystrokes += activity_snap['keystrokes']
        self._total_mouse_dist += activity_snap['mouse_distance']

        if activity_snap['is_idle']:
            self._idle_seconds += POLL_INTERVAL_SECS

            # Detect a break (idle > 10 mins)
            if self._last_break_start is None:
                self._last_break_start = time.time()
            elif time.time() - self._last_break_start >= BREAK_THRESHOLD_SECS:
                self._break_count += 1
                self._last_break_start = None   # reset so we don't double count
        else:
            self._active_seconds   += POLL_INTERVAL_SECS
            self._last_break_start  = None

        # Track late-night usage
        if is_late_night(hour):
            self._late_night_seconds += POLL_INTERVAL_SECS

        # Track plugged-in time
        if is_plugged_in():
            self._plugged_mins += POLL_INTERVAL_SECS / 60

    def _flush_to_db(self, end_of_day=False):
        """Write accumulated data to MySQL."""
        today    = datetime.now().strftime('%Y-%m-%d')
        now_time = datetime.now().isoformat()

        log.info(f"[Collector] Flushing data for {today} "
                 f"({'end of day' if end_of_day else 'interval'})")

        # Write individual app usage records
        for app_name, secs in self._app_seconds.items():
            if secs < POLL_INTERVAL_SECS:
                continue   # skip very brief appearances
            title    = self._app_titles.get(app_name, "")
            category = classify_app(app_name, title)
            insert_app_usage(
                timestamp    = now_time,
                app_name     = app_name,
                window_title = title,
                duration_secs= secs,
                category     = category
            )

        # Compute category totals
        from db import get_connection
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT category, SUM(duration_secs) as secs
            FROM app_usage WHERE date = %s
            GROUP BY category
        """, (today,))
        cats = cursor.fetchall()
        cursor.close()
        conn.close()

        cat_mins = {row['category']: row['secs'] / 60 for row in cats}

        # Compute total screen time (all app usage today)
        total_screen_mins = sum(cat_mins.values())

        # Compute session info
        pc_on_time  = self._session_start
        pc_off_time = now_time

        # Update daily summary
        upsert_daily_summary(
            date                   = today,
            total_screen_time_mins = round(total_screen_mins, 1),
            active_time_mins       = round(self._active_seconds / 60, 1),
            idle_time_mins         = round(self._idle_seconds / 60, 1),
            social_media_mins      = round(cat_mins.get('social_media', 0), 1),
            work_app_mins          = round(cat_mins.get('work', 0), 1),
            entertainment_mins     = round(cat_mins.get('entertainment', 0), 1),
            browser_mins           = round(cat_mins.get('browser', 0), 1),
            late_night_usage_mins  = round(self._late_night_seconds / 60, 1),
            break_count            = self._break_count,
            keystrokes_count       = self._total_keystrokes,
            mouse_distance_px      = round(self._total_mouse_dist, 0),
            pc_on_time             = pc_on_time,
            pc_off_time            = pc_off_time,
            plugged_in_mins        = round(self._plugged_mins, 1),
        )

        # Reset interval accumulators (but not day-level counts)
        self._app_seconds.clear()

        if end_of_day:
            log.info(f"[Collector] Day summary written for {today}")
            log.info(f"            Screen time : {total_screen_mins:.0f} mins")
            log.info(f"            Active      : {self._active_seconds/60:.0f} mins")
            log.info(f"            Idle        : {self._idle_seconds/60:.0f} mins")
            log.info(f"            Social media: {cat_mins.get('social_media',0):.0f} mins")
            log.info(f"            Keystrokes  : {self._total_keystrokes}")

    def _reset_day(self):
        """Reset all day-level accumulators for the new day."""
        self._idle_seconds       = 0
        self._active_seconds     = 0
        self._late_night_seconds = 0
        self._break_count        = 0
        self._plugged_mins       = 0.0
        self._total_keystrokes   = 0
        self._total_mouse_dist   = 0.0
        self._session_start      = datetime.now().isoformat()
        self._last_break_start   = None


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    collector = DataCollector()
    collector.start()
