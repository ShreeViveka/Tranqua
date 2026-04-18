"""
db.py — Universal Database Handler
====================================
Automatically uses MySQL if configured, otherwise falls back to SQLite.

For YOU (already have MySQL):
  → config.env has DB_USER, DB_PASSWORD etc → uses MySQL

For OTHER USERS (no MySQL):
  → no config.env or missing credentials → uses SQLite automatically
  → SQLite file created at data/tracker.db
  → Zero installation required

This means other users just run:
  pip install -r requirements.txt
  python start.py
  ...and everything works. No MySQL needed.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── Detect which DB to use ────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.env from project root."""
    current = Path(__file__).resolve()
    for parent in [current.parent, current.parent.parent]:
        cfg = parent / 'config.env'
        if cfg.exists():
            config = {}
            with open(cfg) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, _, v = line.partition('=')
                        config[k.strip()] = v.strip()
            return config
    return {}

_CFG = _load_config()

# Decide which backend to use
_USE_MYSQL = (
    bool(_CFG.get('DB_PASSWORD'))      # password is set
    and _CFG.get('DB_HOST', 'localhost') != ''
)

if _USE_MYSQL:
    log.info("[DB] Using MySQL backend")
    DB_CONFIG = {
        'host'              : _CFG.get('DB_HOST',     'localhost'),
        'port'              : int(_CFG.get('DB_PORT',  3306)),
        'database'          : _CFG.get('DB_NAME',     'mental_health_tracker'),
        'user'              : _CFG.get('DB_USER',     'mht_user'),
        'password'          : _CFG.get('DB_PASSWORD', ''),
        'charset'           : 'utf8mb4',
        'collation'         : 'utf8mb4_unicode_ci',
        'autocommit'        : False,
        'connection_timeout': 10,
    }
else:
    # SQLite path — stored in project_root/data/tracker.db
    _ROOT    = Path(__file__).resolve().parent.parent
    _DB_PATH = str(_ROOT / 'data' / 'tracker.db')
    os.makedirs(str(_ROOT / 'data'), exist_ok=True)
    log.info(f"[DB] Using SQLite backend → {_DB_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# CONNECTION — works for both MySQL and SQLite
# ════════════════════════════════════════════════════════════════════════════

def get_connection():
    """
    Returns a database connection.
    MySQL connections are standard mysql-connector objects.
    SQLite connections are wrapped to behave identically.
    """
    if _USE_MYSQL:
        return _mysql_connection()
    else:
        return _sqlite_connection()


def _mysql_connection():
    try:
        import mysql.connector
        return mysql.connector.connect(**DB_CONFIG)
    except ImportError:
        print("[ERROR] mysql-connector-python not installed.")
        print("        Run: pip install mysql-connector-python")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot connect to MySQL: {e}")
        print(f"        Check your config.env credentials.")
        sys.exit(1)


def _sqlite_connection():
    """
    SQLite connection wrapped so cursor(dictionary=True) works
    and ON DUPLICATE KEY UPDATE is handled via INSERT OR REPLACE.
    """
    import sqlite3

    class DictCursor:
        """Makes SQLite cursor behave like MySQL dictionary cursor."""
        def __init__(self, conn):
            self._conn   = conn
            self._cursor = conn.cursor()

        def execute(self, sql, params=None):
            # Translate MySQL-specific SQL to SQLite
            sql = _mysql_to_sqlite(sql)
            if params:
                self._cursor.execute(sql, params)
            else:
                self._cursor.execute(sql)

        def executemany(self, sql, params_list):
            sql = _mysql_to_sqlite(sql)
            self._cursor.executemany(sql, params_list)

        def fetchone(self):
            row = self._cursor.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in self._cursor.description]
            return dict(zip(cols, row))

        def fetchall(self):
            rows = self._cursor.fetchall()
            if not rows:
                return []
            cols = [d[0] for d in self._cursor.description]
            return [dict(zip(cols, row)) for row in rows]

        def close(self):
            self._cursor.close()

        @property
        def lastrowid(self):
            return self._cursor.lastrowid

    class SQLiteConn:
        """Wraps sqlite3 connection to look like MySQL connection."""
        def __init__(self, path):
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON")

        def cursor(self, dictionary=False):
            # always return dict cursor (ignore dictionary flag)
            return DictCursor(self._conn)

        def commit(self):
            self._conn.commit()

        def rollback(self):
            self._conn.rollback()

        def close(self):
            self._conn.close()

        def execute(self, sql, params=None):
            """Direct execute for init_db table creation."""
            sql = _mysql_to_sqlite(sql)
            if params:
                return self._conn.execute(sql, params)
            return self._conn.execute(sql)

    return SQLiteConn(_DB_PATH)


def _mysql_to_sqlite(sql: str) -> str:
    """
    Translate MySQL-specific SQL syntax to SQLite-compatible syntax.
    Handles the most common differences.
    """
    import re

    # %s → ? (parameter placeholder)
    sql = sql.replace('%s', '?')

    # ON DUPLICATE KEY UPDATE ... → INSERT OR REPLACE
    # (simplified — works for our upsert patterns)
    sql = re.sub(
        r'INSERT INTO (\w+)',
        lambda m: f'INSERT OR REPLACE INTO {m.group(1)}',
        sql,
        count=1,
        flags=re.IGNORECASE
    ) if 'ON DUPLICATE KEY UPDATE' in sql.upper() else sql
    sql = re.sub(
        r'\s+ON DUPLICATE KEY UPDATE.*',
        '',
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    # AUTO_INCREMENT → AUTOINCREMENT
    sql = sql.replace('AUTO_INCREMENT', 'AUTOINCREMENT')

    # ENGINE=InnoDB ... → (remove)
    sql = re.sub(r'ENGINE=\w+[^;]*', '', sql)

    # DEFAULT CHARSET=... → (remove)
    sql = re.sub(r'DEFAULT CHARSET=\w+[^;]*', '', sql)

    # COLLATE utf8mb4_unicode_ci → (remove)
    sql = re.sub(r'COLLATE \w+', '', sql)

    # TINYINT(1) → INTEGER
    sql = sql.replace('TINYINT(1)', 'INTEGER')
    sql = re.sub(r'TINYINT\(\d+\)', 'INTEGER', sql)

    # BIGINT → INTEGER
    sql = re.sub(r'BIGINT\s+AUTO_INCREMENT', 'INTEGER', sql)
    sql = re.sub(r'BIGINT', 'INTEGER', sql)

    # VARCHAR(n) → TEXT
    sql = re.sub(r'VARCHAR\(\d+\)', 'TEXT', sql)

    # LONGTEXT → TEXT
    sql = sql.replace('LONGTEXT', 'TEXT')

    # DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    # → DATETIME DEFAULT CURRENT_TIMESTAMP (SQLite doesn't support ON UPDATE)
    sql = re.sub(r'ON UPDATE CURRENT_TIMESTAMP', '', sql)

    # INDEX idx_... (...) inside CREATE TABLE → (remove inline indexes)
    sql = re.sub(r',?\s*INDEX \w+\s*\([^)]+\)', '', sql)

    # ROUNDEDCORNERS → (remove)
    sql = re.sub(r"'ROUNDEDCORNERS',?\s*\[\d+\]", '', sql)

    # Remove trailing commas before closing paren
    sql = re.sub(r',(\s*\))', r'\1', sql)

    return sql


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED EXECUTE HELPER
# ════════════════════════════════════════════════════════════════════════════

def execute(sql: str, params=None, fetch: str = 'none'):
    """
    Run any query against whichever DB is configured.
    fetch: 'none' | 'one' | 'all'
    """
    conn   = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(sql, params or ())
        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch == 'all':
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        log.error(f"[DB] Query failed: {e}\nSQL: {sql[:200]}")
        raise
    finally:
        cursor.close()
        conn.close()


# ════════════════════════════════════════════════════════════════════════════
# TABLE CREATION
# ════════════════════════════════════════════════════════════════════════════

def init_db():
    """Create all tables. Safe to run multiple times."""
    conn   = get_connection()
    cursor = conn.cursor()

    tables = [
        """
        CREATE TABLE IF NOT EXISTS app_usage (
            id            BIGINT       AUTO_INCREMENT PRIMARY KEY,
            timestamp     DATETIME     NOT NULL,
            date          DATE         NOT NULL,
            hour          TINYINT      NOT NULL,
            app_name      VARCHAR(255) NOT NULL,
            window_title  TEXT,
            duration_secs INT          NOT NULL DEFAULT 0,
            category      VARCHAR(50)  DEFAULT 'other'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_summary (
            id                      BIGINT   AUTO_INCREMENT PRIMARY KEY,
            date                    DATE     UNIQUE NOT NULL,
            total_screen_time_mins  FLOAT    DEFAULT 0,
            active_time_mins        FLOAT    DEFAULT 0,
            idle_time_mins          FLOAT    DEFAULT 0,
            social_media_mins       FLOAT    DEFAULT 0,
            work_app_mins           FLOAT    DEFAULT 0,
            entertainment_mins      FLOAT    DEFAULT 0,
            browser_mins            FLOAT    DEFAULT 0,
            late_night_usage_mins   FLOAT    DEFAULT 0,
            session_count           INT      DEFAULT 0,
            longest_session_mins    FLOAT    DEFAULT 0,
            break_count             INT      DEFAULT 0,
            keystrokes_count        INT      DEFAULT 0,
            mouse_distance_px       FLOAT    DEFAULT 0,
            pc_on_time              DATETIME,
            pc_off_time             DATETIME,
            plugged_in_mins         FLOAT    DEFAULT 0,
            created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP
                                             ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS diary_entries (
            id           BIGINT   AUTO_INCREMENT PRIMARY KEY,
            date         DATE     UNIQUE NOT NULL,
            entry_text   LONGTEXT NOT NULL,
            word_count   INT      DEFAULT 0,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id                  BIGINT      AUTO_INCREMENT PRIMARY KEY,
            date                DATE        UNIQUE NOT NULL,
            predicted_state     VARCHAR(50) NOT NULL,
            confidence          FLOAT       DEFAULT 0,
            normal_score        FLOAT       DEFAULT 0,
            anxiety_score       FLOAT       DEFAULT 0,
            depression_score    FLOAT       DEFAULT 0,
            stress_score        FLOAT       DEFAULT 0,
            bipolar_score       FLOAT       DEFAULT 0,
            suicidal_score      FLOAT       DEFAULT 0,
            personality_score   FLOAT       DEFAULT 0,
            text_weight         FLOAT       DEFAULT 0.5,
            numeric_weight      FLOAT       DEFAULT 0.5,
            created_at          DATETIME    DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS fl_history (
            id              BIGINT      AUTO_INCREMENT PRIMARY KEY,
            timestamp       DATETIME    DEFAULT CURRENT_TIMESTAMP,
            round_number    INT         DEFAULT 0,
            loss_before     FLOAT,
            loss_after      FLOAT,
            upload_success  TINYINT(1)  DEFAULT 0,
            model_version   VARCHAR(50)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS positive_content (
            id              BIGINT      AUTO_INCREMENT PRIMARY KEY,
            date            DATE        NOT NULL,
            content_type    VARCHAR(50) NOT NULL,
            content_text    TEXT        NOT NULL,
            was_helpful     TINYINT(1),
            created_at      DATETIME    DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    ]

    for sql in tables:
        cursor.execute(sql)

    conn.commit()
    cursor.close()
    conn.close()

    backend = "MySQL" if _USE_MYSQL else "SQLite"
    print(f"[DB] All tables ready ({backend})")


# ════════════════════════════════════════════════════════════════════════════
# APP USAGE
# ════════════════════════════════════════════════════════════════════════════

def insert_app_usage(timestamp, app_name, window_title, duration_secs, category):
    dt = datetime.fromisoformat(timestamp)
    execute("""
        INSERT INTO app_usage
            (timestamp, date, hour, app_name, window_title, duration_secs, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (timestamp, str(dt.date()), dt.hour, app_name, window_title,
          duration_secs, category))


def get_app_usage_today():
    today = datetime.now().date()
    return execute("""
        SELECT app_name, category, SUM(duration_secs) AS total_secs
        FROM app_usage WHERE date = %s
        GROUP BY app_name, category
        ORDER BY total_secs DESC
    """, (str(today),), fetch='all') or []


def get_category_totals_today():
    today = datetime.now().date()
    rows  = execute("""
        SELECT category, SUM(duration_secs) AS secs
        FROM app_usage WHERE date = %s
        GROUP BY category
    """, (str(today),), fetch='all') or []
    return {r['category']: r['secs'] / 60 for r in rows}


# ════════════════════════════════════════════════════════════════════════════
# DAILY SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def upsert_daily_summary(date, **kwargs):
    existing = execute(
        "SELECT id FROM daily_summary WHERE date = %s", (str(date),), fetch='one'
    )
    if existing:
        set_clause = ', '.join(f"{k} = %s" for k in kwargs)
        execute(f"UPDATE daily_summary SET {set_clause} WHERE date = %s",
                list(kwargs.values()) + [str(date)])
    else:
        kwargs['date'] = str(date)
        cols = ', '.join(kwargs.keys())
        ph   = ', '.join('%s' for _ in kwargs)
        execute(f"INSERT INTO daily_summary ({cols}) VALUES ({ph})",
                list(kwargs.values()))


def get_daily_summary(date):
    return execute(
        "SELECT * FROM daily_summary WHERE date = %s", (str(date),), fetch='one'
    )


def get_weekly_summaries(end_date=None):
    if end_date is None:
        end_date = datetime.now().date()
    return execute("""
        SELECT * FROM daily_summary
        WHERE date <= %s ORDER BY date DESC LIMIT 7
    """, (str(end_date),), fetch='all') or []


# ════════════════════════════════════════════════════════════════════════════
# DIARY
# ════════════════════════════════════════════════════════════════════════════

def save_diary_entry(date, text):
    execute("""
        INSERT INTO diary_entries (date, entry_text, word_count)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            entry_text = VALUES(entry_text),
            word_count = VALUES(word_count)
    """, (str(date), text, len(text.split())))


def get_diary_entry(date):
    return execute(
        "SELECT * FROM diary_entries WHERE date = %s", (str(date),), fetch='one'
    )


def get_recent_diary_entries(days=7):
    return execute("""
        SELECT * FROM diary_entries ORDER BY date DESC LIMIT %s
    """, (days,), fetch='all') or []


# ════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════

def save_prediction(date, predicted_state, confidence, scores: dict,
                    text_weight=0.5, numeric_weight=0.5):
    execute("""
        INSERT INTO predictions
            (date, predicted_state, confidence,
             normal_score, anxiety_score, depression_score, stress_score,
             bipolar_score, suicidal_score, personality_score,
             text_weight, numeric_weight)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            predicted_state   = VALUES(predicted_state),
            confidence        = VALUES(confidence),
            normal_score      = VALUES(normal_score),
            anxiety_score     = VALUES(anxiety_score),
            depression_score  = VALUES(depression_score),
            stress_score      = VALUES(stress_score),
            bipolar_score     = VALUES(bipolar_score),
            suicidal_score    = VALUES(suicidal_score),
            personality_score = VALUES(personality_score),
            text_weight       = VALUES(text_weight),
            numeric_weight    = VALUES(numeric_weight)
    """, (
        str(date), predicted_state, confidence,
        scores.get('Normal', 0),      scores.get('Anxiety', 0),
        scores.get('Depression', 0),  scores.get('Stress', 0),
        scores.get('Bipolar', 0),     scores.get('Suicidal', 0),
        scores.get('Personality Disorder', 0),
        text_weight, numeric_weight
    ))


def get_predictions(days=7):
    return execute("""
        SELECT * FROM predictions ORDER BY date DESC LIMIT %s
    """, (days,), fetch='all') or []


# ════════════════════════════════════════════════════════════════════════════
# FL HISTORY
# ════════════════════════════════════════════════════════════════════════════

def log_fl_round(round_number, loss_before, upload_success, model_version=None):
    execute("""
        INSERT INTO fl_history (round_number, loss_before, upload_success, model_version)
        VALUES (%s, %s, %s, %s)
    """, (round_number, loss_before, 1 if upload_success else 0, model_version))


def already_uploaded_today():
    today = datetime.now().date()
    return execute("""
        SELECT id FROM fl_history
        WHERE DATE(timestamp) = %s AND upload_success = 1
    """, (str(today),), fetch='one') is not None


# ════════════════════════════════════════════════════════════════════════════
# POSITIVE CONTENT
# ════════════════════════════════════════════════════════════════════════════

def save_positive_content(date, content_type, content_text):
    execute("""
        INSERT INTO positive_content (date, content_type, content_text)
        VALUES (%s, %s, %s)
    """, (str(date), content_type, content_text))


def rate_positive_content(content_id, was_helpful: bool):
    execute(
        "UPDATE positive_content SET was_helpful = %s WHERE id = %s",
        (1 if was_helpful else 0, content_id)
    )


def get_content_preference_stats():
    return execute("""
        SELECT content_type,
               COUNT(*) AS shown,
               SUM(was_helpful) AS helpful
        FROM positive_content
        WHERE was_helpful IS NOT NULL
        GROUP BY content_type
    """, fetch='all') or []


# ════════════════════════════════════════════════════════════════════════════
# STATUS
# ════════════════════════════════════════════════════════════════════════════

def print_status():
    backend = "MySQL" if _USE_MYSQL else "SQLite"
    print(f"\n===== DATABASE STATUS ({backend}) =====")
    for table in ['app_usage', 'daily_summary', 'diary_entries',
                  'predictions', 'fl_history', 'positive_content']:
        try:
            row = execute(f"SELECT COUNT(*) AS cnt FROM {table}", fetch='one')
            print(f"  {table:<25} {row['cnt']:>6} rows")
        except Exception as e:
            print(f"  {table:<25} ERROR: {e}")
    print("=" * 40)


def which_db() -> str:
    """Return which database backend is being used."""
    return "MySQL" if _USE_MYSQL else "SQLite"


if __name__ == '__main__':
    print(f"Database backend: {which_db()}")
    print("Initialising tables...")
    init_db()
    print_status()
