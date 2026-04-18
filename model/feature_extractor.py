"""
feature_extractor.py — Numerical Feature Engineering
======================================================
Converts the raw daily_summary rows from MySQL into a
normalized feature vector that the NumericalMLP can process.

Also handles:
  - Missing data (if collector wasn't running)
  - Normalization (all values scaled to 0-1)
  - Feature importance weighting for mental health signals
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collector'))

# ── Feature definitions ───────────────────────────────────────────────────────
# Each feature: (column_name, max_value_for_normalization, mental_health_relevance)
# max_value is used to normalize to [0, 1]
# relevance is just documentation — not used in computation

FEATURES = [
    # name                      max_val   notes
    ('total_screen_time_mins',  720,   ),  # 12 hrs = max realistic screen time
    ('social_media_mins',       300,   ),  # 5 hrs social = very high
    ('work_app_mins',           600,   ),  # 10 hrs work
    ('entertainment_mins',      360,   ),  # 6 hrs entertainment
    ('idle_time_mins',          480,   ),  # 8 hrs idle
    ('keystrokes_count',        20000, ),  # typical day: 5k-15k keystrokes
    ('break_count',             20,    ),  # number of 10+ min breaks
    ('late_night_usage_mins',   180,   ),  # 3 hrs late night = concerning
    ('active_time_mins',        600,   ),  # 10 hrs active
    ('mouse_distance_px',       500000,),  # typical day mouse movement
]

FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_MAXES = [f[1] for f in FEATURES]
NUM_FEATURES  = len(FEATURES)

# Path to save normalization stats (fitted on your real data over time)
NORM_STATS_PATH = os.path.join(os.path.dirname(__file__), 'norm_stats.json')


def extract_features_from_summary(summary: dict) -> np.ndarray:
    """
    Convert one daily_summary row (dict from MySQL) into a
    normalized feature vector of shape (NUM_FEATURES,).

    Handles missing values gracefully — uses 0 if data not available.
    """
    vec = []
    for col, max_val in FEATURES:
        raw = summary.get(col, 0) or 0   # None → 0
        # Clip to [0, max_val] then normalize to [0, 1]
        normalized = min(float(raw), max_val) / max_val
        vec.append(normalized)

    return np.array(vec, dtype=np.float32)


def extract_features_from_db(target_date: date = None) -> np.ndarray:
    """
    Pull today's (or a specific date's) summary from MySQL
    and convert it to a feature vector.

    Returns zeros if no data is available for that date.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collector'))
    from db import get_daily_summary

    if target_date is None:
        target_date = date.today()

    summary = get_daily_summary(str(target_date))

    if summary is None:
        print(f"[Features] No data found for {target_date} — using zeros.")
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    return extract_features_from_summary(summary)


def extract_weekly_features(end_date: date = None, days: int = 7) -> np.ndarray:
    """
    Extract features for the past N days and return as a 2D array.
    Shape: (days, NUM_FEATURES)

    The GRU processes this as a time sequence — day by day.
    Most recent day is LAST in the sequence.
    """
    from db import get_weekly_summaries

    if end_date is None:
        end_date = date.today()

    summaries = get_weekly_summaries(str(end_date))

    # summaries are returned most-recent first — reverse to get chronological
    summaries = list(reversed(summaries))

    # Pad with zeros if we have fewer than `days` days of data
    feature_matrix = []
    for i in range(days):
        if i < len(summaries):
            vec = extract_features_from_summary(summaries[i])
        else:
            vec = np.zeros(NUM_FEATURES, dtype=np.float32)
        feature_matrix.append(vec)

    return np.array(feature_matrix, dtype=np.float32)  # (days, NUM_FEATURES)


def compute_derived_features(summary: dict) -> dict:
    """
    Compute higher-level derived signals from raw summary data.
    These are used for the UI insights, not directly fed into the model.
    """
    screen = summary.get('total_screen_time_mins', 0) or 0
    social = summary.get('social_media_mins', 0)       or 0
    work   = summary.get('work_app_mins', 0)           or 0
    idle   = summary.get('idle_time_mins', 0)          or 0
    active = summary.get('active_time_mins', 0)        or 0
    late   = summary.get('late_night_usage_mins', 0)   or 0
    keys   = summary.get('keystrokes_count', 0)        or 0
    breaks = summary.get('break_count', 0)             or 0

    return {
        # What % of screen time was social media?
        'social_ratio'        : round(social / screen * 100, 1) if screen > 0 else 0,

        # What % of screen time was productive?
        'productivity_ratio'  : round(work   / screen * 100, 1) if screen > 0 else 0,

        # Sedentary ratio (idle / total screen)
        'sedentary_ratio'     : round(idle   / screen * 100, 1) if screen > 0 else 0,

        # Was late-night usage significant? (>30 mins)
        'late_night_flag'     : late > 30,

        # Low activity flag (keystrokes very low)
        'low_activity_flag'   : keys < 1000 and screen > 120,

        # Good break pattern (at least 1 break per 2 hrs of screen time)
        'good_break_pattern'  : breaks >= max(1, screen // 120),

        # Screen time category
        'screen_time_level'   : (
            'Low'      if screen < 120 else
            'Moderate' if screen < 300 else
            'High'     if screen < 480 else
            'Very High'
        ),

        # Social media risk level
        'social_risk'         : (
            'Low'    if social < 30  else
            'Medium' if social < 90  else
            'High'   if social < 180 else
            'Very High'
        ),
    }


def get_feature_importance_for_prediction(feature_vec: np.ndarray) -> list[dict]:
    """
    Returns which features are most 'concerning' for the current user.
    Used by the UI to show insights like:
    'Your social media usage is 3x higher than usual today'
    """
    concerns = []

    val_dict = dict(zip(FEATURE_NAMES, feature_vec))

    if val_dict.get('social_media_mins', 0) > 0.5:
        concerns.append({
            'feature': 'Social Media Usage',
            'level'  : 'High',
            'insight': 'High social media time is linked to anxiety and mood dips.',
            'icon'   : '📱'
        })

    if val_dict.get('late_night_usage_mins', 0) > 0.3:
        concerns.append({
            'feature': 'Late Night Screen Time',
            'level'  : 'High',
            'insight': 'Late night usage disrupts sleep and affects next-day mood.',
            'icon'   : '🌙'
        })

    if val_dict.get('idle_time_mins', 0) > 0.6:
        concerns.append({
            'feature': 'Low Activity',
            'level'  : 'Medium',
            'insight': 'Long idle periods may indicate low energy or low motivation.',
            'icon'   : '💤'
        })

    if val_dict.get('keystrokes_count', 0) < 0.1:
        concerns.append({
            'feature': 'Very Low Keyboard Activity',
            'level'  : 'Medium',
            'insight': 'Much lower activity than usual — check in with yourself.',
            'icon'   : '⌨️'
        })

    if val_dict.get('break_count', 0) < 0.1 and val_dict.get('active_time_mins', 0) > 0.5:
        concerns.append({
            'feature': 'No Breaks Taken',
            'level'  : 'Medium',
            'insight': 'Long sessions without breaks increase stress levels.',
            'icon'   : '☕'
        })

    return concerns


if __name__ == '__main__':
    # Test with dummy data
    dummy_summary = {
        'total_screen_time_mins': 360,
        'social_media_mins'     : 120,
        'work_app_mins'         : 90,
        'entertainment_mins'    : 60,
        'idle_time_mins'        : 90,
        'keystrokes_count'      : 5000,
        'break_count'           : 2,
        'late_night_usage_mins' : 45,
        'active_time_mins'      : 270,
        'mouse_distance_px'     : 150000,
    }

    vec = extract_features_from_summary(dummy_summary)
    print("Feature vector:")
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  {name:<30} {val:.4f}")

    derived = compute_derived_features(dummy_summary)
    print("\nDerived features:")
    for k, v in derived.items():
        print(f"  {k:<25} {v}")

    concerns = get_feature_importance_for_prediction(vec)
    print(f"\nConcerns detected: {len(concerns)}")
    for c in concerns:
        print(f"  {c['icon']} {c['feature']} ({c['level']}): {c['insight']}")
