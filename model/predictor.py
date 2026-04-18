"""
predictor.py — Daily Mental State Predictor
=============================================
Uses the trained GRU + Fusion model to predict mental state
from a diary entry + today's laptop usage data from MySQL.

Called by:
  - The FastAPI backend (/api/predict endpoint)
  - Can also be run directly for testing

Usage:
  python model/predictor.py --text "I felt really anxious today..."
  python model/predictor.py --date 2026-04-14
"""

import os
import sys
import json
import argparse
import logging
from datetime import date, datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'collector'))

from preprocessor import (
    Vocabulary, text_to_ids, IDX2LABEL, LABEL2IDX,
    LABELS, LABEL_COLORS, LABEL_EMOJI,
    VOCAB_PATH, CONFIG_PATH, load_config
)
from model import FusionModel, load_model, DEVICE
from feature_extractor import (
    extract_features_from_db, extract_features_from_summary,
    get_feature_importance_for_prediction, compute_derived_features,
    FEATURE_NAMES
)

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model.pt')


# ════════════════════════════════════════════════════════════════════════════
# CONTENT GENERATOR — Positive Daily Content
# ════════════════════════════════════════════════════════════════════════════

POSITIVE_CONTENT = {
    'Normal': {
        'quotes': [
            "You're doing great — keep showing up for yourself every day. 🌱",
            "Today's calm is a gift. Breathe it in fully. 😊",
            "You handled today well. That matters more than you know.",
            "Being okay is worth celebrating too. 🌟",
        ],
        'exercises': [
            "Take a 10-minute walk outside — fresh air boosts mood even on good days.",
            "Write down 3 things you're grateful for today.",
            "Reach out to someone you haven't spoken to in a while.",
        ],
        'type': 'quote'
    },
    'Anxiety': {
        'quotes': [
            "Anxiety lies. You are safe right now, in this moment. 🌊",
            "One breath at a time. You don't have to solve everything today.",
            "Your feelings are valid. You are stronger than your fears.",
            "This anxious feeling will pass — it always does. ☁️➡️☀️",
        ],
        'exercises': [
            "Try box breathing: Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4 times.",
            "Ground yourself: Name 5 things you can see, 4 you can touch, 3 you can hear.",
            "Write down what's worrying you — then write what you can actually control.",
            "Limit social media for the next 2 hours. Your mind needs a break.",
        ],
        'type': 'exercise'
    },
    'Stress': {
        'quotes': [
            "You've survived 100% of your hardest days so far. 💪",
            "Progress, not perfection. You're doing enough.",
            "Rest is not a reward — it's a necessity. Give yourself permission.",
            "Stressed is just desserts spelled backwards. 😄 Take a breath.",
        ],
        'exercises': [
            "Take a 5-minute break RIGHT NOW. Step away from the screen.",
            "Progressive muscle relaxation: Tense each muscle group for 5s, then release.",
            "Make a to-do list and cross off 1 item — just 1. That's enough for now.",
            "Drink a full glass of water and stretch your neck and shoulders.",
        ],
        'type': 'exercise'
    },
    'Depression': {
        'quotes': [
            "Even getting out of bed today was an act of courage. 🌅",
            "You don't have to feel better today. Just stay.",
            "The darkest nights produce the brightest stars. You will get through this. ✨",
            "Small steps still move you forward. One moment at a time.",
        ],
        'exercises': [
            "Go outside for just 5 minutes — sunlight genuinely helps serotonin.",
            "Text or call one person today. Connection, even brief, matters.",
            "Eat something nourishing. Your brain needs fuel.",
            "Make your bed — it's one small act of care for yourself.",
        ],
        'type': 'both'
    },
    'Bipolar': {
        'quotes': [
            "Your mind is more complex, not more broken. 🧠",
            "Stability is a journey, not a destination. You're on the right path.",
            "Track your patterns — understanding yourself is power.",
        ],
        'exercises': [
            "Log your mood and sleep tonight — patterns over time are your superpower.",
            "Stick to your sleep routine tonight — consistency is key for balance.",
            "If energy is high today, channel it into something creative for 30 minutes.",
        ],
        'type': 'exercise'
    },
    'Suicidal': {
        'quotes': [
            "You reaching out and writing today matters. Please keep going. 💙",
            "This pain is real, but it is not permanent. Help is available.",
            "You are not alone in this moment, even if it feels that way.",
        ],
        'exercises': [
            "Please talk to someone right now — iCall: 9152987821 (India) | Crisis Text: Text HOME to 741741",
            "Go to a place where other people are — a cafe, library, anywhere with others.",
            "Tell one trusted person how you're feeling today.",
        ],
        'type': 'crisis'
    },
    'Personality Disorder': {
        'quotes': [
            "Your emotions are intense because you feel deeply. That is also your gift.",
            "Progress in small steps is still progress. Be patient with yourself.",
            "You are not your diagnosis. You are the whole person behind it. 🌈",
        ],
        'exercises': [
            "Practice the STOP skill: Stop → Take a breath → Observe → Proceed mindfully.",
            "Write about one emotion you felt today without judging it — just describe it.",
            "Do one self-care activity you've been putting off.",
        ],
        'type': 'exercise'
    },
}


def get_daily_content(predicted_state: str, user_preferences: dict = None) -> dict:
    """
    Get personalised positive content based on predicted state.
    Considers user's past preferences (what they rated as helpful).
    """
    content_pool = POSITIVE_CONTENT.get(predicted_state, POSITIVE_CONTENT['Normal'])

    import random

    # Determine content type based on preferences or default
    content_type = content_pool.get('type', 'quote')
    if user_preferences:
        # Use what worked for this user before
        helpful_types = [k for k, v in user_preferences.items() if v > 50]
        if helpful_types:
            content_type = helpful_types[0]

    if content_type == 'exercise' and content_pool.get('exercises'):
        content_text = random.choice(content_pool['exercises'])
        ctype        = 'exercise'
    elif content_type == 'crisis':
        content_text = content_pool['exercises'][0]   # always show crisis resource first
        ctype        = 'crisis'
    else:
        content_text = random.choice(content_pool['quotes'])
        ctype        = 'quote'

    return {
        'type'   : ctype,
        'text'   : content_text,
        'state'  : predicted_state,
    }


# ════════════════════════════════════════════════════════════════════════════
# WEEKLY ANALYSIS GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_weekly_analysis(predictions: list[dict], summaries: list[dict]) -> dict:
    """
    Generate the weekly 'Letter to Yourself' and analysis.

    predictions : last 7 days of prediction rows from MySQL
    summaries   : last 7 days of daily_summary rows from MySQL
    """
    if not predictions:
        return {'available': False, 'message': 'Not enough data yet for weekly analysis.'}

    # Count states this week
    states      = [p['predicted_state'] for p in predictions]
    state_counts= {s: states.count(s) for s in set(states)}
    dominant    = max(state_counts, key=state_counts.get)

    # Trend — compare first half vs second half of week
    if len(states) >= 4:
        first_half  = states[:len(states)//2]
        second_half = states[len(states)//2:]
        positive    = {'Normal'}
        first_pos   = sum(1 for s in first_half  if s in positive)
        second_pos  = sum(1 for s in second_half if s in positive)
        if second_pos > first_pos:
            trend = 'improving'
        elif second_pos < first_pos:
            trend = 'declining'
        else:
            trend = 'stable'
    else:
        trend = 'stable'

    # Average screen time
    avg_screen = (
        sum(s.get('total_screen_time_mins', 0) or 0 for s in summaries) /
        max(len(summaries), 1)
    )
    avg_social = (
        sum(s.get('social_media_mins', 0) or 0 for s in summaries) /
        max(len(summaries), 1)
    )

    # Generate the weekly letter
    letter = _generate_weekly_letter(dominant, trend, state_counts,
                                     avg_screen, avg_social)

    return {
        'available'    : True,
        'dominant_state': dominant,
        'trend'        : trend,
        'state_counts' : state_counts,
        'avg_screen_time_mins': round(avg_screen, 1),
        'avg_social_media_mins': round(avg_social, 1),
        'weekly_letter': letter,
        'days_analysed': len(predictions),
    }


def _generate_weekly_letter(dominant: str, trend: str,
                             state_counts: dict, avg_screen: float,
                             avg_social: float) -> str:
    """Generate a warm, personal weekly summary letter."""

    trend_text = {
        'improving': "and you seem to be moving in a better direction as the week went on",
        'declining': "though the week got harder towards the end — that's okay, every week is different",
        'stable'   : "with a fairly consistent pattern throughout the week"
    }.get(trend, "")

    screen_insight = ""
    if avg_screen > 360:
        screen_insight = (f"Your screen time averaged {avg_screen:.0f} minutes a day this week. "
                         "Taking a few more breaks might help you feel a bit more grounded.")
    elif avg_screen < 120:
        screen_insight = "You had relatively low screen time this week — well done for taking breaks."

    social_insight = ""
    if avg_social > 90:
        social_insight = (f"You spent about {avg_social:.0f} minutes per day on social media. "
                         "It might be worth noticing how it affects your mood.")

    dominant_msg = {
        'Normal'    : "This was largely a stable week for you",
        'Anxiety'   : "This was an anxious week for you — your feelings are valid",
        'Stress'    : "This was a stressful week — you carried a lot",
        'Depression': "This was a heavy week — thank you for showing up anyway",
        'Bipolar'   : "This was a complex week emotionally",
        'Suicidal'  : "This was an extremely difficult week — please reach out for support",
        'Personality Disorder': "This was an emotionally intense week",
    }.get(dominant, "This was another week")

    letter = f"""Dear You,

{dominant_msg}, {trend_text}.

{screen_insight} {social_insight}

You wrote in your diary {sum(state_counts.values())} time(s) this week — that act of checking in with yourself matters more than you might think.

{"You're doing better than you realize." if trend == 'improving' else "Next week is a fresh start." if trend == 'declining' else "Consistency is its own kind of strength."}

Take care of yourself this coming week. You deserve it.

— Your Tracker 🌱
"""
    return letter.strip()


# ════════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR CLASS
# ════════════════════════════════════════════════════════════════════════════

class MentalHealthPredictor:
    """
    Loads the trained model and vocabulary once,
    then makes predictions efficiently on new diary entries.
    """

    def __init__(self):
        self._model  = None
        self._vocab  = None
        self._config = None

    def _load(self):
        """Lazy load — only loads when first prediction is needed."""
        if self._model is None:
            log.info("[Predictor] Loading model and vocabulary...")
            self._config = load_config(CONFIG_PATH)
            self._vocab  = Vocabulary.load(VOCAB_PATH)
            self._model  = load_model(MODEL_PATH)
            log.info("[Predictor] Ready.")

    def predict(self, diary_text: str,
                target_date: date = None,
                numerical_features: np.ndarray = None) -> dict:
        """
        Make a prediction for a diary entry.

        diary_text          : the raw diary text written by the user
        target_date         : date to pull numerical features from MySQL (default: today)
        numerical_features  : optional pre-computed feature vector (overrides DB lookup)

        Returns a rich prediction dict ready for the UI.
        """
        self._load()

        if target_date is None:
            target_date = date.today()

        # ── Text features ──────────────────────────────────────────────────────
        max_len   = self._config.get('max_seq_len', 128)
        token_ids = text_to_ids(diary_text, self._vocab.word2idx, max_len)

        # ── Numerical features ─────────────────────────────────────────────────
        if numerical_features is None:
            numerical_features = extract_features_from_db(target_date)

        # ── Run model ──────────────────────────────────────────────────────────
        result = self._model.predict_single(
            token_ids       = token_ids,
            num_features    = numerical_features.tolist()
        )

        # ── Enrich result ──────────────────────────────────────────────────────
        result['date']          = str(target_date)
        result['word_count']    = len(diary_text.split())
        result['feature_names'] = FEATURE_NAMES
        result['feature_values']= numerical_features.tolist()

        # UI-friendly score list
        result['score_list'] = [
            {
                'label'     : label,
                'score'     : round(result['scores'][label], 4),
                'color'     : LABEL_COLORS[label],
                'emoji'     : LABEL_EMOJI[label],
                'is_top'    : label == result['predicted_state'],
            }
            for label in LABELS
        ]

        # Get daily positive content
        result['daily_content'] = get_daily_content(result['predicted_state'])

        # Get feature concerns
        result['concerns'] = get_feature_importance_for_prediction(numerical_features)

        # Save to MySQL
        self._save_to_db(result, target_date)

        return result

    def _save_to_db(self, result: dict, target_date: date):
        """Save prediction to MySQL."""
        try:
            from db import save_prediction, save_positive_content
            save_prediction(
                date            = str(target_date),
                predicted_state = result['predicted_state'],
                confidence      = result['confidence'],
                scores          = result['scores'],
                text_weight     = result['text_weight'],
                numeric_weight  = result['num_weight'],
            )
            content = result['daily_content']
            save_positive_content(str(target_date), content['type'], content['text'])
            log.info(f"[Predictor] Prediction saved for {target_date}")
        except Exception as e:
            log.warning(f"[Predictor] Could not save to DB: {e}")


# ── Singleton instance ────────────────────────────────────────────────────────
_predictor = MentalHealthPredictor()


def predict(diary_text: str, target_date: date = None,
            numerical_features: np.ndarray = None) -> dict:
    """Module-level convenience function."""
    return _predictor.predict(diary_text, target_date, numerical_features)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str,
                        default="I have been feeling really anxious lately "
                                "and cannot stop worrying about everything.")
    parser.add_argument('--date', type=str, default=None)
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()

    print("\nRunning prediction...")
    result = predict(args.text, target_date)

    print(f"\n{'='*55}")
    print(f"  PREDICTION RESULT")
    print(f"{'='*55}")
    print(f"  Date            : {result['date']}")
    print(f"  Predicted State : {result['emoji']} {result['predicted_state']}")
    print(f"  Confidence      : {result['confidence']:.2%}")
    print(f"  Text weight     : {result['text_weight']:.2%}")
    print(f"  Numeric weight  : {result['num_weight']:.2%}")
    print(f"\n  All Scores:")
    for item in sorted(result['score_list'], key=lambda x: x['score'], reverse=True):
        bar = '█' * int(item['score'] * 20)
        print(f"    {item['emoji']} {item['label']:<22} {item['score']:.2%}  {bar}")
    print(f"\n  Today's Message:")
    print(f"  {result['daily_content']['text']}")
    if result['concerns']:
        print(f"\n  Usage Insights:")
        for c in result['concerns']:
            print(f"  {c['icon']} {c['feature']}: {c['insight']}")
    print(f"{'='*55}\n")
