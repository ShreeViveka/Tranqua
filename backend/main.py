"""
main.py — FastAPI Backend Server
==================================
Central API that connects:
  - MySQL database (via collector/db.py)
  - GRU Fusion Model (via model/predictor.py)
  - React Frontend (via CORS-enabled REST endpoints)
  - Federated Learning server endpoints

Run with:
  cd D:\\nlp\\mental_health_app
  uvicorn backend.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (auto-generated API docs)
"""

import os
import sys
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'collector'))
sys.path.insert(0, os.path.join(ROOT, 'model'))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(ROOT, 'data', 'backend.log'),
            encoding='utf-8'
        )
    ]
)
log = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Mental Health Tracker API",
    description = "Personal AI diary with federated learning",
    version     = "1.0.0",
    docs_url    = "/docs",
)

# ── CORS — allow React frontend at localhost:3000 ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Lazy-load predictor (loads model only on first prediction request) ─────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from predictor import MentalHealthPredictor
        _predictor = MentalHealthPredictor()
        log.info("[API] Predictor loaded.")
    return _predictor


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class DiaryEntryRequest(BaseModel):
    text : str  = Field(..., min_length=10,
                        description="Diary entry text (min 10 characters)")
    date : Optional[str] = Field(None,
                        description="Date in YYYY-MM-DD format (default: today)")

class DiaryEntryResponse(BaseModel):
    date            : str
    saved           : bool
    word_count      : int

class PredictionRequest(BaseModel):
    date : Optional[str] = Field(None,
                        description="Date to predict for (default: today)")

class RateContentRequest(BaseModel):
    content_id  : int
    was_helpful : bool

class DiaryUpdateRequest(BaseModel):
    date : str
    text : str


# ════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status"  : "running",
        "app"     : "Mental Health Tracker API",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Check if API, database, and model are all reachable."""
    status = {"api": "ok", "database": "unknown", "model": "unknown"}

    # Check DB
    try:
        from db import get_connection
        conn = get_connection()
        conn.close()
        status["database"] = "ok"
    except Exception as e:
        status["database"] = f"error: {str(e)}"

    # Check model files exist
    model_path = os.path.join(ROOT, 'model', 'saved_model.pt')
    vocab_path = os.path.join(ROOT, 'model', 'vocab.pkl')
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        status["model"] = f"ok ({size_mb:.1f} MB)"
    else:
        status["model"] = "not trained yet - run trainer.py first"

    return status


# ════════════════════════════════════════════════════════════════════════════
# DIARY ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/diary", tags=["Diary"])
def save_diary(req: DiaryEntryRequest):
    """Save or update a diary entry for a given date."""
    from db import save_diary_entry

    entry_date = req.date or str(date.today())

    # Validate date format
    try:
        datetime.strptime(entry_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400,
                            detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        save_diary_entry(entry_date, req.text)
        log.info(f"[API] Diary saved for {entry_date} ({len(req.text.split())} words)")
        return {
            "date"      : entry_date,
            "saved"     : True,
            "word_count": len(req.text.split())
        }
    except Exception as e:
        log.error(f"[API] Failed to save diary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/diary/{entry_date}", tags=["Diary"])
def get_diary(entry_date: str):
    """Get diary entry for a specific date."""
    from db import get_diary_entry

    entry = get_diary_entry(entry_date)
    if not entry:
        return {"date": entry_date, "exists": False, "text": "", "word_count": 0}

    return {
        "date"      : entry_date,
        "exists"    : True,
        "text"      : entry["entry_text"],
        "word_count": entry["word_count"],
        "created_at": str(entry.get("created_at", "")),
        "updated_at": str(entry.get("updated_at", "")),
    }


@app.get("/api/diary", tags=["Diary"])
def get_recent_diary(days: int = 7):
    """Get the last N diary entries."""
    from db import get_recent_diary_entries

    entries = get_recent_diary_entries(days)
    return {
        "entries": [
            {
                "date"      : str(e["date"]),
                "text"      : e["entry_text"],
                "word_count": e["word_count"],
            }
            for e in entries
        ],
        "count": len(entries)
    }


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict", tags=["Prediction"])
def predict(req: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Run mental state prediction for a given date.
    Combines diary text + laptop usage data from MySQL.
    The model must be trained first (run trainer.py).
    """
    from db import get_diary_entry

    entry_date = req.date or str(date.today())

    # Get diary entry
    entry = get_diary_entry(entry_date)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"No diary entry found for {entry_date}. Write your diary first."
        )

    diary_text = entry["entry_text"]

    try:
        predictor   = get_predictor()
        target_date = date.fromisoformat(entry_date)
        result      = predictor.predict(diary_text, target_date)

        log.info(f"[API] Prediction for {entry_date}: "
                 f"{result['predicted_state']} ({result['confidence']:.2%})")

        return {
            "date"           : result["date"],
            "predicted_state": result["predicted_state"],
            "confidence"     : result["confidence"],
            "emoji"          : result["emoji"],
            "color"          : result["color"],
            "scores"         : result["scores"],
            "score_list"     : result["score_list"],
            "text_weight"    : result["text_weight"],
            "num_weight"     : result["num_weight"],
            "daily_content"  : result["daily_content"],
            "concerns"       : result["concerns"],
            "word_count"     : result["word_count"],
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {str(e)}. Run trainer.py first."
        )
    except Exception as e:
        log.error(f"[API] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prediction/{entry_date}", tags=["Prediction"])
def get_prediction(entry_date: str):
    """Get a previously saved prediction for a specific date."""
    from db import get_predictions, get_connection

    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM predictions WHERE date = %s", (entry_date,)
    ) if hasattr(conn, 'execute') else None

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM predictions WHERE date = %s", (entry_date,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        return {"date": entry_date, "exists": False}

    from preprocessor import LABEL_COLORS, LABEL_EMOJI
    state = row["predicted_state"]

    return {
        "exists"         : True,
        "date"           : str(row["date"]),
        "predicted_state": state,
        "confidence"     : row["confidence"],
        "emoji"          : LABEL_EMOJI.get(state, ""),
        "color"          : LABEL_COLORS.get(state, "#888"),
        "normal_score"   : row["normal_score"],
        "anxiety_score"  : row["anxiety_score"],
        "depression_score": row["depression_score"],
        "stress_score"   : row["stress_score"],
        "text_weight"    : row["text_weight"],
        "num_weight"     : row["numeric_weight"],
    }


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD / TRACKER ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard", tags=["Dashboard"])
def get_dashboard():
    """
    Get all data needed for the home screen in one call:
    - Today's diary entry
    - Today's prediction (if available)
    - Today's usage summary
    - Streak count
    """
    from db import get_diary_entry, get_daily_summary, get_predictions
    from predictor import get_daily_content
    from preprocessor import LABEL_COLORS, LABEL_EMOJI

    today      = str(date.today())
    diary      = get_diary_entry(today)
    summary    = get_daily_summary(today)
    predictions= get_predictions(days=1)
    prediction = predictions[0] if predictions and str(predictions[0].get('date','')) == today else None

    # Calculate streak (consecutive days with diary entries)
    streak = _calculate_streak()

    # Build response
    result = {
        "today"     : today,
        "streak"    : streak,
        "has_diary" : diary is not None,
        "diary"     : {
            "text"      : diary["entry_text"] if diary else "",
            "word_count": diary["word_count"]  if diary else 0,
        } if diary else None,
        "prediction": None,
        "usage"     : None,
    }

    if prediction:
        state = prediction["predicted_state"]
        result["prediction"] = {
            "predicted_state": state,
            "confidence"     : prediction["confidence"],
            "emoji"          : LABEL_EMOJI.get(state, ""),
            "color"          : LABEL_COLORS.get(state, "#888"),
        }

    if summary:
        result["usage"] = {
            "screen_time_mins" : summary.get("total_screen_time_mins", 0),
            "social_media_mins": summary.get("social_media_mins", 0),
            "work_mins"        : summary.get("work_app_mins", 0),
            "active_mins"      : summary.get("active_time_mins", 0),
            "break_count"      : summary.get("break_count", 0),
            "keystrokes"       : summary.get("keystrokes_count", 0),
        }

    return result


@app.get("/api/tracker", tags=["Tracker"])
def get_tracker(days: int = 7):
    """
    Get weekly tracker data:
    - Mood history (predictions per day)
    - Usage trends
    - Weekly analysis letter
    """
    from db import get_predictions, get_weekly_summaries
    from predictor import generate_weekly_analysis
    from preprocessor import LABEL_COLORS, LABEL_EMOJI

    predictions = get_predictions(days=days)
    summaries   = get_weekly_summaries()

    # Build mood history for chart
    mood_history = []
    for pred in reversed(predictions):   # chronological order
        state = pred["predicted_state"]
        mood_history.append({
            "date"           : str(pred["date"]),
            "predicted_state": state,
            "confidence"     : pred["confidence"],
            "emoji"          : LABEL_EMOJI.get(state, ""),
            "color"          : LABEL_COLORS.get(state, "#888"),
        })

    # Usage trend
    usage_trend = []
    for s in reversed(summaries):
        usage_trend.append({
            "date"             : str(s["date"]),
            "screen_time_mins" : s.get("total_screen_time_mins", 0) or 0,
            "social_media_mins": s.get("social_media_mins", 0)       or 0,
            "active_mins"      : s.get("active_time_mins", 0)        or 0,
            "late_night_mins"  : s.get("late_night_usage_mins", 0)   or 0,
        })

    # Weekly analysis
    weekly = generate_weekly_analysis(predictions, summaries)

    return {
        "mood_history" : mood_history,
        "usage_trend"  : usage_trend,
        "weekly"       : weekly,
        "days"         : days,
    }


@app.get("/api/usage/today", tags=["Usage"])
def get_today_usage():
    """Get today's app usage breakdown."""
    from db import get_app_usage_today, get_daily_summary
    from categories import get_category_display_name
    from feature_extractor import compute_derived_features

    today    = str(date.today())
    apps     = get_app_usage_today()
    summary  = get_daily_summary(today)

    # Group by category
    categories = {}
    for app in apps:
        cat  = app["category"]
        name = get_category_display_name(cat)
        if name not in categories:
            categories[name] = {"mins": 0, "apps": []}
        categories[name]["mins"]  += round(app["total_secs"] / 60, 1)
        categories[name]["apps"].append(app["app_name"])

    # Top apps
    top_apps = sorted(apps, key=lambda x: x["total_secs"], reverse=True)[:5]

    derived = compute_derived_features(summary) if summary else {}

    return {
        "date"      : today,
        "categories": categories,
        "top_apps"  : [
            {
                "name"    : a["app_name"],
                "category": a["category"],
                "mins"    : round(a["total_secs"] / 60, 1),
            }
            for a in top_apps
        ],
        "summary"   : {
            "total_screen_mins": summary.get("total_screen_time_mins", 0) if summary else 0,
            "active_mins"      : summary.get("active_time_mins", 0)       if summary else 0,
            "idle_mins"        : summary.get("idle_time_mins", 0)         if summary else 0,
            "break_count"      : summary.get("break_count", 0)            if summary else 0,
            "keystrokes"       : summary.get("keystrokes_count", 0)       if summary else 0,
        },
        "insights"  : derived,
    }


# ════════════════════════════════════════════════════════════════════════════
# POSITIVE CONTENT ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/content/today", tags=["Content"])
def get_today_content():
    """Get today's positive content (quote/exercise)."""
    from db import get_connection

    today = str(date.today())
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM positive_content
            WHERE date = %s ORDER BY created_at DESC LIMIT 1
        """, (today,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        # No prediction done yet — return a default
        from predictor import get_daily_content
        content = get_daily_content("Normal")
        return {"date": today, "exists": False, **content}

    return {
        "date"        : today,
        "exists"      : True,
        "id"          : row["id"],
        "type"        : row["content_type"],
        "text"        : row["content_text"],
        "was_helpful" : row["was_helpful"],
    }


@app.post("/api/content/rate", tags=["Content"])
def rate_content(req: RateContentRequest):
    """Rate whether today's positive content was helpful."""
    from db import rate_positive_content
    rate_positive_content(req.content_id, req.was_helpful)
    return {"rated": True, "content_id": req.content_id, "was_helpful": req.was_helpful}


# ════════════════════════════════════════════════════════════════════════════
# FEDERATED LEARNING ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/fl/status", tags=["Federated Learning"])
def fl_status():
    """Check federated learning upload status."""
    from db import already_uploaded_today
    from fl_client import should_upload, get_or_create_client_id, has_enough_local_data

    can_upload, reason = should_upload()

    return {
        "uploaded_today"   : already_uploaded_today(),
        "can_upload_now"   : can_upload,
        "reason"           : reason,
        "enough_data"      : has_enough_local_data(),
        "client_id_preview": get_or_create_client_id()[:8] + "...",
    }


@app.post("/api/fl/upload", tags=["Federated Learning"])
def trigger_fl_upload(background_tasks: BackgroundTasks):
    """Manually trigger a federated learning weight upload."""
    from fl_client import should_upload

    can_upload, reason = should_upload()
    if not can_upload:
        return {"triggered": False, "reason": reason}

    background_tasks.add_task(_do_fl_upload)
    return {"triggered": True, "message": "Upload started in background."}


def _do_fl_upload():
    """Background task for FL upload."""
    log.info("[FL] Background upload triggered.")
    # In real deployment: load model weights and call fl_client.upload_weight_update()
    # For now, just log the attempt
    log.info("[FL] Upload would happen here with real model weights.")


# Server-side FL aggregation endpoint (for Railway deployment)
@app.post("/fl/upload", tags=["FL Server"])
async def receive_weight_update(payload: dict):
    """
    Receive weight updates from clients.
    This endpoint runs on the Railway server, not locally.
    """
    client_id    = payload.get("client_id", "unknown")
    round_number = payload.get("round_number", 0)
    log.info(f"[FL Server] Received update from {client_id[:8]}... round {round_number}")

    # In real FL: aggregate weights using FedAvg
    # For now: acknowledge receipt
    return {
        "received"      : True,
        "client_id"     : client_id[:8] + "...",
        "round"         : round_number + 1,
        "model_version" : "1.0.0",
        "message"       : "Weight update received. Thank you for contributing!"
    }


@app.get("/fl/model", tags=["FL Server"])
def get_global_model():
    """Serve the global model to clients."""
    model_path = os.path.join(ROOT, 'model', 'saved_model.pt')
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Global model not available yet.")
    return {"model_version": "1.0.0", "message": "Use /fl/model/download for the actual file."}


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _calculate_streak() -> int:
    """Count consecutive days with diary entries ending today."""
    from db import get_connection
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT date FROM diary_entries ORDER BY date DESC LIMIT 30"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception:
        return 0

    if not rows:
        return 0

    streak    = 0
    check_day = date.today()
    for row in rows:
        row_date = row["date"]
        if isinstance(row_date, str):
            row_date = date.fromisoformat(str(row_date))
        if hasattr(row_date, 'date'):
            row_date = row_date.date()
        if row_date == check_day:
            streak   += 1
            check_day -= timedelta(days=1)
        else:
            break
    return streak




# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/report", tags=["Report"])
def download_report(period: str = "week"):
    """
    Download a PDF mental health report.
    period: week or month
    pip install reportlab  (required)
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, 'backend'))
        from report import generate_pdf_report
        pdf_bytes = generate_pdf_report(period)
        filename  = f"serenity_report_{period}_{date.today()}.pdf"
        return Response(
            content     = pdf_bytes,
            media_type  = "application/pdf",
            headers     = {"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except ImportError:
        raise HTTPException(
            status_code = 503,
            detail      = "reportlab not installed. Run: pip install reportlab"
        )
    except Exception as e:
        log.error(f"[API] PDF report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    log.info("Starting Mental Health Tracker API...")
    uvicorn.run(
        "main:app",
        host    = "127.0.0.1",
        port    = 8000,
        reload  = True,
        workers = 1,
    )
