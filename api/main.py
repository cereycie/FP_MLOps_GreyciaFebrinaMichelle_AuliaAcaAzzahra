import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from registry import load_champion
from pipeline import (
    get_nearby_crime_count,
    categorize_level,
    has_recent_nearby_incident,
    REFERENCE_TZ,
)
from api.schemas import (
    RiskScoreResponse,
    StartJourneyRequest,
    StartJourneyResponse,
    VerifyPinRequest,
    VerifyPinResponse,
    EndJourneyRequest,
    EndJourneyResponse,
    ExtendJourneyRequest,
    ExtendJourneyResponse,
    ForgotPinRequest,
    ForgotPinResponse,
)

MODELS_DIR = PROJECT_ROOT / "models"
EVENTS_PATH = PROJECT_ROOT / "data" / "events_scored.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

model = None
model_meta = None
events = None
TEST_MODE = True

sharelock_sessions = {}

def check_session_expiry(session):
    if datetime.now(timezone.utc) > session["expires_at"]:
        session["status"] = "EXPIRED"
    return session

def _load_state():
    global model, model_meta, events
    try:
        model, model_meta = load_champion(models_dir=MODELS_DIR)
    except FileNotFoundError as e:
        if not TEST_MODE:
            raise RuntimeError(
            "champion.joblib tidak ditemukan di folder models/. "
            "Jalankan notebook training dulu sebelum menjalankan API ini."
        ) from e

    model = None
    model_meta = None

    events = pd.read_csv(EVENTS_PATH, parse_dates=["Datetime"])
    LOG_PATH.parent.mkdir(exist_ok=True)


def log_prediction(entry):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield


app = FastAPI(
    title="Crime Risk Score API",
    description="Serving model prediksi risk_score, FP MLOps Checkpoint 2",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_version": f"v{model_meta['version']}" if model_meta else None,
    }


@app.get("/risk-score", response_model=RiskScoreResponse)
def risk_score(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    datetime_: datetime = Query(..., alias="datetime", description="Waktu query, ISO 8601"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum termuat.")

    request_time = datetime.now(timezone.utc)
    start = time.perf_counter()

    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    hour = datetime_.hour if datetime_.tzinfo is None else datetime_.astimezone(REFERENCE_TZ).hour
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))

    reference_date = datetime.now(timezone.utc)
    nearby_crime_count = get_nearby_crime_count(lat_r, lon_r, events, reference_date)

    if nearby_crime_count == 0:
        response = RiskScoreResponse(
            risk_score=0.0,
            level=categorize_level(0.0, False),
            model_version=f"v{model_meta['version']}",
            last_updated=model_meta["updated_at"][:10],
        )
        log_prediction({
            "timestamp": request_time.isoformat(),
            "lat": lat, "lon": lon, "requested_datetime": datetime_.isoformat(),
            "nearby_crime_count": nearby_crime_count,
            "risk_score": response.risk_score, "level": response.level,
            "model_version": response.model_version,
            "used_model": False,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        return response

    features = pd.DataFrame([{
        "lat_r": lat_r, "lon_r": lon_r,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "nearby_crime_count": nearby_crime_count,
    }])[model_meta["feature_cols"]]

    prediction = float(model.predict(features)[0])
    prediction = max(0.0, min(100.0, prediction))

    recent_incident = has_recent_nearby_incident(lat_r, lon_r, events, reference_date)
    level = categorize_level(prediction, recent_incident)

    response = RiskScoreResponse(
        risk_score=round(prediction, 2),
        level=level,
        model_version=f"v{model_meta['version']}",
        last_updated=model_meta["updated_at"][:10],
    )
    log_prediction({
        "timestamp": request_time.isoformat(),
        "lat": lat, "lon": lon, "requested_datetime": datetime_.isoformat(),
        "nearby_crime_count": nearby_crime_count,
        "risk_score": response.risk_score, "level": response.level,
        "model_version": response.model_version,
        "used_model": True,
        "recent_incident_override": recent_incident,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })
    return response


@app.get("/metrics")
def metrics(last_n: int = 100):
    summary = {
        "current_model_version": f"v{model_meta['version']}" if model_meta else None,
        "current_model_metrics_holdout": model_meta["metrics_holdout"] if model_meta else None,
    }

    registry_path = MODELS_DIR / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry_entries = json.load(f)
        summary["model_version_history"] = [
            {
                "version": e.get("version"),
                "batch": e.get("batch"),
                "decision": e.get("decision"),
                "candidate_MAE": e.get("candidate_metrics", {}).get("MAE"),
            }
            for e in registry_entries
        ]
    else:
        summary["model_version_history"] = []

    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            lines = f.readlines()
        recent = [json.loads(line) for line in lines[-last_n:]]
        summary["total_predictions_logged"] = len(lines)
        summary["predictions_in_window"] = len(recent)
        if recent:
            summary["avg_risk_score_in_window"] = round(
                sum(r["risk_score"] for r in recent) / len(recent), 2
            )
            summary["avg_latency_ms_in_window"] = round(
                sum(r["latency_ms"] for r in recent) / len(recent), 2
            )
            level_counts = {}
            for r in recent:
                level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1
            summary["level_distribution_in_window"] = level_counts
            summary["zero_crime_shortcut_rate_in_window"] = round(
                sum(1 for r in recent if not r["used_model"]) / len(recent), 3
            )
    else:
        summary["total_predictions_logged"] = 0
        summary["predictions_in_window"] = 0

    return summary

@app.post("/sharelock/start", response_model=StartJourneyResponse)
def start_sharelock(request: StartJourneyRequest):
    session_id = str(uuid4())

    started_at = datetime.now(timezone.utc)
    expires_at = started_at + timedelta(
        minutes=request.duration_minutes
    )

    tracking_link = (
        f"http://localhost:8000/sharelock/public/{session_id}"
    )

    sharelock_sessions[session_id] = {
        "session_id": session_id,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "pin": request.pin,
        "status": "ACTIVE",
        "started_at": started_at,
        "expires_at": expires_at,
        "tracking_link": tracking_link,
        "pin_attempt": 0,
        "verified": False,
    }

    return StartJourneyResponse(
        session_id=session_id,
        status="ACTIVE",
        started_at=started_at,
        expires_at=expires_at,
        tracking_link=tracking_link,
    )

@app.get("/sharelock/public/{session_id}")
def public_tracking(session_id: str):
    session = sharelock_sessions.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session Sharelock tidak ditemukan."
        )
    
    session = check_session_expiry(session)

    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "latitude": session["latitude"],
        "longitude": session["longitude"],
        "started_at": session["started_at"],
        "expires_at": session["expires_at"],
    }

@app.post("/sharelock/verify-pin", response_model=VerifyPinResponse)
def verify_sharelock_pin(request: VerifyPinRequest):
    session = sharelock_sessions.get(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session Sharelock tidak ditemukan."
        )

    session = check_session_expiry(session)

    if session["status"] != "ACTIVE":
        return VerifyPinResponse(
            success=False,
            message="Sesi Sharelock sudah tidak aktif."
        )

    if session["pin_attempt"] >= 3:
        return VerifyPinResponse(
            success=False,
            message="Percobaan PIN sudah mencapai batas. Silakan reset PIN."
        )

    # PIN benar
    if request.pin == session["pin"]:
        session["verified"] = True

        return VerifyPinResponse(
            success=True,
            message="PIN berhasil diverifikasi."
        )

    # PIN salah
    session["pin_attempt"] += 1

    if session["pin_attempt"] >= 3:
        return VerifyPinResponse(
            success=False,
            message="PIN salah 3 kali. Silakan lakukan reset PIN."
        )

    remaining_attempts = 3 - session["pin_attempt"]

    return VerifyPinResponse(
        success=False,
        message=f"PIN salah. Sisa percobaan: {remaining_attempts}."
    )

@app.post("/sharelock/end", response_model=EndJourneyResponse)
def end_sharelock(request: EndJourneyRequest):
    session = sharelock_sessions.get(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session Sharelock tidak ditemukan."
        )
    session = check_session_expiry(session)

    if session["status"] != "ACTIVE":
        return EndJourneyResponse(
            success=False,
            status=session["status"],
            message="Sesi Sharelock sudah tidak aktif."
        )

    if request.pin != session["pin"]:
        return EndJourneyResponse(
            success=False,
            status="ACTIVE",
            message="PIN salah. Sesi masih berjalan."
        )

    session["status"] = "SAFE_ARRIVAL"
    session["verified"] = True

    return EndJourneyResponse(
        success=True,
        status="SAFE_ARRIVAL",
        message="Perjalanan berhasil dikonfirmasi. Live location dihentikan."
    )

@app.post("/sharelock/extend", response_model=ExtendJourneyResponse)
def extend_sharelock(request: ExtendJourneyRequest):
    session = sharelock_sessions.get(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session Sharelock tidak ditemukan."
        )

    session = check_session_expiry(session)

    if session["status"] != "ACTIVE":
        return ExtendJourneyResponse(
            success=False,
            status=session["status"],
            expires_at=session["expires_at"],
            message="Sesi Sharelock sudah tidak aktif."
        )

    session["expires_at"] = session["expires_at"] + timedelta(minutes=10)

    return ExtendJourneyResponse(
        success=True,
        status="ACTIVE",
        expires_at=session["expires_at"],
        message="Durasi Sharelock berhasil diperpanjang 10 menit."
    )

@app.post("/sharelock/forgot-pin", response_model=ForgotPinResponse)
def forgot_sharelock_pin(request: ForgotPinRequest):
    session = sharelock_sessions.get(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session Sharelock tidak ditemukan."
        )

    session = check_session_expiry(session)
    
    if session["status"] != "ACTIVE":
        return ForgotPinResponse(
            success=False,
            status=session["status"],
            message="Sesi Sharelock sudah tidak aktif.",
            waiting_timer_paused=False,
        )

    session["waiting_timer_paused"] = True
    session["pin_reset_requested"] = True

    return ForgotPinResponse(
        success=True,
        status="PIN_RESET_REQUIRED",
        message="Silakan lakukan reset PIN melalui email terdaftar.",
        waiting_timer_paused=True,
    )