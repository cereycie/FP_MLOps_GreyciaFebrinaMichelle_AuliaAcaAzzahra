import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

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
from api.schemas import RiskScoreResponse

MODELS_DIR = PROJECT_ROOT / "models"
EVENTS_PATH = PROJECT_ROOT / "data" / "events_scored.csv"

model = None
model_meta = None
events = None


def _load_state():
    global model, model_meta, events
    try:
        model, model_meta = load_champion(models_dir=MODELS_DIR)
    except FileNotFoundError as e:
        raise RuntimeError(
            "champion.joblib tidak ditemukan di folder models/. "
            "Jalankan notebook training dulu sebelum menjalankan API ini."
        ) from e
    events = pd.read_csv(EVENTS_PATH, parse_dates=["Datetime"])


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

    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    hour = datetime_.hour if datetime_.tzinfo is None else datetime_.astimezone(REFERENCE_TZ).hour
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))

    reference_date = datetime.now(timezone.utc)
    nearby_crime_count = get_nearby_crime_count(lat_r, lon_r, events, reference_date)

    if nearby_crime_count == 0:
        return RiskScoreResponse(
            risk_score=0.0,
            level=categorize_level(0.0, False),
            model_version=f"v{model_meta['version']}",
            last_updated=model_meta["updated_at"][:10],
        )

    features = pd.DataFrame([{
        "lat_r": lat_r, "lon_r": lon_r,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "nearby_crime_count": nearby_crime_count,
    }])[model_meta["feature_cols"]]

    prediction = float(model.predict(features)[0])
    prediction = max(0.0, min(100.0, prediction))

    recent_incident = has_recent_nearby_incident(lat_r, lon_r, events, reference_date)
    level = categorize_level(prediction, recent_incident)

    return RiskScoreResponse(
        risk_score=round(prediction, 2),
        level=level,
        model_version=f"v{model_meta['version']}",
        last_updated=model_meta["updated_at"][:10],
    )
