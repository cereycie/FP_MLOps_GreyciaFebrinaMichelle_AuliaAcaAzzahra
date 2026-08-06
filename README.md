# Chicago Crime Risk Score, MLOps Final Project

SISTECH 2026, Machine Learning Operations path, Group 4
Greycia Febrina Michelle & Aulia Aca Azzahra

## Overview

This project builds an end to end Risk Score prediction system for a location and time, based on historical Chicago crime data. It continues directly from Hands-On 1 (pseudo-labeling and feature engineering) and Hands-On 2 (modeling and continual learning), now combined into one system with a model, a REST API, and monitoring.

Risk Score reflects three dimensions at once: how severe past crimes were, how recently they happened, and how close they were to the location being asked about.

Disclaimer: this system produces an estimate based on historical patterns, not a certainty that anything will happen. The dataset is from Chicago, not Indonesia, and is used here to validate the modeling approach until a local dataset becomes available.

## Project Status

- CP1, Pseudo-Labeling and Feature Engineering: done, revised for FE integration (see Changelog)
- CP2, Model Training, Baseline Comparison, REST API Serving: done (see Changelog)
- CP3, Continual Learning, Monitoring and Logging: done (see Changelog). Final report and presentation not started yet

## Changelog

### CP3
- Rebuilt Aca's original Hands-On 2 continual learning loop (drift detection, checkpoint versioning, champion/challenger promotion) against CP1's feature schema instead of Hands-On 2's, since the underlying logic was already sound, only the features under it had changed
- Simulated batches arriving quarterly: 2024 as the initial pool, 2025 Q1 through Q4 as incoming data, 2026 held out entirely
- First drift threshold, reused directly from Hands-On 2 (ks_stat >= 0.02), turned out to sit inside pure sampling noise at CP1's grid size, measured by splitting one quarter randomly in half and checking the KS statistic between the two halves that share the same source. Recalibrated to 0.035 (`nearby_crime_count`) and 0.025 (`risk_raw`), both comfortably above the measured noise ceiling
- First holdout design compared candidates against a narrow future time window while training on an ever-growing cumulative pool, `nearby_crime_count` accumulates without decay so the two were never on the same scale, MAE got worse the more training data was added. Replaced with a fixed spatial holdout (145 of 724 cells) re-evaluated on the same current snapshot the candidate was trained from, so the champion and the candidate are always compared on identical ground
- Promotion margin (0.186) calibrated the same way as CP1's category thresholds, bootstrap resampling the holdout 500 times and requiring the candidate to beat the champion by more than three standard deviations of that resampling noise
- Result: drift was real at every quarterly checkpoint, but only 2 of 4 candidates cleared the promotion margin, the other 2 were kept as checkpoints (`model_v1_rejected_2025Q2.joblib`, `model_v2_rejected_2025Q4.joblib`) but did not replace the champion
- Added `logs/predictions.jsonl`, one line per `/risk-score` call, and a `GET /metrics` endpoint that reads it together with `models/registry.json` to report current model version, full promotion history, and recent prediction activity (average score, average latency, level distribution, zero-crime shortcut rate)

### CP2
- Trained and compared Linear Regression, Random Forest, and Gradient Boosting on `feature_table_fp.csv`, against two baselines (global mean, mean by hour)
- Re-tested Random Forest and Gradient Boosting across four different random seeds for the train/test cell split, Gradient Boosting had the lower MAE in all four
- Added `get_nearby_crime_count` to `src/pipeline.py`, an additive function that computes the same `nearby_crime_count` feature for an arbitrary coordinate, needed because the API accepts raw coordinates, not pre-engineered features
- Rebuilt `api/main.py` and `api/schemas.py` around `GET /risk-score?lat=&lon=&datetime=`, matching the endpoint contract in the task document and the raw-coordinate input FE confirmed they will send
- Fixed a startup bug where `@app.on_event("startup")` was silently not loading the model under the installed FastAPI version, replaced with the `lifespan` pattern
- Added a short-circuit for locations with no recorded crime nearby, returning a score of 0 directly instead of asking the model to extrapolate below the lowest crime count it was trained on
- Re-trained `champion.joblib` to match the 5-feature CP1 schema (`lat_r, lon_r, hour_sin, hour_cos, nearby_crime_count`), the checkpoint previously in this slot was left over from an unrelated Hands-On 2 pipeline with a different feature set
- Added a cell to `02_model_training_baseline.ipynb` that saves `champion.joblib` and `champion_meta.json` directly, the notebook previously only saved `model_v0.joblib` and `registry.json`, so the API's expected artifact was never actually produced by running the notebook, only by a separate script
- Pinned `scikit-learn==1.6.1` in `requirements.txt` to match the version Colab uses, loading `champion.joblib` on a different scikit-learn version raised an `InconsistentVersionWarning` that pip's unpinned `scikit-learn` would have reproduced on a fresh local install
- Removed `COPY models/` from `Dockerfile`, `data/` and `models/` are both mounted as volumes at `docker run` time now instead of being baked into the image, consistent with keeping large or environment-specific artifacts out of anything committed or built

### CP1 revision 2
- Unified the clip bound used by the point-query function and the training table into one value, computed once from the full 726-cell grid instead of two separately calibrated numbers
- Added timezone handling, timestamps sent with a UTC offset are converted to Chicago local time before scoring
- Switched risk levels from three percentile-based tiers to four fixed tiers (Low/Medium/High/Very High at 25/50/75), matching the FE badge system
- Documented that most locations read Very High under the fixed tiers, this reflects three years of accumulated crime history within 1200 meters, not a bug

## Repository Structure

| Path | Description |
|---|---|
| `notebooks/` | Step by step notebooks for each stage of the project |
| `src/` | Shared logic reused by both notebooks and the API, so scoring behaves identically in training and serving |
| `api/` | The FastAPI application, added in CP2 |
| `models/` | Saved model checkpoints and the version registry, not committed to Git, see `models/README.md` |
| `data/` | Instructions for obtaining the raw dataset, see `data/README.md` |
| `logs/` | Prediction activity log, created automatically the first time the API runs, see `logs/README.md` |
| `FP_MLOps_GreyciaFebrinaMichelle_AuliaAcaAzzahra_Output.json` | Example predictions from a real run of the API |

## How to Run the Notebooks

1. Get `events_scored.csv` following the instructions in `data/README.md`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/01_pseudo_labeling_feature_engineering.ipynb` and run all cells in order
4. Open `notebooks/02_model_training_baseline.ipynb` and run all cells in order, this produces `models/champion.joblib` and `models/registry.json`
5. Open `notebooks/03_continual_learning.ipynb` and run all cells in order, this simulates data arriving quarterly and overwrites `models/champion.joblib` with whichever version won promotion last

## How to Run the API

1. Make sure `data/events_scored.csv` and `data/feature_table_fp.csv` are in place, see `data/README.md`
2. Make sure `models/champion.joblib` and `models/champion_meta.json` exist, see `models/README.md` if they are missing
3. Install dependencies: `pip install -r requirements.txt`
4. From the project root, start the server:

```
uvicorn api.main:app --reload
```

5. Call the endpoint:

```
curl "http://127.0.0.1:8000/risk-score?lat=41.89&lon=-87.63&datetime=2026-08-03T20:00:00"
```

Example response:

```json
{
  "risk_score": 99.51,
  "level": "Very High",
  "model_version": "v2",
  "last_updated": "2026-08-06"
}
```

A location with no recorded crime within 1200 meters returns `"risk_score": 0.0, "level": "Low"` instead of a model prediction, since the model was never trained on locations with zero history and should not guess at that extreme.

## Monitoring

Every call to `/risk-score` appends one line to `logs/predictions.jsonl`, see `logs/README.md` for the schema. `GET /metrics` reads that log together with `models/registry.json` and returns the current model version and its holdout metrics, the full promotion history across every checkpoint, and a summary of recent activity:

```
curl "http://127.0.0.1:8000/metrics"
```

## How to Run with Docker

The image only contains code, `data/` and `models/` are mounted at run time rather than baked in, so the image never needs rebuilding just because the data or the model changed.

```
docker build -t risk-score-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models risk-score-api
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}`.
