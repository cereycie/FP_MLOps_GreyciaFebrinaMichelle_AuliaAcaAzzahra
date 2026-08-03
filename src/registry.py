import json, os
from pathlib import Path
from datetime import datetime, timezone
import joblib

REGISTRY_PATH = "models/registry.json"
CHAMPION_PATH = "models/champion.joblib"
CHAMPION_META_PATH = "models/champion_meta.json"

def ensure_models_dir():
    os.makedirs("models", exist_ok=True)

def save_checkpoint(fitted_model, version):
    path = f"models/model_v{version}.joblib"
    joblib.dump(fitted_model, path)
    return path

def log_registry(entry, registry):
    registry.append(entry)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def save_champion(fitted_model, version, metrics, feature_cols, target_col):
    joblib.dump(fitted_model, CHAMPION_PATH)
    meta = {
        "version": version, "feature_cols": feature_cols, "target_col": target_col,
        "metrics_holdout": metrics, "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHAMPION_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    return CHAMPION_PATH

def load_champion(models_dir="models"):
    """Load model champion aktif. models_dir bisa path relatif (dipakai notebook)
    atau path absolut (dipakai API, supaya tidak tergantung folder mana uvicorn dijalankan)."""
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "champion.joblib")
    with open(models_dir / "champion_meta.json") as f:
        meta = json.load(f)
    return model, meta