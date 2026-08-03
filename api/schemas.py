from datetime import datetime

from pydantic import BaseModel, Field


class RiskScoreResponse(BaseModel):
    risk_score: float = Field(..., description="Skor 0-100")
    level: str = Field(..., description="Low, Medium, High, atau Very High")
    model_version: str = Field(..., description="Versi model yang menjawab, contoh: v0")
    last_updated: str = Field(..., description="Tanggal model ini terakhir dilatih")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 74.0,
                "level": "High",
                "model_version": "v0",
                "last_updated": "2026-08-03",
            }
        }
