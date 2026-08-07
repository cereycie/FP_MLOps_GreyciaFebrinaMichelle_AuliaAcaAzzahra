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

# Smart Sharelock API Schemas
class StartJourneyRequest(BaseModel):
    latitude: float
    longitude: float
    duration_minutes: int = Field(..., gt=0)
    pin: str = Field(..., min_length=4, max_length=4)


class StartJourneyResponse(BaseModel):
    session_id: str
    status: str
    started_at: datetime
    expires_at: datetime
    tracking_link: str


class VerifyPinRequest(BaseModel):
    session_id: str
    pin: str = Field(..., min_length=4, max_length=4)


class VerifyPinResponse(BaseModel):
    success: bool
    message: str

class EndJourneyRequest(BaseModel):
    session_id: str
    pin: str = Field(..., min_length=4, max_length=4)


class EndJourneyResponse(BaseModel):
    success: bool
    status: str
    message: str

class ExtendJourneyRequest(BaseModel):
    session_id: str


class ExtendJourneyResponse(BaseModel):
    success: bool
    status: str
    expires_at: datetime
    message: str

class ForgotPinRequest(BaseModel):
    session_id: str


class ForgotPinResponse(BaseModel):
    success: bool
    status: str
    message: str
    waiting_timer_paused: bool