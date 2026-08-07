from datetime import datetime, timezone, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

TEST_MODE = True
sharelock_sessions = {}


class StartJourneyRequest(BaseModel):
    latitude: float
    longitude: float
    pin: str
    duration_minutes: int


class StartJourneyResponse(BaseModel):
    session_id: str
    status: str
    started_at: datetime
    expires_at: datetime
    tracking_link: str


class VerifyPinRequest(BaseModel):
    session_id: str
    pin: str


class VerifyPinResponse(BaseModel):
    success: bool
    message: str


class EndJourneyRequest(BaseModel):
    session_id: str
    pin: str


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


def check_session_expiry(session):
    if datetime.now(timezone.utc) > session["expires_at"]:
        session["status"] = "EXPIRED"
    return session


@router.post("/start", response_model=StartJourneyResponse)
def start_sharelock(request: StartJourneyRequest):
    session_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    expires_at = started_at + timedelta(minutes=request.duration_minutes)
    tracking_link = f"http://localhost:8000/sharelock/public/{session_id}"

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
        session_id=session_id, status="ACTIVE", started_at=started_at,
        expires_at=expires_at, tracking_link=tracking_link,
    )


@router.get("/public/{session_id}")
def public_tracking(session_id: str):
    session = sharelock_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session Sharelock tidak ditemukan.")

    session = check_session_expiry(session)
    return {
        "session_id": session["session_id"], "status": session["status"],
        "latitude": session["latitude"], "longitude": session["longitude"],
        "started_at": session["started_at"], "expires_at": session["expires_at"],
    }


@router.post("/verify-pin", response_model=VerifyPinResponse)
def verify_sharelock_pin(request: VerifyPinRequest):
    session = sharelock_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session Sharelock tidak ditemukan.")

    session = check_session_expiry(session)
    if session["status"] != "ACTIVE":
        return VerifyPinResponse(success=False, message="Sesi Sharelock sudah tidak aktif.")

    if session["pin_attempt"] >= 3:
        return VerifyPinResponse(success=False, message="Percobaan PIN sudah mencapai batas. Silakan reset PIN.")

    if request.pin == session["pin"]:
        session["verified"] = True
        return VerifyPinResponse(success=True, message="PIN berhasil diverifikasi.")

    session["pin_attempt"] += 1
    if session["pin_attempt"] >= 3:
        return VerifyPinResponse(success=False, message="PIN salah 3 kali. Silakan lakukan reset PIN.")

    remaining_attempts = 3 - session["pin_attempt"]
    return VerifyPinResponse(success=False, message=f"PIN salah. Sisa percobaan: {remaining_attempts}.")


@router.post("/end", response_model=EndJourneyResponse)
def end_sharelock(request: EndJourneyRequest):
    session = sharelock_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session Sharelock tidak ditemukan.")

    session = check_session_expiry(session)
    if session["status"] != "ACTIVE":
        return EndJourneyResponse(success=False, status=session["status"], message="Sesi Sharelock sudah tidak aktif.")

    if request.pin != session["pin"]:
        return EndJourneyResponse(success=False, status="ACTIVE", message="PIN salah. Sesi masih berjalan.")

    session["status"] = "SAFE_ARRIVAL"
    session["verified"] = True
    return EndJourneyResponse(
        success=True, status="SAFE_ARRIVAL",
        message="Perjalanan berhasil dikonfirmasi. Live location dihentikan.",
    )


@router.post("/extend", response_model=ExtendJourneyResponse)
def extend_sharelock(request: ExtendJourneyRequest):
    session = sharelock_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session Sharelock tidak ditemukan.")

    session = check_session_expiry(session)
    if session["status"] != "ACTIVE":
        return ExtendJourneyResponse(
            success=False, status=session["status"], expires_at=session["expires_at"],
            message="Sesi Sharelock sudah tidak aktif.",
        )

    session["expires_at"] = session["expires_at"] + timedelta(minutes=10)
    return ExtendJourneyResponse(
        success=True, status="ACTIVE", expires_at=session["expires_at"],
        message="Durasi Sharelock berhasil diperpanjang 10 menit.",
    )


@router.post("/forgot-pin", response_model=ForgotPinResponse)
def forgot_sharelock_pin(request: ForgotPinRequest):
    session = sharelock_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session Sharelock tidak ditemukan.")

    session = check_session_expiry(session)
    if session["status"] != "ACTIVE":
        return ForgotPinResponse(
            success=False, status=session["status"],
            message="Sesi Sharelock sudah tidak aktif.", waiting_timer_paused=False,
        )

    session["waiting_timer_paused"] = True
    session["pin_reset_requested"] = True
    return ForgotPinResponse(
        success=True, status="PIN_RESET_REQUIRED",
        message="Silakan lakukan reset PIN melalui email terdaftar.", waiting_timer_paused=True,
    )
