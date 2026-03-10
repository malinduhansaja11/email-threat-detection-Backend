"""
temporal_evasion/routes.py
══════════════════════════════════════════════════════════════════════════════
FastAPI router — mounts at  /temporal-evasion

Endpoints
─────────
POST /temporal-evasion/predict
    Body   : TemporalEvasionRequest
    Returns: TemporalEvasionResponse

GET  /temporal-evasion/status
    Returns model load status + bundle metadata
"""

import os
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .predictor import predict_temporal_evasion, _load_error, MODEL_PATH

router = APIRouter(prefix="/temporal-evasion", tags=["Temporal Evasion"])


# ── Request / Response schemas ────────────────────────────────────────────────

class TemporalEvasionRequest(BaseModel):
    subject:     str = Field("", description="Email subject line")
    body:        str = Field("", description="Plain-text email body")
    sent_at:     str = Field(
        "",
        description="ISO-8601 timestamp of when email was sent, e.g. 2025-03-10T03:14:00",
    )
    burst_count: int = Field(
        1,
        ge=1,
        description="How many emails received from this sender in the last hour",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "subject":     "URGENT ACTION REQUIRED — account suspended in 2 hours",
                "body":        "Click here immediately to reset your password or lose access.",
                "sent_at":     "2025-03-10T03:14:00",
                "burst_count": 22,
            }
        }


class TemporalEvasionResponse(BaseModel):
    is_threat:         bool
    confidence:        float           # 0 – 100 %
    risk_level:        str             # safe | low | medium | high | critical
    threat_type:       str
    model_used:        str
    temporal_features: Dict[str, float]
    indicators:        List[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=TemporalEvasionResponse)
def predict(req: TemporalEvasionRequest):
    """
    Run the Temporal-Evasion stacking ensemble on one email.

    Temporal feature vector (11 dims):
      hour, hour_sin, hour_cos, is_suspicious_time, is_burst,
      time_drift, day_of_week, is_weekend, arrival_epoch,
      burst_count, is_anomaly

    Combined with TF-IDF text features → fed to
    RF(2200) + XGBoost(800) + Calibrated-SVM → LogReg meta-judge
    """
    payload = {
        "subject":     req.subject,
        "body":        req.body,
        "sent_at":     req.sent_at,
        "burst_count": req.burst_count,
    }
    result = predict_temporal_evasion(payload)
    return TemporalEvasionResponse(**result)


@router.get("/status")
def model_status():
    """Returns whether GLOBAL_TEMPORAL_EVASION_ULTIMATE_FINAL.pkl loaded."""
    from .predictor import _bundle
    loaded = _bundle is not None
    meta   = {}
    if loaded and "research_metadata" in _bundle:
        meta = _bundle["research_metadata"]
    return {
        "module":        "temporal_evasion",
        "model_loaded":  loaded,
        "model_path":    MODEL_PATH,
        "model_exists":  os.path.exists(MODEL_PATH),
        "error":         _load_error if not loaded else None,
        "metadata":      meta,
    }
