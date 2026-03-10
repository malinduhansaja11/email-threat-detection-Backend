"""
header_spoofing/routes.py
━━━━━━━━━━━━━━━━━━━━━━━━
FastAPI router — mounts at  /header-spoofing

Endpoints
─────────
POST /header-spoofing/predict
    Body   : HeaderSpoofingRequest
    Returns: HeaderSpoofingResponse

GET  /header-spoofing/status
    Returns model load status
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from .predictor import predict_header_spoofing, _load_error

router = APIRouter(prefix="/header-spoofing", tags=["Header Spoofing"])


# ── Request / Response schemas ────────────────────────────────────────────────

class HeaderSpoofingRequest(BaseModel):
    """All fields that come from the email."""
    # Email metadata
    email_from:     str  = Field("", description="From: header value")
    subject:        str  = Field("", description="Subject line")
    body:           str  = Field("", description="Plain-text body")
    reply_to:       str  = Field("", description="Reply-To header (empty = same as From)")
    # Authentication signals
    dkim:           bool = Field(True,  description="DKIM verification passed")
    spf:            bool = Field(True,  description="SPF verification passed")
    has_attachment: bool = Field(False, description="Email has file attachments")

    class Config:
        json_schema_extra = {
            "example": {
                "email_from":     "billing@susp1c10us-bank.tk",
                "subject":        "URGENT: verify your account NOW",
                "body":           "Click here to reset your p@ssw0rd immediately.",
                "reply_to":       "harvest@other-domain.xyz",
                "dkim":           False,
                "spf":            False,
                "has_attachment": True,
            }
        }


class HeaderSpoofingResponse(BaseModel):
    is_threat:   bool
    confidence:  float          # 0–100 %
    risk_level:  str            # safe | low | medium | high | critical
    threat_type: str
    model_used:  str
    details:     List[str]
    scores:      Dict[str, float]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=HeaderSpoofingResponse)
def predict(req: HeaderSpoofingRequest):
    """
    Run the Header-Spoofing ensemble model on one email.

    Feature vector (8 dims, each 0–1):
      sender_score, link_score, language_score, header_score,
      server_score, login_score, path_score, attachment_score

    Model: XGBoost + MLP + RandomForest → LogReg meta-judge
    """
    payload = {
        "from":           req.email_from,
        "subject":        req.subject,
        "body":           req.body,
        "reply_to":       req.reply_to,
        "dkim":           req.dkim,
        "spf":            req.spf,
        "has_attachment": req.has_attachment,
    }
    result = predict_header_spoofing(payload)
    return HeaderSpoofingResponse(**result)


@router.get("/status")
def model_status():
    """Returns whether MASTER_MODEL_LARGE.pkl loaded successfully."""
    import os
    from .predictor import MODEL_PATH, _pkg
    loaded = _pkg is not None
    return {
        "module":       "header_spoofing",
        "model_loaded": loaded,
        "model_path":   MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "error":        _load_error if not loaded else None,
    }
