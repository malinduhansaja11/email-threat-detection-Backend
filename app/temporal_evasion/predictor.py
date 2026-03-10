"""
temporal_evasion/predictor.py
══════════════════════════════════════════════════════════════════════════════
Temporal-Evasion Detection Module — Kaveesha
Based on: Kaveesha_Temporal_Evasion_Pipeline.ipynb

Pipeline summary (from notebook):
  • Datasets   : 8 CSVs from MyDrive/KaveeshaDatasets/
  • Text       : TF-IDF (max_features=8000, ngram=(1,2)) on subject + message
  • Temporal   : 11 time-based features engineered per email
      hour, hour_sin, hour_cos, is_suspicious_time, is_burst,
      time_drift, day_of_week, is_weekend, arrival_epoch,
      burst_count, is_anomaly
  • Models     : RF(2200 trees) + XGBoost(800 trees) + Calibrated-SGD/SVM
  • Meta-judge : LogisticRegression stacker
  • PKL bundle : GLOBAL_TEMPORAL_EVASION_ULTIMATE_FINAL.pkl
      Keys: vectorizer, label_encoder, rf_model, xgb_model,
            svm_model, meta_model, scanned_metrics

If the .pkl is absent the module falls back to a rule-based engine so
the server still starts and returns meaningful results.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ── model cache ──────────────────────────────────────────────────────────────
_bundle: dict | None = None
_load_error: str = ""

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models",
    "GLOBAL_TEMPORAL_EVASION_ULTIMATE_FINAL.pkl"
)

TEMPORAL_COLS = [
    "hour", "hour_sin", "hour_cos", "is_suspicious_time",
    "is_burst", "time_drift", "day_of_week", "is_weekend",
    "arrival_epoch", "burst_count", "is_anomaly",
]


def _load_bundle() -> dict | None:
    global _bundle, _load_error
    if _bundle is not None:
        return _bundle
    if not os.path.exists(MODEL_PATH):
        _load_error = f"Model file not found: {MODEL_PATH}"
        return None
    try:
        import joblib
        _bundle = joblib.load(MODEL_PATH)
        return _bundle
    except Exception as exc:
        _load_error = str(exc)
        return None


# ── temporal feature engineering (mirrors notebook Section 5 & 13) ────────────

SUSPICIOUS_HOURS = set(range(0, 5)) | set(range(22, 24))   # 10 PM – 4 AM
URGENCY_RE = re.compile(
    r"\b(urgent|immediately|suspend|verify|confirm|reset|expire|"
    r"click now|act now|account|credential|password|limited|warning)\b",
    re.I,
)


def _extract_temporal_features(payload: Dict[str, Any]) -> np.ndarray:
    """
    Build the 11-dimensional temporal feature vector used by the notebook.
    All values derived from the email metadata supplied in the API request.
    """
    # ── raw inputs ────────────────────────────────────────────────────────────
    sent_at: str  = payload.get("sent_at", "")        # ISO-8601 or empty
    body:    str  = payload.get("body",    "")
    subject: str  = payload.get("subject", "")
    burst_count_raw: int = int(payload.get("burst_count", 1))

    # ── parse timestamp ───────────────────────────────────────────────────────
    dt = _parse_dt(sent_at)
    hour       = dt.hour
    dow        = dt.weekday()          # 0=Mon … 6=Sun
    is_weekend = int(dow >= 5)
    epoch      = int(dt.timestamp())

    # ── cyclic hour encoding ──────────────────────────────────────────────────
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # ── derived flags ─────────────────────────────────────────────────────────
    is_suspicious_time = int(hour in SUSPICIOUS_HOURS)

    burst_count = burst_count_raw
    is_burst    = int(burst_count > 5)

    # time_drift: seconds since a "normal" 9 AM baseline
    normal_epoch = dt.replace(hour=9, minute=0, second=0).timestamp()
    time_drift   = abs(epoch - normal_epoch)

    # anomaly: suspicious hour AND burst AND urgency keywords
    urgency_hits = len(URGENCY_RE.findall(subject + " " + body))
    is_anomaly   = int(is_suspicious_time and is_burst and urgency_hits > 0)

    return np.array([
        hour, hour_sin, hour_cos, is_suspicious_time,
        is_burst, time_drift, dow, is_weekend,
        epoch, burst_count, is_anomaly,
    ], dtype=np.float64)


def _parse_dt(sent_at: str) -> datetime:
    """Parse ISO string; fall back to current UTC time."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(sent_at, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


# ── ML inference ──────────────────────────────────────────────────────────────

def _ml_predict(
    text: str, temporal: np.ndarray, bundle: dict
) -> Tuple[bool, float, str]:
    from scipy.sparse import hstack, csr_matrix

    vectorizer    = bundle["vectorizer"]
    rf_model      = bundle["rf_model"]
    xgb_model     = bundle["xgb_model"]
    svm_model     = bundle["svm_model"]
    meta_model    = bundle["meta_model"]
    scanned_cols: list = bundle.get("scanned_metrics", TEMPORAL_COLS)
    le            = bundle.get("label_encoder", None)

    # text features
    X_text = vectorizer.transform([text])

    # temporal features — only columns the model was trained on
    n_temp = len(scanned_cols)
    temp_vals = temporal[:n_temp].reshape(1, -1)
    X_temp    = csr_matrix(temp_vals)

    X_in = hstack([X_text, X_temp]) if n_temp > 0 else X_text

    p1 = rf_model.predict_proba(X_in)[:, 1]
    p2 = xgb_model.predict_proba(X_in)[:, 1]
    p3 = svm_model.predict_proba(X_in)[:, 1]

    meta_in = np.column_stack([p1, p2, p3])
    pred    = int(meta_model.predict(meta_in)[0])
    prob    = float(meta_model.predict_proba(meta_in)[0, 1])

    # decode label
    if le is not None:
        label = str(le.inverse_transform([pred])[0])
        is_threat = label.lower() in ("spam", "1", "malicious", "threat")
    else:
        is_threat = pred == 1

    confidence = prob if is_threat else 1.0 - prob
    model_used = (
        "RandomForest(2200) + XGBoost(800) + Calibrated-SGD/SVM "
        "+ LogReg Meta-Judge (Stacking Ensemble)"
    )
    return is_threat, confidence, model_used


# ── rule-based fallback ───────────────────────────────────────────────────────

def _rule_predict(
    temporal: np.ndarray, payload: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """Simple heuristic when the PKL file is absent."""
    hour            = int(temporal[0])
    is_suspicious   = int(temporal[3])
    is_burst        = int(temporal[4])
    time_drift      = float(temporal[5])
    is_anomaly      = int(temporal[10])

    score = 0.0
    if hour in SUSPICIOUS_HOURS:         score += 0.35
    if is_burst:                          score += 0.25
    if time_drift > 7200:                 score += 0.20
    if is_anomaly:                        score += 0.20
    urgency = len(URGENCY_RE.findall(
        payload.get("subject", "") + " " + payload.get("body", "")
    ))
    score += min(urgency * 0.08, 0.25)
    score  = min(score, 0.99)

    is_threat  = score >= 0.45
    confidence = score if is_threat else 1.0 - score
    return is_threat, confidence, f"Rule-Based Engine ({_load_error or 'model not loaded'})"


# ── risk helpers ─────────────────────────────────────────────────────────────

def _risk_label(confidence: float, is_threat: bool) -> str:
    c = confidence if is_threat else 1.0 - confidence
    if not is_threat:
        return "safe"
    if c >= 0.90: return "critical"
    if c >= 0.70: return "high"
    if c >= 0.50: return "medium"
    return "low"


def _build_indicators(temporal: np.ndarray, payload: Dict[str, Any]) -> List[str]:
    hour       = int(temporal[0])
    is_susp    = int(temporal[3])
    is_burst   = int(temporal[4])
    time_drift = float(temporal[5])
    dow        = int(temporal[6])
    is_weekend = int(temporal[7])
    burst_cnt  = int(temporal[9])
    is_anomaly = int(temporal[10])

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    info: List[str] = []

    info.append(f"Send time: {hour:02d}:00 on {days[dow]} "
                f"({'weekend' if is_weekend else 'weekday'})")
    if is_susp:
        info.append(f"⚠ Off-hours send: {hour:02d}:00 falls in suspicious window (10 PM – 4 AM)")
    if is_burst:
        info.append(f"⚠ Burst activity detected: {burst_cnt} emails in short window")
    if time_drift > 3600:
        hrs = time_drift / 3600
        info.append(f"⚠ Time drift {hrs:.1f} h from normal business-hours baseline")
    urgency = URGENCY_RE.findall(payload.get("subject","") + " " + payload.get("body",""))
    if urgency:
        info.append(f"⚠ Urgency keywords detected: {', '.join(set(urgency[:5]))}")
    if is_anomaly:
        info.append("🚨 Anomaly flag: off-hours + burst + urgency keywords all present")
    if not any(s.startswith("⚠") or s.startswith("🚨") for s in info):
        info.append("✅ No suspicious temporal patterns detected")
    return info


# ── public API ────────────────────────────────────────────────────────────────

def predict_temporal_evasion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry-point called by routes.py.

    Parameters (payload keys)
    ─────────────────────────
    subject     : str   — email subject
    body        : str   — plain-text email body
    sent_at     : str   — ISO-8601 timestamp  (e.g. "2025-03-10T03:14:00")
    burst_count : int   — emails received from same sender in last hour (default 1)

    Returns
    ───────
    dict: is_threat, confidence, risk_level, threat_type,
          model_used, temporal_features, indicators
    """
    temporal = _extract_temporal_features(payload)
    bundle   = _load_bundle()
    text     = payload.get("subject", "") + " " + payload.get("body", "")

    if bundle:
        try:
            is_threat, confidence, model_used = _ml_predict(text, temporal, bundle)
        except Exception as exc:
            is_threat, confidence, model_used = _rule_predict(temporal, payload)
            model_used = f"Rule-Based Fallback (ML error: {exc})"
    else:
        is_threat, confidence, model_used = _rule_predict(temporal, payload)

    feat_dict = dict(zip(TEMPORAL_COLS, [round(float(v), 4) for v in temporal]))

    return {
        "is_threat":         is_threat,
        "confidence":        round(confidence * 100, 1),
        "risk_level":        _risk_label(confidence, is_threat),
        "threat_type":       "Temporal Evasion / Suspicious Timing" if is_threat else "Normal Email Pattern",
        "model_used":        model_used,
        "temporal_features": feat_dict,
        "indicators":        _build_indicators(temporal, payload),
    }
