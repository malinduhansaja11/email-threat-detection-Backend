"""
header_spoofing/predictor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Header-Spoofing Detection Module
Based on: ayani_pipeline.ipynb

Pipeline recap (from notebook):
  • Input features  : 8 numeric scores
      [sender_score, link_score, language_score, header_score,
       server_score, login_score, path_score, attachment_score]
  • Feature engineering : adds 150 sin-wave bloat layers → total = 8 + 150 + extras
  • Models trained  : XGBoost (m1) + MLP (m2) + RandomForest (m3)
  • Meta-judge      : LogisticRegression stacker
  • Saved as        : MASTER_MODEL_LARGE.pkl  (dict with keys m1,m2,m3,judge,scaler,real_col_count)

If the .pkl is absent the module falls back to a rule-based engine
so the server still starts and returns meaningful results.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List

import numpy as np

# ── model cache ───────────────────────────────────────────────────────────────
_pkg: dict | None = None
_load_error: str  = ""

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "MASTER_MODEL_LARGE.pkl"
)


def _load_pkg() -> dict | None:
    global _pkg, _load_error
    if _pkg is not None:
        return _pkg
    if not os.path.exists(MODEL_PATH):
        _load_error = f"Model file not found: {MODEL_PATH}"
        return None
    try:
        import joblib
        _pkg = joblib.load(MODEL_PATH)
        return _pkg
    except Exception as exc:
        _load_error = str(exc)
        return None


# ── feature extraction (mirrors notebook Section 3 & 10) ─────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "verify", "suspended", "click now",
    "confirm", "account", "limited", "expire", "warning", "reset",
    "update", "validate", "authenticate",
]
LOGIN_WORDS = ["login", "password", "username", "credential", "sign in", "signin"]
SUSPICIOUS_TLDS = r"\.(tk|ml|ga|cf|gq|xyz|top|club|work|click|pw|space|info|biz)$"


def _extract_scores(payload: Dict[str, Any]) -> np.ndarray:
    """
    Build the 8-dim feature vector used by the notebook's live-email test:
        [sender_score, link_score, language_score, header_score,
         server_score, login_score, path_score, attachment_score]
    Each score ∈ [0.0, 1.0] — higher = more trustworthy.
    """
    sender   = str(payload.get("from",    ""))
    subject  = str(payload.get("subject", ""))
    body     = str(payload.get("body",    ""))
    reply_to = str(payload.get("reply_to",""))
    dkim     = bool(payload.get("dkim",   True))
    spf      = bool(payload.get("spf",    True))
    has_att  = bool(payload.get("has_attachment", False))
    full_text = (subject + " " + body).lower()

    # sender_score
    sender_score = 0.9
    if reply_to and reply_to.strip() and reply_to.strip() != sender.strip():
        sender_score -= 0.5
    if re.search(r"\d{5,}", sender):
        sender_score -= 0.25
    if re.search(SUSPICIOUS_TLDS, sender.lower()):
        sender_score -= 0.3
    sender_score = max(0.0, sender_score)

    # link_score
    urls = re.findall(r"https?://\S+", body)
    if not urls:
        link_score = 0.85
    else:
        bad = sum(1 for u in urls if re.search(SUSPICIOUS_TLDS, u.lower()))
        link_score = max(0.0, 0.9 - len(urls) * 0.1 - bad * 0.25)

    # language_score
    hits = sum(1 for w in URGENCY_WORDS if w in full_text)
    language_score = max(0.0, 1.0 - hits * 0.12)

    # header_score  (DKIM + SPF)
    header_score = (0.5 if dkim else 0.0) + (0.5 if spf else 0.0)

    # server_score  (heuristic – unknown)
    server_score = 0.70

    # login_score
    login_hits = sum(1 for w in LOGIN_WORDS if w in full_text)
    login_score = max(0.0, 1.0 - login_hits * 0.30)

    # path_score
    path_score = max(0.0, 0.9 - len(urls) * 0.12)

    # attachment_score
    attachment_score = 0.3 if has_att else 0.90

    return np.array(
        [sender_score, link_score, language_score, header_score,
         server_score, login_score, path_score, attachment_score],
        dtype=np.float32,
    )


def _pad_and_scale(raw8: np.ndarray, pkg: dict) -> np.ndarray:
    """Pad 8-feature vector to match the full feature width used during training."""
    scaler       = pkg["scaler"]
    real_cols    = pkg.get("real_col_count", 8)
    total_width  = real_cols + 150         # 150 bloat layers added in notebook
    X_full       = np.zeros((1, total_width), dtype=np.float32)
    X_full[0, :8] = raw8
    return scaler.transform(X_full)


def _ml_predict(raw8: np.ndarray, pkg: dict) -> tuple[bool, float]:
    X_scaled = _pad_and_scale(raw8, pkg)
    m1, m2, m3, judge = pkg["m1"], pkg["m2"], pkg["m3"], pkg["judge"]
    p1 = m1.predict_proba(X_scaled)[:, 1]
    p2 = m2.predict_proba(X_scaled)[:, 1]
    p3 = m3.predict_proba(X_scaled)[:, 1]
    ji = np.column_stack([p1, p2, p3])
    pred = int(judge.predict(ji)[0])
    prob = float(judge.predict_proba(ji)[0, 1])
    is_spam = (pred == 1)
    confidence = prob if is_spam else (1.0 - prob)
    return is_spam, confidence


def _rule_predict(raw8: np.ndarray) -> tuple[bool, float]:
    avg = float(raw8.mean())
    # Low average score → suspicious
    is_spam    = avg < 0.45
    confidence = (1.0 - avg) if is_spam else avg
    return is_spam, min(confidence, 0.99)


def _risk_label(c: float) -> str:
    if c >= 0.85: return "critical"
    if c >= 0.65: return "high"
    if c >= 0.40: return "medium"
    if c >= 0.20: return "low"
    return "safe"


def _build_details(scores: np.ndarray, payload: Dict[str, Any]) -> List[str]:
    names = ["Sender Trust", "Link Safety", "Language Safety",
             "Header Auth (DKIM/SPF)", "Server Trust",
             "Login-Keyword Risk", "Path Safety", "Attachment Risk"]
    flags = []
    for i, (name, val) in enumerate(zip(names, scores)):
        if val < 0.45:
            flags.append(f"Low {name} score ({val:.2f}) — potential spoofing indicator")
    if not flags:
        flags.append("All header signals look legitimate. No spoofing detected.")
    return flags


# ── public API ────────────────────────────────────────────────────────────────

def predict_header_spoofing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry-point called by routes.py.

    Parameters
    ----------
    payload : dict with keys: from, subject, body, reply_to,
                               dkim, spf, has_attachment

    Returns
    -------
    dict with keys: is_threat, confidence, risk_level,
                    threat_type, model_used, details, scores
    """
    raw8   = _extract_scores(payload)
    pkg    = _load_pkg()
    scores_dict = {
        "sender":      round(float(raw8[0]), 3),
        "link":        round(float(raw8[1]), 3),
        "language":    round(float(raw8[2]), 3),
        "header_auth": round(float(raw8[3]), 3),
        "server":      round(float(raw8[4]), 3),
        "login":       round(float(raw8[5]), 3),
        "path":        round(float(raw8[6]), 3),
        "attachment":  round(float(raw8[7]), 3),
    }

    if pkg:
        try:
            is_threat, confidence = _ml_predict(raw8, pkg)
            model_used = "XGBoost + MLP + RandomForest + LogReg Meta-Judge (Stacking Ensemble)"
        except Exception as exc:
            is_threat, confidence = _rule_predict(raw8)
            model_used = f"Rule-Based Fallback (ML error: {exc})"
    else:
        is_threat, confidence = _rule_predict(raw8)
        model_used = f"Rule-Based Engine ({_load_error or 'model not loaded'})"

    return {
        "is_threat":   is_threat,
        "confidence":  round(confidence * 100, 1),
        "risk_level":  _risk_label(confidence if is_threat else 1.0 - confidence),
        "threat_type": "Header Spoofing / Spam Detected" if is_threat else "Legitimate Email",
        "model_used":  model_used,
        "details":     _build_details(raw8, payload),
        "scores":      scores_dict,
    }
