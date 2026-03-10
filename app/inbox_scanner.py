# =============================================================================
# app/inbox_scanner.py  —  Kaveesha: Auto-scan all inbox emails
# Fetches inbox emails and runs the temporal evasion model on each one.
# =============================================================================

from typing import List, Dict, Any

from .gmail_fetch import fetch_inbox
from .temporal_evasion.predictor import predict_temporal_evasion



def scan_single_email(email: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the temporal evasion model on one email dict
    (as returned by fetch_inbox / get_emails).
    Returns the original email enriched with threat analysis fields.
    """
    try:
        result = predict_temporal_evasion({
            "subject":     email.get("subject", ""),
            "body":        email.get("body", ""),
            "sent_at":     email.get("received_at", ""),
            "burst_count": 1,
        })

        # Map predictor output → threat shape the frontend expects
        is_threat  = result["is_threat"]
        confidence = result["confidence"]   # already 0-100
        risk_level = result["risk_level"].upper()  # safe/low/medium/high/critical → uppercase
        verdict    = "SPAM" if is_threat else "HAM"

        risk_color = {
            "SAFE":     "#2e7d32",
            "LOW":      "#2e7d32",
            "MEDIUM":   "#ed6c02",
            "HIGH":     "#d32f2f",
            "CRITICAL": "#7b1fa2",
        }.get(risk_level, "#9e9e9e")

        return {
            **email,
            "threat": {
                "verdict":          verdict,
                "spam_probability": round(confidence, 2),
                "risk_level":       risk_level,
                "risk_color":       risk_color,
                "model_scores":     [],   # stacking ensemble — no individual scores exposed
                "temporal_flags":   result["indicators"],
                "scan_summary":     (f"Classified as {verdict} with "
                                     f"{confidence:.1f}% confidence. Risk: {risk_level}."),
                "model_version":    result["model_used"],
                "scanned":          True,
            },
        }

    except Exception as e:
        return {
            **email,
            "threat": {
                "verdict":          "ERROR",
                "spam_probability": 0.0,
                "risk_level":       "UNKNOWN",
                "risk_color":       "#9e9e9e",
                "model_scores":     [],
                "temporal_flags":   [f"Scan error: {str(e)}"],
                "scan_summary":     f"Scan failed: {str(e)}",
                "model_version":    "GlobalTemporalEvasion-v1",
                "scanned":          False,
            },
        }


def scan_inbox(source: str = "gmail", max_results: int = 20) -> Dict[str, Any]:
    """
    Fetch inbox emails and auto-scan every one.
    source: "gmail" | "demo"
    """
    from .email_store import EmailStore

    if source == "gmail":
        emails = fetch_inbox(max_results=max_results)
        connected = emails is not None
        raw_list  = emails if emails else []
    else:
        store    = EmailStore()
        raw_list = store.list()
        connected = False

    scanned = [scan_single_email(e) for e in raw_list]

    spam_count = sum(1 for e in scanned if e["threat"]["verdict"] == "SPAM")
    high_risk  = sum(1 for e in scanned if e["threat"]["risk_level"] in ("HIGH", "CRITICAL"))

    return {
        "connected":   connected,
        "source":      "gmail" if connected else "demo",
        "total":       len(scanned),
        "spam_count":  spam_count,
        "ham_count":   len(scanned) - spam_count,
        "high_risk":   high_risk,
        "emails":      scanned,
    }
