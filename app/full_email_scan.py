from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field


router = APIRouter(prefix="/scan", tags=["Full Email Scan"])


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class SingleEmailScanRequest(BaseModel):
    # Common email identifiers
    id: Optional[str] = None
    message_id: Optional[str] = None

    # Sender fields
    sender: str = ""
    email_from: str = ""
    from_: str = Field("", alias="from")

    # Email content
    subject: str = ""
    body: str = ""
    body_preview: str = ""

    # Date/time fields
    date: str = ""
    sent_at: str = ""
    email_date: str = ""

    # Header/authentication fields
    reply_to: str = ""
    return_path: str = ""
    spf: Any = "unknown"
    dkim: Any = "unknown"
    dmarc: Any = "unknown"

    # Extra metadata
    headers: Dict[str, Any] = Field(default_factory=dict)
    received: List[Any] = Field(default_factory=list)
    attachments: List[Any] = Field(default_factory=list)
    has_attachment: bool = False

    # Temporal feature
    burst_count: int = 1

    class Config:
        allow_population_by_field_name = True


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

URL_REGEX = re.compile(
    r"https?://[^\s<>'\"{}|\\^`\[\]]+|www\.[^\s<>'\"{}|\\^`\[\]]+",
    re.IGNORECASE,
)


def safe_run(module_name: str, func: Callable[[], Any]) -> Dict[str, Any]:
    """
    Runs one scanner safely.
    If a model file is missing or one module crashes, the full scan still returns.
    """
    try:
        result = func()
        return {
            "ok": True,
            "result": jsonable_encoder(result),
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "result": None,
            "error": f"{module_name} failed: {str(e)}",
        }


def get_sender(email: SingleEmailScanRequest) -> str:
    return email.sender or email.email_from or email.from_ or ""


def get_body(email: SingleEmailScanRequest) -> str:
    return email.body or email.body_preview or ""


def get_sent_at(email: SingleEmailScanRequest) -> str:
    return email.sent_at or email.date or email.email_date or ""


def auth_passed(value: Any) -> bool:
    """
    Converts SPF/DKIM values into boolean.
    Accepts values like: pass, passed, true, yes, authenticated.
    """
    if isinstance(value, bool):
        return value

    text = str(value).lower().strip()

    return text in [
        "pass",
        "passed",
        "true",
        "yes",
        "ok",
        "authenticated",
        "valid",
    ]


def extract_urls_from_text(text: str) -> List[str]:
    urls = URL_REGEX.findall(text or "")
    cleaned = []

    for url in urls:
        url = url.strip().rstrip(".,);]")
        if url.startswith("www."):
            url = "http://" + url
        cleaned.append(url)

    return list(dict.fromkeys(cleaned))


def action_rank(action: str) -> int:
    order = {
        "ALLOW": 0,
        "CLEAN": 0,
        "SAFE": 0,
        "WARN": 1,
        "WARNING": 1,
        "QUARANTINE": 2,
        "BLOCK": 3,
        "MALICIOUS": 3,
    }
    return order.get(str(action).upper(), 0)


def rank_to_action(rank: int) -> str:
    if rank >= 3:
        return "BLOCK"
    if rank == 2:
        return "QUARANTINE"
    if rank == 1:
        return "WARN"
    return "ALLOW"


# ---------------------------------------------------------
# Individual scanner runners
# ---------------------------------------------------------

def run_obfuscation_scan(body: str) -> Dict[str, Any]:
    """
    Uses existing app/analyzer.py analyze_body().
    This may use the ML model files inside app/models.
    """
    from .analyzer import analyze_body

    return analyze_body(body or "")


def run_temporal_scan(email: SingleEmailScanRequest) -> Dict[str, Any]:
    """
    Uses existing temporal_evasion predictor.
    If the temporal model file is missing, that module already has rule fallback.
    """
    from .temporal_evasion.predictor import predict_temporal_evasion

    payload = {
        "subject": email.subject or "",
        "body": get_body(email),
        "sent_at": get_sent_at(email),
        "burst_count": email.burst_count or 1,
    }

    return predict_temporal_evasion(payload)


def run_header_spoofing_scan(email: SingleEmailScanRequest) -> Dict[str, Any]:
    """
    Uses existing header_spoofing predictor.
    If MASTER_MODEL_LARGE.pkl is missing, that module already has rule fallback.
    """
    from .header_spoofing.predictor import predict_header_spoofing

    has_attachment = bool(email.has_attachment or len(email.attachments) > 0)

    payload = {
        "from": get_sender(email),
        "subject": email.subject or "",
        "body": get_body(email),
        "reply_to": email.reply_to or "",
        "dkim": auth_passed(email.dkim),
        "spf": auth_passed(email.spf),
        "has_attachment": has_attachment,
    }

    return predict_header_spoofing(payload)


def run_url_scan(email: SingleEmailScanRequest) -> Dict[str, Any]:
    """
    URL scanner using:
    1. Existing URL redirection detector.
    2. Existing URL ML classifier if model files are available.
    3. Safe rule fallback if model import fails.
    """

    sender = get_sender(email)
    body = get_body(email)

    scan_text = "\n".join([
        f"From: {sender}",
        f"Reply-To: {email.reply_to}",
        f"Return-Path: {email.return_path}",
        f"Subject: {email.subject}",
        "",
        body,
    ])

    urls = extract_urls_from_text(scan_text)

    if not urls:
        return {
            "verdict": "CLEAN",
            "message": "No URLs found",
            "total_urls": 0,
            "max_score": 0,
            "results": [],
        }

    # Try to load existing URL modules safely
    detector = None
    try:
        from url_threat_detection.redirection_detector import URLRedirectionDetector
        detector = URLRedirectionDetector()
    except Exception:
        detector = None

    predict_url = None
    try:
        from url_threat_detection.url_classifier import predict_url as existing_predict_url
        predict_url = existing_predict_url
    except Exception:
        predict_url = None

    results = []
    max_rank = 0
    max_score = 0.0

    for url in urls:
        url_result: Dict[str, Any] = {
            "url": url,
            "model_prediction": None,
            "redirection_prediction": None,
            "action": "ALLOW",
            "suspicion_score": 0.0,
            "reasons": [],
        }

        # 1. ML URL classifier, if available
        if predict_url is not None:
            try:
                model_result = predict_url(url)
                url_result["model_prediction"] = model_result

                label = str(model_result.get("label", "")).lower()
                confidence = float(model_result.get("confidence", 0))

                if label in ["phishing", "malware", "defacement", "malicious"]:
                    if confidence >= 0.85:
                        url_result["action"] = "BLOCK"
                        url_result["suspicion_score"] = max(
                            url_result["suspicion_score"],
                            0.85,
                        )
                    else:
                        url_result["action"] = "QUARANTINE"
                        url_result["suspicion_score"] = max(
                            url_result["suspicion_score"],
                            0.60,
                        )

                    url_result["reasons"].append(
                        f"URL ML model classified this URL as {label} with confidence {confidence}"
                    )

                elif label == "benign":
                    url_result["reasons"].append("URL ML model classified this URL as benign")

            except Exception as e:
                url_result["model_prediction"] = {
                    "error": str(e)
                }

        # 2. Redirection/rule detector, if available
        if detector is not None:
            try:
                redirection_result = detector.analyze_url(url)
                url_result["redirection_prediction"] = redirection_result

                redirection_action = str(
                    redirection_result.get("action", "ALLOW")
                ).upper()

                redirection_score = float(
                    redirection_result.get("suspicion_score", 0)
                )

                if action_rank(redirection_action) > action_rank(url_result["action"]):
                    url_result["action"] = redirection_action

                url_result["suspicion_score"] = max(
                    url_result["suspicion_score"],
                    redirection_score,
                )

                for reason in redirection_result.get("reasons", []):
                    url_result["reasons"].append(reason)

            except Exception as e:
                url_result["redirection_prediction"] = {
                    "error": str(e)
                }

        # 3. Simple fallback rule if both modules fail
        if predict_url is None and detector is None:
            lower_url = url.lower()

            suspicious_keywords = [
                "login",
                "verify",
                "password",
                "reset",
                "bank",
                "paypal",
                "credential",
                "wallet",
                "secure",
                "account",
                "update",
            ]

            suspicious_tlds = [
                ".tk",
                ".ml",
                ".ga",
                ".cf",
                ".gq",
                ".xyz",
                ".top",
                ".click",
            ]

            keyword_hits = [kw for kw in suspicious_keywords if kw in lower_url]
            tld_hits = [tld for tld in suspicious_tlds if tld in lower_url]

            score = 0.0

            if keyword_hits:
                score += 0.35
                url_result["reasons"].append(
                    f"Suspicious URL keywords found: {keyword_hits}"
                )

            if tld_hits:
                score += 0.35
                url_result["reasons"].append(
                    f"Suspicious TLD found: {tld_hits}"
                )

            if "@" in lower_url:
                score += 0.20
                url_result["reasons"].append("URL contains @ symbol")

            if score >= 0.70:
                url_result["action"] = "BLOCK"
            elif score >= 0.40:
                url_result["action"] = "QUARANTINE"
            elif score >= 0.20:
                url_result["action"] = "WARN"
            else:
                url_result["action"] = "ALLOW"

            url_result["suspicion_score"] = round(score, 3)

            if not url_result["reasons"]:
                url_result["reasons"].append("No suspicious URL patterns detected")

        current_rank = action_rank(url_result["action"])
        max_rank = max(max_rank, current_rank)
        max_score = max(max_score, float(url_result["suspicion_score"]))

        results.append(url_result)

    final_action = rank_to_action(max_rank)

    return {
        "verdict": final_action,
        "total_urls": len(urls),
        "max_score": round(max_score, 3),
        "blocked_urls": len([r for r in results if r["action"] == "BLOCK"]),
        "quarantine_urls": len([r for r in results if r["action"] == "QUARANTINE"]),
        "warned_urls": len([r for r in results if r["action"] == "WARN"]),
        "allowed_urls": len([r for r in results if r["action"] == "ALLOW"]),
        "results": results,
    }


# ---------------------------------------------------------
# Final combined verdict
# ---------------------------------------------------------

def get_confidence_percent(result: Optional[Dict[str, Any]]) -> float:
    if not result:
        return 0.0

    try:
        confidence = float(result.get("confidence", 0))
    except Exception:
        return 0.0

    # Existing modules return confidence as 0-100
    if confidence <= 1:
        return confidence * 100

    return confidence


def build_final_verdict(modules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines all scanners into one final frontend-friendly result.
    Final risk_score is 0-100.
    """

    score = 0.0
    reasons: List[str] = []

    # 1. Obfuscation score
    obf_result = modules.get("obfuscation", {}).get("result")

    if isinstance(obf_result, dict):
        obf_score = float(obf_result.get("risk_score", 0) or 0)

        # Max contribution: 25
        score += min(obf_score * 0.25, 25)

        obf_tokens = obf_result.get("obf_tokens") or obf_result.get("obfuscated_tokens") or []

        if obf_score >= 60:
            reasons.append("High obfuscation risk detected")
        elif obf_score >= 30:
            reasons.append("Medium obfuscation risk detected")
        elif obf_score > 0:
            reasons.append("Low obfuscation signals detected")

        if obf_tokens:
            reasons.append(f"Suspicious obfuscated tokens found: {obf_tokens[:8]}")

    # 2. Temporal evasion score
    temporal_result = modules.get("temporal_evasion", {}).get("result")

    if isinstance(temporal_result, dict):
        if temporal_result.get("is_threat") is True:
            confidence = get_confidence_percent(temporal_result)

            # Max contribution: 25
            score += 15 + min(confidence * 0.10, 10)

            reasons.append("Temporal evasion or suspicious sending pattern detected")

            indicators = temporal_result.get("indicators", [])
            if indicators:
                reasons.extend(indicators[:3])

    # 3. Header spoofing score
    header_result = modules.get("header_spoofing", {}).get("result")

    if isinstance(header_result, dict):
        if header_result.get("is_threat") is True:
            confidence = get_confidence_percent(header_result)

            # Max contribution: 25
            score += 15 + min(confidence * 0.10, 10)

            reasons.append("Header spoofing indicators detected")

            details = header_result.get("details", [])
            if details:
                reasons.extend(details[:3])

    # 4. URL threat score
    url_result = modules.get("url_threat", {}).get("result")

    if isinstance(url_result, dict):
        verdict = str(url_result.get("verdict", "ALLOW")).upper()
        max_url_score = float(url_result.get("max_score", 0) or 0)

        if verdict == "BLOCK":
            score += 25
            reasons.append("Malicious URL detected")
        elif verdict == "QUARANTINE":
            score += 20
            reasons.append("Suspicious URL requires quarantine")
        elif verdict == "WARN":
            score += 10
            reasons.append("URL warning detected")
        elif max_url_score > 0:
            score += min(max_url_score * 10, 5)

        total_urls = int(url_result.get("total_urls", 0) or 0)
        if total_urls > 0:
            reasons.append(f"{total_urls} URL(s) scanned")

    # 5. Failed modules
    failed_modules = []

    for module_name, module_data in modules.items():
        if not module_data.get("ok", False):
            failed_modules.append(module_name)

    if failed_modules:
        reasons.append(f"Some modules failed: {failed_modules}")

    score = round(min(score, 100), 1)

    if score >= 80:
        verdict = "CRITICAL"
        is_threat = True
    elif score >= 60:
        verdict = "HIGH"
        is_threat = True
    elif score >= 35:
        verdict = "MEDIUM"
        is_threat = True
    elif score >= 15:
        verdict = "LOW"
        is_threat = False
    else:
        verdict = "SAFE"
        is_threat = False

    if not reasons:
        reasons = ["No strong threat indicators detected"]

    return {
        "verdict": verdict,
        "is_threat": is_threat,
        "risk_score": score,
        "reasons": reasons,
        "failed_modules": failed_modules,
    }


def build_module_summary(modules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small summary for easy frontend display.
    """

    summary = {}

    obf = modules.get("obfuscation", {}).get("result")
    if isinstance(obf, dict):
        summary["obfuscation"] = {
            "ok": modules["obfuscation"]["ok"],
            "risk_score": obf.get("risk_score", 0),
            "detected_tokens": obf.get("obf_tokens") or obf.get("obfuscated_tokens") or [],
        }
    else:
        summary["obfuscation"] = {
            "ok": modules.get("obfuscation", {}).get("ok", False),
            "error": modules.get("obfuscation", {}).get("error"),
        }

    temporal = modules.get("temporal_evasion", {}).get("result")
    if isinstance(temporal, dict):
        summary["temporal_evasion"] = {
            "ok": modules["temporal_evasion"]["ok"],
            "is_threat": temporal.get("is_threat"),
            "risk_level": temporal.get("risk_level"),
            "confidence": temporal.get("confidence"),
            "model_used": temporal.get("model_used"),
        }
    else:
        summary["temporal_evasion"] = {
            "ok": modules.get("temporal_evasion", {}).get("ok", False),
            "error": modules.get("temporal_evasion", {}).get("error"),
        }

    header = modules.get("header_spoofing", {}).get("result")
    if isinstance(header, dict):
        summary["header_spoofing"] = {
            "ok": modules["header_spoofing"]["ok"],
            "is_threat": header.get("is_threat"),
            "risk_level": header.get("risk_level"),
            "confidence": header.get("confidence"),
            "model_used": header.get("model_used"),
        }
    else:
        summary["header_spoofing"] = {
            "ok": modules.get("header_spoofing", {}).get("ok", False),
            "error": modules.get("header_spoofing", {}).get("error"),
        }

    url = modules.get("url_threat", {}).get("result")
    if isinstance(url, dict):
        summary["url_threat"] = {
            "ok": modules["url_threat"]["ok"],
            "verdict": url.get("verdict"),
            "total_urls": url.get("total_urls", 0),
            "max_score": url.get("max_score", 0),
            "blocked_urls": url.get("blocked_urls", 0),
            "quarantine_urls": url.get("quarantine_urls", 0),
            "warned_urls": url.get("warned_urls", 0),
        }
    else:
        summary["url_threat"] = {
            "ok": modules.get("url_threat", {}).get("ok", False),
            "error": modules.get("url_threat", {}).get("error"),
        }

    return summary


# ---------------------------------------------------------
# Main full scan endpoint
# ---------------------------------------------------------

@router.post("/single-email")
def scan_single_email(email: SingleEmailScanRequest):
    """
    One endpoint for frontend single button.

    Frontend calls:
    POST /scan/single-email

    Backend runs:
    1. Obfuscation scan
    2. Temporal evasion scan
    3. Header spoofing scan
    4. URL threat scan
    5. Final combined verdict
    """

    sender = get_sender(email)
    body = get_body(email)
    sent_at = get_sent_at(email)

    modules = {
        "obfuscation": safe_run(
            "Obfuscation scan",
            lambda: run_obfuscation_scan(body),
        ),
        "temporal_evasion": safe_run(
            "Temporal evasion scan",
            lambda: run_temporal_scan(email),
        ),
        "header_spoofing": safe_run(
            "Header spoofing scan",
            lambda: run_header_spoofing_scan(email),
        ),
        "url_threat": safe_run(
            "URL threat scan",
            lambda: run_url_scan(email),
        ),
    }

    final_result = build_final_verdict(modules)
    module_summary = build_module_summary(modules)

    return {
        "email_id": email.id or email.message_id,
        "sender": sender,
        "subject": email.subject,
        "sent_at": sent_at,
        "final_result": final_result,
        "module_summary": module_summary,
        "modules": modules,
    }


@router.get("/status")
def full_scan_status():
    """
    Optional status endpoint.
    Useful to check whether this router is connected.
    """

    return {
        "module": "full_email_scan",
        "status": "ok",
        "endpoint": "/scan/single-email",
        "scanners": [
            "obfuscation",
            "temporal_evasion",
            "header_spoofing",
            "url_threat",
        ],
    }