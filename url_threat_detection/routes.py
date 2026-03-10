# url_threat_detection/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tldextract
 
from .url_classifier import rf_model, WHITELIST, feature_cols
from .url_classifier import PHISHING_KEYWORDS, MALICIOUS_PATHS
from .redirection_detector import URLRedirectionDetector
from .email_scanner import EmailURLScanner
from .feature_extractor import extract_features
 
router   = APIRouter(prefix="/api/url", tags=["URL Threat"])
detector = URLRedirectionDetector()
 
class URLRequest(BaseModel):
    url: str
 
class EmailRequest(BaseModel):
    email_body: str
 
 
@router.post("/predict")
async def predict(req: URLRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
 
    # ── STEP 1: Normalise URL (same as Colab) ─────────────────────────
    url_lower = url.lower()
    url_norm  = url_lower if url_lower.startswith(("http://","https://")) \
                else "http://" + url_lower
 
    # ── STEP 2: Whitelist gate — BEFORE any tracing or ML ────────────
    ext    = tldextract.extract(url_norm)
    domain = ext.domain + "." + ext.suffix if ext.suffix else ext.domain
 
    if domain in WHITELIST:
        has_phishing  = any(kw in url_lower for kw in PHISHING_KEYWORDS)
        has_malicious = any(p  in url_lower for p  in MALICIOUS_PATHS)
        if not has_phishing and not has_malicious:
            # ── Matches Colab: [OK] BENIGN (trusted domain, clean path)
            return {
                "original_url":    url,
                "action":          "ALLOW",
                "suspicion_score": 0.0,
                "is_malicious":    False,
                "ml_prediction":   "benign",
                "label":           "benign",
                "confidence":      1.0,
                "source":          "whitelist",
                "reasons":         ["Trusted domain — whitelist bypass"],
                "trace": {
                    "chain":[], "total_hops":0,
                    "final_url":url_norm, "total_ms":0
                }
            }
 
    # ── STEP 3: Not whitelisted — run full ML + redirection analysis ──
    result = detector.analyze_url(
        url_norm,
        ml_model=rf_model,
        feature_extractor=extract_features
    )
    return result
 
 
@router.post("/scan-email")
async def scan_email(req: EmailRequest):
    if not req.email_body:
        raise HTTPException(status_code=400, detail="email_body required")
    scanner = EmailURLScanner(ml_model=rf_model, feature_extractor=extract_features)
    return scanner.scan_email(req.email_body)
