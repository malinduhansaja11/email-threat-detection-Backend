# url_threat_detection/url_classifier.py
import joblib, os, pandas as pd
import tldextract
from .feature_extractor import extract_features
 
BASE         = os.path.dirname(__file__)
rf_model     = joblib.load(os.path.join(BASE, "models", "rf_model.pkl"))
feature_cols = joblib.load(os.path.join(BASE, "models", "feature_columns.pkl"))
 
LABEL_MAP = {0:"benign", 1:"phishing", 2:"malware", 3:"defacement"}
 
# ── BUG A FIX: Whitelist defined at module level (importable) ────────
WHITELIST = {
    "google.com", "github.com", "youtube.com", "facebook.com",
    "twitter.com", "linkedin.com", "microsoft.com", "apple.com",
    "amazon.com", "stackoverflow.com", "wikipedia.org",
    "instagram.com", "reddit.com", "netflix.com", "spotify.com"
}
 
PHISHING_KEYWORDS = [
    "login","signin","verify","secure","account","update",
    "confirm","password","bank","paypal","credential",
    "wallet","reset","unlock","suspend"
]
 
MALICIOUS_PATHS = [
    ".exe",".dll",".zip",".bat",".ps1",".vbs",".msi",
    "raw/master","raw/main","/download/","/payload/"
]
 
 
def predict_url(url: str) -> dict:
    # ── BUG A FIX: Normalise URL FIRST before any check ─────────────
    url       = str(url).strip()
    url_lower = url.lower()
    url_norm  = url_lower if url_lower.startswith(("http://","https://")) \
                else "http://" + url_lower
 
    # ── BUG A FIX: Extract domain from NORMALISED url ────────────────
    ext    = tldextract.extract(url_norm)
    domain = ext.domain + "." + ext.suffix if ext.suffix else ext.domain
 
    # ── Smart whitelist — same logic as Colab predict_url() ──────────
    if domain in WHITELIST:
        has_phishing  = any(kw in url_lower for kw in PHISHING_KEYWORDS)
        has_malicious = any(p  in url_lower for p  in MALICIOUS_PATHS)
        if not has_phishing and not has_malicious:
            return {
                "label":      "benign",
                "confidence": 1.0,
                "source":     "whitelist",
                "domain":     domain
            }
 
    # ── BUG B FIX: Extract features from NORMALISED url ──────────────
    raw_feats = extract_features(url_norm)
    feats_df  = pd.DataFrame([raw_feats])
 
    # ── BUG B FIX: Force EXACT column order matching training ─────────
    for col in feature_cols:
        if col not in feats_df.columns:
            feats_df[col] = 0       # add any column the model expects
    feats_df = feats_df[feature_cols]  # reorder to match training order
    feats_df = feats_df.fillna(0)
 
    pred  = int(rf_model.predict(feats_df)[0])
    proba = rf_model.predict_proba(feats_df)[0].tolist()
 
    return {
        "label":      LABEL_MAP[pred],
        "confidence": round(max(proba), 4),
        "source":     "ml_model",
        "domain":     domain
    }
