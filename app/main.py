import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .inbox_scanner import scan_inbox
from .schemas import PredictRequest, PredictResponse 
from .email_store import EmailStore
from .analyzer import analyze_body

from .gmail_oauth import get_auth_url, exchange_code_for_token, load_credentials, clear_token
from .gmail_fetch import fetch_inbox

# ── NEW: Kaveesha's Temporal-Evasion module ───────────────────────────────────
from .temporal_evasion.routes import router as temporal_evasion_router
# ─────────────────────────────────────────────────────────────────────────────

# ── NEW: Ayani's Header-Spoofing module ───────────────────────────────────────
from .header_spoofing.routes import router as header_spoofing_router
# ─────────────────────────────────────────────────────────────────────────────

from .full_email_scan import router as full_email_scan_router

# === Pramudika — URL Threat Detection ===
from url_threat_detection.routes import router as url_router
app = FastAPI(title="Obfuscation Detector API")
app.include_router(url_router)
app.include_router(header_spoofing_router)
app.include_router(temporal_evasion_router)
app.include_router(full_email_scan_router)

# NEW: analyzer (ML model)




ENV_PATH = Path(__file__).resolve().parents[1] / ".env"   # backend/.env
load_dotenv(ENV_PATH)

print("GOOGLE_CLIENT_ID:", os.getenv("GOOGLE_CLIENT_ID"))
print("GOOGLE_CLIENT_SECRET:", "SET" if os.getenv("GOOGLE_CLIENT_SECRET") else None)


store = EmailStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EmailStore()



@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# OLD MOCK DETECTOR (keep)
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tokens = [t for t in re.split(r"\s+", req.body.strip()) if t]
    labels = [1 if re.search(r"[@_$\d]", t) else 0 for t in tokens]
    scores = [0.9 if lbl == 1 else 0.05 for lbl in labels]
    obf_tokens = [t for t, lbl in zip(tokens, labels) if lbl == 1]
    risk_score = 0 if len(tokens) == 0 else round((len(obf_tokens) / len(tokens)) * 100)

    return PredictResponse(
        tokens=tokens,
        labels=labels,
        scores=scores,
        obf_tokens=obf_tokens,
        risk_score=risk_score,
    )


# ----------------------------
# NEW ML ANALYZER
# ----------------------------

    
# @app.post("/analyze")
# def analyze(req: PredictRequest):
#     try:
#         return analyze_body(req.body, req.model_type)
#     except Exception as e:
#         import traceback
#         return {
#             "error": str(e),
#             "trace": traceback.format_exc()
#         }

@app.post("/analyze", response_model=PredictResponse)
def analyze(req: PredictRequest):
    return analyze_body(req.body)


# ----------------------------
# EMAIL STORE / GMAIL
# ----------------------------
@app.get("/emails")
def list_emails():
    return store.list()


@app.get("/emails/gmail")
def gmail_emails():
    emails = fetch_inbox(max_results=15)
    if emails is None:
        return {"connected": False, "emails": []}
    return {"connected": True, "emails": emails}


# ── Kaveesha: AUTO-SCAN — must be before /emails/{email_id} ─────────────────
@app.get("/emails/scan")
def scan_all_emails():
    """
    Fetch inbox emails and auto-run Kaveesha temporal model on each.
    Returns every email enriched with a 'threat' object.
    """
    try:
        gmail_result = scan_inbox(source="gmail", max_results=20)
        if gmail_result["connected"]:
            return gmail_result
    except Exception:
        pass
    return scan_inbox(source="demo", max_results=20)


@app.get("/emails/{email_id}")
def get_email(email_id: str):
    item = store.get(email_id)
    if not item:
        raise HTTPException(status_code=404, detail="Email not found")
    return item



@app.post("/emails/seed")
def seed_emails():
    store.add(
        sender="billing@demo.com",
        subject="Invoice Update",
        body="Please verify your invoice2025 details and api_v2 changes."
    )
    store.add(
        sender="security@demo.com",
        subject="Account Alert",
        body="Urgent: reset your p@ssw0rd now. Click link ASAP."
    )
    store.add(
        sender="team@demo.com",
        subject="Meeting Notes",
        body="Hi team, meeting is at 3pm. Agenda shared in doc."
    )

    return {"status": "seeded", "count": len(store.list())}


# ----------------------------
# AUTH
# ----------------------------
@app.get("/auth/status")
def auth_status():
    try:
        return {"connected": load_credentials() is not None}
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.post("/auth/logout")
def logout():
    clear_token()
    return {"ok": True}


@app.get("/auth/google/login")
def google_login():
    url = get_auth_url()
    return RedirectResponse(url)


@app.get("/auth/google/callback")
def google_callback(request: Request, code: str | None = None):
    if not code:
        return {"error": "Missing code", "query": dict(request.query_params)}

    try:
        exchange_code_for_token(code)
    except Exception as e:
        return {"error": str(e), "query": dict(request.query_params)}

    frontend = os.getenv("FRONTEND_REDIRECT", "http://localhost:5173/inbox")
    return RedirectResponse(f"{frontend}?connected=1")



