import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from firebase_admin import firestore

from .analyzer import analyze_body
from .auth_guard import db as firebase_db
from .auth_guard import require_approved_firebase_user
from .email_store import EmailStore
from .full_email_scan import router as full_email_scan_router
from .gmail_fetch import fetch_inbox
from .gmail_oauth import (
    clear_token,
    exchange_code_for_token,
    get_auth_url,
    load_credentials,
    make_oauth_state,
    read_oauth_state,
)
from .header_spoofing.routes import router as header_spoofing_router
from .inbox_scanner import scan_inbox
from .schemas import PredictRequest, PredictResponse
from .temporal_evasion.routes import router as temporal_evasion_router

# Pramudika — URL Threat Detection
from url_threat_detection.routes import router as url_router

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Obfuscation Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(url_router)
app.include_router(header_spoofing_router)
app.include_router(temporal_evasion_router)
app.include_router(full_email_scan_router)

store = EmailStore()

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# OLD MOCK DETECTOR
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
# ML ANALYZER
# ----------------------------
@app.post("/analyze", response_model=PredictResponse)
def analyze(req: PredictRequest):
    return analyze_body(req.body)


# ----------------------------
# EMAIL STORE / GMAIL
# ----------------------------
@app.get("/emails")
def list_emails(user=Depends(require_approved_firebase_user)):
    return store.list()


@app.post("/emails/seed")
def seed_emails(user=Depends(require_approved_firebase_user)):
    store.add(
        sender="billing@demo.com",
        subject="Invoice Update",
        body="Please verify your invoice2025 details and api_v2 changes.",
    )

    store.add(
        sender="security@demo.com",
        subject="Account Alert",
        body="Urgent: reset your p@ssw0rd now. Click link ASAP.",
    )

    store.add(
        sender="team@demo.com",
        subject="Meeting Notes",
        body="Hi team, meeting is at 3pm. Agenda shared in doc.",
    )

    return {"status": "seeded", "count": len(store.list())}


@app.get("/emails/gmail")
def gmail_emails(user=Depends(require_approved_firebase_user)):
    uid = user["uid"]

    try:
        emails = fetch_inbox(uid=uid, max_results=15)

        if emails is None:
            return {
                "connected": False,
                "emails": [],
                "message": "Gmail is not connected for this user",
            }

        return {
            "connected": True,
            "emails": emails,
        }

    except Exception as e:
        error_message = str(e)
        print("\n================ GMAIL FETCH ERROR ================")
        print(error_message)
        print("===================================================\n")

        return {
            "connected": False,
            "emails": [],
            "error": error_message,
        }


@app.get("/emails/scan")
def scan_all_emails(user=Depends(require_approved_firebase_user)):
    uid = user["uid"]

    try:
        gmail_result = scan_inbox(uid=uid, source="gmail", max_results=20)

        if gmail_result["connected"]:
            return gmail_result

    except Exception:
        pass

    return scan_inbox(source="demo", max_results=20)


@app.get("/emails/{email_id}")
def get_email(email_id: str, user=Depends(require_approved_firebase_user)):
    item = store.get(email_id)

    if not item:
        raise HTTPException(status_code=404, detail="Email not found")

    return item


# ----------------------------
# AUTH / GMAIL BINDING
# ----------------------------
@app.get("/auth/status")
def auth_status(user=Depends(require_approved_firebase_user)):
    uid = user["uid"]

    try:
        creds = load_credentials(uid)

        return {
            "connected": creds is not None,
        }

    except Exception as e:
        print("AUTH STATUS ERROR:", str(e))

        return {
            "connected": False,
            "error": str(e),
        }

@app.post("/auth/google/start")
def google_start(user=Depends(require_approved_firebase_user)):
    uid = user["uid"]

    state = make_oauth_state(uid)
    url = get_auth_url(state)

    return {"url": url}


@app.get("/auth/google/callback")
def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
):
    if not code or not state:
        return {
            "error": "Missing code or state",
            "query": dict(request.query_params),
        }

    try:
        uid = read_oauth_state(state)
        exchange_code_for_token(code=code, uid=uid)

        firebase_db.collection("users").document(uid).set(
            {
                "gmailConnected": True,
                "gmailBoundAt": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    except Exception as e:
        return {
            "error": str(e),
            "query": dict(request.query_params),
        }

    frontend = os.getenv("FRONTEND_REDIRECT", "http://localhost:5173/inbox")
    return RedirectResponse(f"{frontend}?connected=1")


@app.post("/auth/logout")
def logout(user=Depends(require_approved_firebase_user)):
    uid = user["uid"]

    clear_token(uid)

    firebase_db.collection("users").document(uid).set(
        {
            "gmailConnected": False,
            "gmailDisconnectedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    return {"ok": True}