from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse
from .email_store import EmailStore
import re

app = FastAPI(title="Obfuscation Detector API (Mock)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EmailStore()

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.get("/emails")
def list_emails():
    return store.list()

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
