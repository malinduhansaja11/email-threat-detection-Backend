import os
import re
import joblib
from typing import Dict, Any, List

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"\S+")  # keeps symbols (important for obfuscations)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "obfuscation_token_classifier.joblib")

_model = None


def get_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}. Put it in backend/app/models/"
            )
        _model = joblib.load(MODEL_PATH)

    return _model


def analyze_body(body: str) -> Dict[str, Any]:
    model = get_model()

    tokens: List[str] = TOKEN_RE.findall(body or "")
    emails = EMAIL_RE.findall(body or "")
    urls = URL_RE.findall(body or "")

    token_labels = []
    obf_tokens = []
    scores = []

    has_proba = hasattr(model, "predict_proba")

    for t in tokens:
        label = model.predict([t])[0]  # "email" / "url" / "obfuscation_word" / "normal_word"
        token_labels.append(label)

        if label == "obfuscation_word":
            obf_tokens.append(t)

        if has_proba:
            probs = model.predict_proba([t])[0]
            scores.append(float(probs.max()))
        else:
            scores.append(1.0 if label != "normal_word" else 0.0)

    risk_score = 0 if len(tokens) == 0 else round((len(obf_tokens) / len(tokens)) * 100)

    return {
        "tokens": tokens,
        "labels": token_labels,
        "scores": scores,
        "emails": sorted(list(set(emails))),
        "urls": sorted(list(set(urls))),
        "obf_tokens": sorted(list(set(obf_tokens))),
        "risk_score": risk_score,
    }
