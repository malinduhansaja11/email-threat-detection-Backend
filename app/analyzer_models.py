import os
import joblib
import torch
import numpy as np
import scipy.sparse as sp
import re
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

TFIDF_WORD_PATH = os.path.join(MODELS_DIR, "tfidf_word.pkl")
TFIDF_CHAR_PATH = os.path.join(MODELS_DIR, "tfidf_char.pkl")
CLASSICAL_MODEL_PATH = os.path.join(MODELS_DIR, "classical_model.pkl")
BERT_MODEL_DIR = MODELS_DIR

_tfidf_word = None
_tfidf_char = None
_classical_model = None
_tokenizer = None
_bert_model = None


def load_classical_bundle():
    global _tfidf_word, _tfidf_char, _classical_model

    if _tfidf_word is None:
        _tfidf_word = joblib.load(TFIDF_WORD_PATH)

    if _tfidf_char is None:
        _tfidf_char = joblib.load(TFIDF_CHAR_PATH)

    if _classical_model is None:
        _classical_model = joblib.load(CLASSICAL_MODEL_PATH)

    return _tfidf_word, _tfidf_char, _classical_model


def load_bert_bundle():
    global _tokenizer, _bert_model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)

    if _bert_model is None:
        _bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        _bert_model.eval()

    return _tokenizer, _bert_model


def extract_extra_features(email_text: str):
    text = email_text or ""
    tokens = text.split()

    num_urls = len(re.findall(r"(https?://\S+|www\.\S+)", text))
    num_emails = len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    num_digits = sum(ch.isdigit() for ch in text)
    num_special = sum(ch in "@$_%><:/\\|;=" for ch in text)
    num_upper = sum(ch.isupper() for ch in text)
    num_tokens = len(tokens)
    avg_token_len = (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0

    num_suspicious_tokens = sum(
        1
        for t in tokens
        if any(ch in t for ch in "@$_%><:/\\|;=") or any(ch.isdigit() for ch in t)
    )

    text_len = len(text)

    return np.array(
        [[
            num_urls,
            num_emails,
            num_digits,
            num_special,
            num_upper,
            num_tokens,
            avg_token_len,
            num_suspicious_tokens,
            text_len,
        ]],
        dtype=float,
    )


def classical_predict(email_text: str) -> Dict[str, Any]:
    tfidf_word, tfidf_char, model = load_classical_bundle()

    X_word = tfidf_word.transform([email_text])
    X_char = tfidf_char.transform([email_text])

    X_base = sp.hstack([X_word, X_char])
    X_extra = sp.csr_matrix(extract_extra_features(email_text))

    X = sp.hstack([X_base, X_extra])

    pred = model.predict(X)[0]
    score = 0.9 if int(pred) == 1 else 0.05

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            if len(proba) > 1:
                score = float(proba[1])
        except Exception:
            pass

    elif hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)[0]
            score = 1 / (1 + np.exp(-float(decision)))
        except Exception:
            pass

    return {
        "label": int(pred),
        "score": float(score),
    }


def bert_predict(email_text: str) -> Dict[str, Any]:
    tokenizer, model = load_bert_bundle()

    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    if len(probs) > 1:
        suspicious_prob = float(probs[1].item())
        pred = 1 if suspicious_prob >= 0.5 else 0
    else:
        suspicious_prob = float(probs[0].item())
        pred = 1 if suspicious_prob >= 0.5 else 0

    return {
        "label": int(pred),
        "score": float(suspicious_prob),
    }