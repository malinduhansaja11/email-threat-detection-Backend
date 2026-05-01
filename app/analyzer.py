import os
import re
import joblib
import torch
import numpy as np
import scipy.sparse as sp
import unicodedata

from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .listing import whitelist_words, whitelist_patterns

TOKEN_RE = re.compile(r"\S+")

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

def clean_token(token: str) -> str:
    t = token.strip()
    

    # remove opening punctuation
    t = t.lstrip('(<[{\'"')

    # remove trailing punctuation
    t = t.rstrip('.,!?;:)]}\'"')

    if t.endswith('.'):
        t = t[:-1]

    return t

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
        1 for t in tokens
        if any(ch in t for ch in "@$_%><:/\\|;=") or any(ch.isdigit() for ch in t)
    )
    text_len = len(text)

    return np.array([[
        num_urls,
        num_emails,
        num_digits,
        num_special,
        num_upper,
        num_tokens,
        avg_token_len,
        num_suspicious_tokens,
        text_len,
    ]], dtype=float)


def classical_predict(email_text: str) -> Dict[str, Any]:
    tfidf_word, tfidf_char, model = load_classical_bundle()

    X_word = tfidf_word.transform([email_text])
    X_char = tfidf_char.transform([email_text])

    # TF-IDF features
    X_base = sp.hstack([X_word, X_char])

    # 9 handcrafted features
    X_extra = sp.csr_matrix(extract_extra_features(email_text))

    # Final feature vector => 50009 features
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


LEET_MAP = str.maketrans({
    "@": "a",
    "4": "a",
    "$": "s",
    "5": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "3": "e",
    "7": "t",
    "+": "t",
    "8": "b",
    "6": "g",
    "9": "g",
})

SUSPICIOUS_KEYWORDS = [
    "password", "passcode", "passwd",
    "login", "logon", "signin", "signon",
    "verify", "verification",
    "account", "security", "secure",
    "update", "invoice", "payment",
    "bank", "paypal", "crypto", "wallet",
    "click", "urgent", "reset",
    "giftcard", "otp", "pin",
    "microsoft", "google", "apple", "facebook",
    "gmail", "outlook", "github",
    "viagra", "casino", "lottery",
]

SYMBOL_SEPARATORS = ["-", "_", ".", ",", ":", ";", "/", "\\", "|", "~", "^"]

SUSPICIOUS_SYMBOLS = [
    "@", "$", "%", ">", "<", "/", "\\", "|", ":", ";", "=", "!", "#", "*",
    "(", ")", "[", "]", "{", "}", "~", "^", "+"
]


def normalize_token(token: str) -> str:
    t = unicodedata.normalize("NFKC", token.lower().strip())
    t = t.translate(LEET_MAP)
    # remove common separators for joined-word checking
    for sep in SYMBOL_SEPARATORS:
        t = t.replace(sep, "")
    return t


def has_weird_unicode(token: str) -> bool:
    for ch in token:
        if ord(ch) > 127:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("N") or cat.startswith("S"):
                return True
    return False


def has_mixed_script_like_pattern(token: str) -> bool:
    # simple heuristic for strange obfuscation-looking text
    alpha = sum(ch.isalpha() for ch in token)
    digit = sum(ch.isdigit() for ch in token)
    symbol = sum(not ch.isalnum() and not ch.isspace() for ch in token)
    return (alpha > 0 and digit > 0) or (alpha > 0 and symbol > 0)


def looks_like_split_keyword(token: str) -> bool:
    n = normalize_token(token)
    return any(keyword in n for keyword in SUSPICIOUS_KEYWORDS)


def has_repeated_separators(token: str) -> bool:
    patterns = ["..", "__", "--", "::", "//", "\\\\", "==", "~~", "^^"]
    return any(p in token for p in patterns)


def looks_like_obfuscated_domain(token: str) -> bool:
    t = token.lower()
    if "http" in t or "www." in t:
        return True
    if "." in t and any(ch.isalpha() for ch in t):
        # domain-like token
        return True
    return False


def looks_like_obfuscated_email(token: str) -> bool:
    t = token.lower()
    return "@" in t and "." in t


def is_whitelisted(token: str) -> bool:
    raw_token = token.strip()
    cleaned_token = clean_token(raw_token)

    raw_lower = raw_token.lower()
    cleaned_lower = cleaned_token.lower()

    if raw_lower in whitelist_words or cleaned_lower in whitelist_words:
        return True

    candidates = [raw_token, cleaned_token]

    for pattern in whitelist_patterns:
        try:
            for candidate in candidates:
                if re.fullmatch(pattern, candidate):
                    return True
        except re.error:
            continue

    return False

def clean_token(token: str) -> str:
    t = token.strip()
    t = t.lstrip('(<[{\'"')
    t = t.rstrip('.,!?;:)]}\'"')
    return t

def extract_suspicious_tokens(email_text: str) -> List[str]:
    tokens = TOKEN_RE.findall(email_text or "")
    suspicious = []

    for tok in tokens:
        cleaned_tok = clean_token(tok)

        if not cleaned_tok:
            continue

        # whitelist skip
        if is_whitelisted(cleaned_tok):
            continue

        token_lower = cleaned_tok.lower()

        has_digit = any(ch.isdigit() for ch in cleaned_tok)
        has_symbol = any(ch in cleaned_tok for ch in [
            "@", "$", "_", "%", ">", "<", "\\", "|", "=", "#", "*", "~", "^", "+"
        ])

        # URLs / emails
        has_url = token_lower.startswith("http://") or token_lower.startswith("https://") or token_lower.startswith("www.")
        has_email = "@" in cleaned_tok and "." in cleaned_tok

        alpha_count = sum(ch.isalpha() for ch in cleaned_tok)
        digit_count = sum(ch.isdigit() for ch in cleaned_tok)
        symbol_count = sum((not ch.isalnum() and not ch.isspace()) for ch in cleaned_tok)

        has_mixed_alpha_digit = alpha_count > 0 and digit_count > 0
        has_mixed_alpha_symbol = alpha_count > 0 and symbol_count > 0
        

        if (
            has_url
            or has_email
            or has_mixed_alpha_digit
            or has_mixed_alpha_symbol
            or (has_digit and len(cleaned_tok) > 3)
            or has_symbol
        ):
            suspicious.append(cleaned_tok)

    return suspicious


def analyze_body(email_text: str) -> Dict[str, Any]:
    text = email_text or ""
    tokens = TOKEN_RE.findall(text)

    classical_res = classical_predict(text)
    bert_res = bert_predict(text)

    # Colab-like final decision
    final_label = 1 if (classical_res["label"] == 1 or bert_res["label"] == 1) else 0
    final_score = (classical_res["score"] + bert_res["score"]) / 4

    # ✅ Step 5 - suspicious token extraction
    suspicious_tokens = extract_suspicious_tokens(text)

    labels: List[int] = []
    scores: List[float] = []
    obf_tokens: List[str] = []

    suspicious_set = set(suspicious_tokens)

    for tok in tokens:
        cleaned_tok = clean_token(tok)

        

        if not cleaned_tok:
            labels.append(0)
            scores.append(0.05)
            continue

        # whitelist tokens never highlight
        if is_whitelisted(cleaned_tok):
            labels.append(0)
            scores.append(0.05)
            continue

        if final_label == 1 and cleaned_tok in suspicious_set:
            labels.append(1)
            scores.append(float(final_score))
            obf_tokens.append(cleaned_tok)
        else:
            labels.append(0)
            scores.append(0.05)
    

   




   # risk_score = round(float(final_score) * 100)
    risk_score = int(min(95, final_score * 100))

    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 40:
        risk_level = "MEDIUM"
    elif risk_score < 50:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "tokens": tokens,
        "labels": labels,
        "scores": scores,
        "obf_tokens": obf_tokens,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "meta": {
            "classical_label": classical_res["label"],
            "classical_score": classical_res["score"],
            "bert_label": bert_res["label"],
            "bert_score": bert_res["score"],
            "final_label": final_label,
            "final_score": final_score,
        },
    }