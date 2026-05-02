import re
from typing import Dict, Any, List

from .analyzer_models import classical_predict, bert_predict
from .gmail_services import (
    TOKEN_RE,
    clean_token,
    is_whitelisted,
    extract_suspicious_tokens,
)


def analyze_body(email_text: str) -> Dict[str, Any]:
    text = email_text or ""
    tokens = TOKEN_RE.findall(text)

    classical_res = classical_predict(text)
    bert_res = bert_predict(text)

    # Colab-like final decision
    final_label = 1 if (classical_res["label"] == 1 or bert_res["label"] == 1) else 0

    final_score = (classical_res["score"] + bert_res["score"]) / 2

    suspicious_tokens = extract_suspicious_tokens(text)
    suspicious_set = set(suspicious_tokens)

    labels: List[int] = []
    scores: List[float] = []
    obf_tokens: List[str] = []

    for tok in tokens:
        cleaned_tok = clean_token(tok)

        if not cleaned_tok:
            labels.append(0)
            scores.append(0.05)
            continue

        # Whitelist tokens never highlight
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

    # -----------------------------
    # Risk score calculation
    # Gradually increases by obfuscation token count
    # -----------------------------

    obf_count = len(obf_tokens)

    if obf_count == 0:
        risk_score = 0

    elif obf_count <= 10:
        # 1 token = 14%, 5 tokens = 30%, 8 tokens = 42%, 10 tokens = 50%
        risk_score = 10 + (obf_count * 4)

    else:
        # More than 10 tokens starts from 50% and increases gradually
        # 11 = 52%, 15 = 60%, 20 = 70%, 25 = 80%, 30 = 90%
        risk_score = 50 + ((obf_count - 10) * 2)

    risk_score = int(min(95, risk_score))

    if risk_score <30:
        risk_level = "LOW"
    elif risk_score < 50:
        risk_level = "MEDIUM"
    elif risk_score < 75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "tokens": tokens,
        "labels": labels,
        "scores": scores,
        "obf_tokens": obf_tokens,
        "has_obfuscation": obf_count > 0,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "meta": {
            "classical_label": classical_res["label"],
            "classical_score": classical_res["score"],
            "bert_label": bert_res["label"],
            "bert_score": bert_res["score"],
            "final_label": final_label,
            "final_score": final_score,
            "obf_count": obf_count,
        },
    }