
# import os
# import re
# import joblib
# from typing import Dict, Any, List

# EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
# TOKEN_RE = re.compile(r"\S+")  # keeps symbols (important for obfuscations)

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "obfuscation_token_classifier.joblib")

# _model = None


# def get_model():
#     global _model

#     if _model is None:
#         if not os.path.exists(MODEL_PATH):
#             raise FileNotFoundError(
#                 f"Model not found at: {MODEL_PATH}. Put it in backend/app/models/"
#             )
#         _model = joblib.load(MODEL_PATH)

#     return _model


# def analyze_body(body: str) -> Dict[str, Any]:
#     model = get_model()

#     tokens: List[str] = TOKEN_RE.findall(body or "")
#     emails = EMAIL_RE.findall(body or "")
#     urls = URL_RE.findall(body or "")

#     token_labels = []
#     obf_tokens = []
#     scores = []

#     has_proba = hasattr(model, "predict_proba")

#     for t in tokens:
#         label = model.predict([t])[0]  # "email" / "url" / "obfuscation_word" / "normal_word"
#         token_labels.append(label)

#         if label == "obfuscation_word":
#             obf_tokens.append(t)

#         if has_proba:
#             probs = model.predict_proba([t])[0]
#             scores.append(float(probs.max()))
#         else:
#             scores.append(1.0 if label != "normal_word" else 0.0)

#     risk_score = 0 if len(tokens) == 0 else round((len(obf_tokens) / len(tokens)) * 100)

#     return {
#         "tokens": tokens,
#         "labels": token_labels,
#         "scores": scores,
#         "emails": sorted(list(set(emails))),
#         "urls": sorted(list(set(urls))),
#         "obf_tokens": sorted(list(set(obf_tokens))),
#         "risk_score": risk_score, 
#     } 












# import os
# import re
# import joblib
# import pickle
# from typing import Dict, Any, List

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
# TOKEN_RE = re.compile(r"\S+")

# MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# PKL_MODEL_PATH = os.path.join(MODELS_DIR, "email_pipeline_new.pkl")
# HF_MODEL_DIR = MODELS_DIR  # config.json, tokenizer.json, model.safetensors all here

# _pkl_model = None
# _hf_tokenizer = None
# _hf_model = None


# def load_pkl_model():
#     global _pkl_model
#     if _pkl_model is not None:
#         return _pkl_model

#     if not os.path.exists(PKL_MODEL_PATH):
#         raise FileNotFoundError(f"PKL model not found: {PKL_MODEL_PATH}")

#     try:
#         _pkl_model = joblib.load(PKL_MODEL_PATH)
#     except Exception:
#         with open(PKL_MODEL_PATH, "rb") as f:
#             _pkl_model = pickle.load(f)

#     return _pkl_model


# def load_hf_model():
#     global _hf_tokenizer, _hf_model

#     if _hf_tokenizer is not None and _hf_model is not None:
#         return _hf_tokenizer, _hf_model

#     required_files = [
#         os.path.join(HF_MODEL_DIR, "config.json"),
#         os.path.join(HF_MODEL_DIR, "tokenizer.json"),
#         os.path.join(HF_MODEL_DIR, "tokenizer_config.json"),
#         os.path.join(HF_MODEL_DIR, "model.safetensors"),
#     ]

#     for f in required_files:
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Required HF file missing: {f}")

#     _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
#     _hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
#     _hf_model.eval()

#     return _hf_tokenizer, _hf_model


# def predict_with_pkl(tokens: List[str]) -> Dict[str, Any]:
#     model = load_pkl_model()

#     labels: List[int] = []
#     scores: List[float] = []

#     for token in tokens:
#         try:
#             pred = model.predict([token])[0]
#             label = int(pred)
#             labels.append(label)

#             # probability if available
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba([token])[0]
#                 if len(proba) > 1:
#                     scores.append(float(proba[1]))
#                 else:
#                     scores.append(float(label))
#             else:
#                 scores.append(0.9 if label == 1 else 0.05)

#         except Exception:
#             labels.append(0)
#             scores.append(0.05)

#     return {
#         "labels": labels,
#         "scores": scores,
#     }


# def predict_with_safetensors(text: str, tokens: List[str]) -> Dict[str, Any]:
#     """
#     IMPORTANT:
#     This assumes the HF model is a sequence classifier (whole email classification),
#     not token classification. Therefore it predicts one score for the whole email,
#     then applies that score to suspicious tokens heuristically.
#     """
#     tokenizer, model = load_hf_model()

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=-1)[0]

#     # assume binary classification: index 1 = suspicious
#     if probs.shape[0] > 1:
#         suspicious_prob = float(probs[1].item())
#     else:
#         suspicious_prob = float(probs[0].item())

#     labels: List[int] = []
#     scores: List[float] = []

#     # Heuristic token mapping from whole-email score
#     for token in tokens:
#         suspicious_token = bool(
#             EMAIL_RE.search(token)
#             or URL_RE.search(token)
#             or any(ch in token for ch in ["@", "$", "_", "%"])
#             or any(ch.isdigit() for ch in token)
#         )

#         if suspicious_prob >= 0.5 and suspicious_token:
#             labels.append(1)
#             scores.append(suspicious_prob)
#         else:
#             labels.append(0)
#             scores.append(min(suspicious_prob, 0.49))

#     return {
#         "labels": labels,
#         "scores": scores,
#         "email_level_score": suspicious_prob,
#     }


# def combine_predictions(
#     pkl_result: Dict[str, Any],
#     hf_result: Dict[str, Any],
# ) -> Dict[str, Any]:
#     final_labels: List[int] = []
#     final_scores: List[float] = []

#     for pkl_label, pkl_score, hf_label, hf_score in zip(
#         pkl_result["labels"],
#         pkl_result["scores"],
#         hf_result["labels"],
#         hf_result["scores"],
#     ):
#         score = (float(pkl_score) + float(hf_score)) / 2.0
#         label = 1 if score >= 0.5 or (pkl_label == 1 and hf_label == 1) else 0
#         final_labels.append(label)
#         final_scores.append(score)

#     return {
#         "labels": final_labels,
#         "scores": final_scores,
#         "hf_email_level_score": hf_result.get("email_level_score", 0.0),
#     }


# def analyze_body(text: str, model_type: str = "combined") -> Dict[str, Any]:
#     tokens = TOKEN_RE.findall(text or "")

#     if model_type == "pkl":
#         pkl_result = predict_with_pkl(tokens)
#         final_labels = pkl_result["labels"]
#         final_scores = pkl_result["scores"]
#         meta = {"model_used": "pkl"}

#     elif model_type == "safetensors":
#         hf_result = predict_with_safetensors(text, tokens)
#         final_labels = hf_result["labels"]
#         final_scores = hf_result["scores"]
#         meta = {
#             "model_used": "safetensors",
#             "email_level_score": hf_result.get("email_level_score", 0.0),
#         }

#     elif model_type == "combined":
#         pkl_result = predict_with_pkl(tokens)
#         hf_result = predict_with_safetensors(text, tokens)
#         combined = combine_predictions(pkl_result, hf_result)
#         final_labels = combined["labels"]
#         final_scores = combined["scores"]
#         meta = {
#             "model_used": "combined",
#             "hf_email_level_score": combined.get("hf_email_level_score", 0.0),
#         }

#     else:
#         raise ValueError("model_type must be one of: pkl, safetensors, combined")

#     obf_tokens = [tok for tok, lbl in zip(tokens, final_labels) if lbl == 1]
#     risk_score = 0 if len(tokens) == 0 else round((len(obf_tokens) / len(tokens)) * 100)

#     return {
#         "tokens": tokens,
#         "labels": final_labels,
#         "scores": final_scores,
#         "obf_tokens": obf_tokens,
#         "risk_score": risk_score,
#         "meta": meta,
#     }







































# import os
# import re
# import joblib
# import pickle
# from typing import Dict, Any, List
# from scipy.sparse import hstack, csr_matrix

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# TOKEN_RE = re.compile(r"\S+")
# EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

# MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# PKL_MODEL_PATH = os.path.join(MODELS_DIR, "email_pipeline_new.pkl")
# HF_MODEL_DIR = MODELS_DIR

# _pkl_model = None
# _hf_tokenizer = None
# _hf_model = None


# def load_pkl_model():
#     global _pkl_model
#     if _pkl_model is not None:
#         return _pkl_model

#     if not os.path.exists(PKL_MODEL_PATH):
#         raise FileNotFoundError(f"PKL model not found: {PKL_MODEL_PATH}")

#     try:
#         _pkl_model = joblib.load(PKL_MODEL_PATH)
#     except Exception:
#         with open(PKL_MODEL_PATH, "rb") as f:
#             _pkl_model = pickle.load(f)

#     return _pkl_model




# def load_hf_model():
#     global _hf_tokenizer, _hf_model

#     if _hf_tokenizer is not None and _hf_model is not None:
#         return _hf_tokenizer, _hf_model

#     required_files = [
#         os.path.join(HF_MODEL_DIR, "config.json"),
#         os.path.join(HF_MODEL_DIR, "tokenizer.json"),
#         os.path.join(HF_MODEL_DIR, "tokenizer_config.json"),
#         os.path.join(HF_MODEL_DIR, "model.safetensors"),
#     ]

#     for f in required_files:
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Required HF file missing: {f}")

#     _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
#     _hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
#     _hf_model.eval()

#     return _hf_tokenizer, _hf_model






# def predict_email_level_with_pkl(text: str) -> Dict[str, Any]:
#     loaded = load_pkl_model()

#     loaded_bundle = load_pkl_model()
#     print("WHITELIST WORDS:", loaded_bundle.get("whitelist_words", []))
#     print("WHITELIST PATTERNS:", loaded_bundle.get("whitelist_patterns", []))

#     if not isinstance(loaded, dict):
#         raise ValueError("Loaded PKL is not a dict bundle as expected.")

#     required_keys = ["tfidf_word", "tfidf_char", "classical_model"]
#     for key in required_keys:
#         if key not in loaded:
#             raise ValueError(f"Missing key in PKL bundle: {key}")

#     tfidf_word = loaded["tfidf_word"]
#     tfidf_char = loaded["tfidf_char"]
#     model = loaded["classical_model"]

#     X_word = tfidf_word.transform([text])
#     X_char = tfidf_char.transform([text])

#     # base TF-IDF features
#     X_base = hstack([X_word, X_char])

#     # temporary 9 extra handcrafted features
#     extra_features = csr_matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

#     X = hstack([X_base, extra_features])

#     pred = model.predict(X)[0]

#     score = 0.9 if int(pred) == 1 else 0.05

#     if hasattr(model, "predict_proba"):
#         try:
#             proba = model.predict_proba(X)[0]
#             if len(proba) > 1:
#                 score = float(proba[1])
#         except Exception:
#             pass
#     elif hasattr(model, "decision_function"):
#         try:
#             decision = model.decision_function(X)[0]
#             score = 1 / (1 + pow(2.71828, -float(decision)))
#         except Exception:
#             pass

#     return {
#         "label": int(pred),
#         "score": float(score),
#     }

# def predict_email_level_with_safetensors(text: str) -> Dict[str, Any]:
#     tokenizer, model = load_hf_model()

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=-1)[0]

#     if probs.shape[0] > 1:
#         suspicious_prob = float(probs[1].item())
#         label = 1 if suspicious_prob >= 0.5 else 0
#     else:
#         suspicious_prob = float(probs[0].item())
#         label = 1 if suspicious_prob >= 0.5 else 0

#     return {
#         "label": int(label),
#         "score": float(suspicious_prob),
#     }


# def token_is_suspicious(tok: str, loaded_bundle=None) -> bool:
#     tok_lower = tok.lower().strip()

#     whitelist_words = set()
#     whitelist_patterns = []

#     if isinstance(loaded_bundle, dict):
#         whitelist_words = set(
#             str(w).lower().strip() for w in loaded_bundle.get("whitelist_words", [])
#         )
#         whitelist_patterns = loaded_bundle.get("whitelist_patterns", [])

#     # normalize token for whitelist check
#     normalized_tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok_lower)

#     # exact whitelist word match
#     if tok_lower in whitelist_words or normalized_tok in whitelist_words:
#         return False

#     # whitelist regex patterns
#     for pattern in whitelist_patterns:
#         try:
#             if re.fullmatch(pattern, tok) or re.fullmatch(pattern, normalized_tok):
#                 return False
#         except re.error:
#             continue

#     # suspicious symbol rules
#     suspicious_symbols = ["@", "#", "$", "%", "^", "&", "*",
#     "(", ")", "-", "_", "+", "=",
#     "{", "}", "[", "]",
#     "|", "\\", ":", ";",
#      "'", "<", ">",  "?", "/",
#     "~", "`"]

#     has_symbol = any(ch in tok for ch in suspicious_symbols)
#     has_digit = any(ch.isdigit() for ch in tok)
#     has_url = "http" in tok_lower or "www." in tok_lower

#     return bool(has_symbol or has_digit or has_url)

# def analyze_body(text: str, model_type: str = "combined") -> Dict[str, Any]:
#     body = text or ""
#     tokens = TOKEN_RE.findall(body)

#     if model_type == "pkl":
#         pkl_res = predict_email_level_with_pkl(body)
#         final_score = pkl_res["score"]
#         final_label = pkl_res["label"]
#         meta = {
#             "model_used": "pkl",
#             "pkl_score": pkl_res["score"],
#             "pkl_label": pkl_res["label"],
#         }

#     elif model_type == "safetensors":
#         hf_res = predict_email_level_with_safetensors(body)
#         final_score = hf_res["score"]
#         final_label = hf_res["label"]
#         meta = {
#             "model_used": "safetensors",
#             "hf_score": hf_res["score"],
#             "hf_label": hf_res["label"],
#         }

#     elif model_type == "combined":
#         pkl_res = predict_email_level_with_pkl(body)
#         hf_res = predict_email_level_with_safetensors(body)

#         final_score = (pkl_res["score"] + hf_res["score"]) / 2.0
#         final_label = 1 if final_score >= 0.5 else 0

#         meta = {
#             "model_used": "combined",
#             "pkl_score": pkl_res["score"],
#             "pkl_label": pkl_res["label"],
#             "hf_score": hf_res["score"],
#             "hf_label": hf_res["label"],
#             "combined_score": final_score,
#         }

#     else:
#         raise ValueError("model_type must be one of: pkl, safetensors, combined")
#     loaded_bundle = load_pkl_model
    
#     labels: List[int] = []
#     scores: List[float] = []
#     obf_tokens: List[str] = []

#     for tok in tokens:
       
#         suspicious = token_is_suspicious(tok, loaded_bundle)

#         if final_label == 1 and suspicious:
#             labels.append(1)
#             scores.append(float(final_score))
#             obf_tokens.append(tok)
#         else:
#             labels.append(0)
#             scores.append(0.05)

#     risk_score = round(float(final_score) * 100)

#     return {
#         "tokens": tokens,
#         "labels": labels,
#         "scores": scores,
#         "obf_tokens": obf_tokens,
#         "risk_score": risk_score,
#         "meta": meta,
#     }











# import os
# import re
# import joblib
# import torch
# import numpy as np
# import scipy.sparse as sp

# from typing import Dict, Any, List
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# TOKEN_RE = re.compile(r"\S+")

# MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# TFIDF_WORD_PATH = os.path.join(MODELS_DIR, "tfidf_word.pkl")
# TFIDF_CHAR_PATH = os.path.join(MODELS_DIR, "tfidf_char.pkl")
# CLASSICAL_MODEL_PATH = os.path.join(MODELS_DIR, "classical_model.pkl")
# BERT_MODEL_DIR = MODELS_DIR

# _tfidf_word = None
# _tfidf_char = None
# _classical_model = None
# _tokenizer = None
# _bert_model = None


# def load_classical_bundle():
#     global _tfidf_word, _tfidf_char, _classical_model

#     if _tfidf_word is None:
#         _tfidf_word = joblib.load(TFIDF_WORD_PATH)

#     if _tfidf_char is None:
#         _tfidf_char = joblib.load(TFIDF_CHAR_PATH)

#     if _classical_model is None:
#         _classical_model = joblib.load(CLASSICAL_MODEL_PATH)

#     return _tfidf_word, _tfidf_char, _classical_model


# def load_bert_bundle():
#     global _tokenizer, _bert_model

#     if _tokenizer is None:
#         _tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)

#     if _bert_model is None:
#         _bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
#         _bert_model.eval()

#     return _tokenizer, _bert_model


# def classical_predict(email_text: str) -> Dict[str, Any]:
#     tfidf_word, tfidf_char, model = load_classical_bundle()

#     X_word = tfidf_word.transform([email_text])
#     X_char = tfidf_char.transform([email_text])

#     X = sp.hstack([X_word, X_char])

#     pred = model.predict(X)[0]

#     score = 0.9 if int(pred) == 1 else 0.05

#     if hasattr(model, "predict_proba"):
#         try:
#             proba = model.predict_proba(X)[0]
#             if len(proba) > 1:
#                 score = float(proba[1])
#         except Exception:
#             pass
#     elif hasattr(model, "decision_function"):
#         try:
#             decision = model.decision_function(X)[0]
#             score = 1 / (1 + np.exp(-float(decision)))
#         except Exception:
#             pass

#     return {
#         "label": int(pred),
#         "score": float(score),
#     }


# def bert_predict(email_text: str) -> Dict[str, Any]:
#     tokenizer, model = load_bert_bundle()

#     inputs = tokenizer(
#         email_text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512,
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=-1)[0]

#     if len(probs) > 1:
#         suspicious_prob = float(probs[1].item())
#         pred = 1 if suspicious_prob >= 0.5 else 0
#     else:
#         suspicious_prob = float(probs[0].item())
#         pred = 1 if suspicious_prob >= 0.5 else 0

#     return {
#         "label": int(pred),
#         "score": float(suspicious_prob),
#     }


# def extract_suspicious_tokens(email_text: str) -> List[str]:
#     tokens = TOKEN_RE.findall(email_text or "")

#     suspicious = []
#     for tok in tokens:
#         token_lower = tok.lower()

#         has_symbol = any(ch in tok for ch in ["@", "$", "_", "%", ">", "<", "/", "\\", "|", ":", ";", "="])
#         has_digit = any(ch.isdigit() for ch in tok)
#         has_url = "http" in token_lower or "www." in token_lower

#         if has_symbol or has_digit or has_url:
#             suspicious.append(tok)

#     return suspicious


# def analyze_body(email_text: str) -> Dict[str, Any]:
#     text = email_text or ""
#     tokens = TOKEN_RE.findall(text)

#     classical_res = classical_predict(text)
#     bert_res = bert_predict(text)

#     # Colab-like final decision
#     final_label = 1 if (classical_res["label"] == 1 or bert_res["label"] == 1) else 0
#     final_score = max(classical_res["score"], bert_res["score"])

#     suspicious_tokens = extract_suspicious_tokens(text)

#     labels: List[int] = []
#     scores: List[float] = []
#     obf_tokens: List[str] = []

#     suspicious_set = set(suspicious_tokens)

#     for tok in tokens:
#         if final_label == 1 and tok in suspicious_set:
#             labels.append(1)
#             scores.append(float(final_score))
#             obf_tokens.append(tok)
#         else:
#             labels.append(0)
#             scores.append(0.05)

#     risk_score = round(float(final_score) * 100)

#     return {
#         "tokens": tokens,
#         "labels": labels,
#         "scores": scores,
#         "obf_tokens": obf_tokens,
#         "risk_score": risk_score,
#         "meta": {
#             "classical_label": classical_res["label"],
#             "classical_score": classical_res["score"],
#             "bert_label": bert_res["label"],
#             "bert_score": bert_res["score"],
#             "final_label": final_label,
#             "final_score": final_score,
#         },
#     }




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