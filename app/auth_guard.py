import os
from pathlib import Path
from typing import Optional

import firebase_admin
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status
from firebase_admin import auth, credentials, firestore

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

if not firebase_admin._apps:
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")

    if service_account_path:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

db = firestore.client()


def require_firebase_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Firebase Authorization token",
        )

    token = authorization.split(" ", 1)[1]

    try:
        decoded = auth.verify_id_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    uid = decoded.get("uid")

    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase user",
        )

    user_doc = db.collection("users").document(uid).get()

    if not user_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User profile not found",
        )

    profile = user_doc.to_dict() or {}

    decoded["profile"] = profile
    return decoded


def require_approved_user(user=Header(default=None)):
    # This wrapper is intentionally not used directly.
    # FastAPI dependency version is below.
    return user


def require_approved_firebase_user(
    authorization: Optional[str] = Header(default=None),
):
    user = require_firebase_user(authorization)
    profile = user.get("profile") or {}

    if profile.get("status") != "approved":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not approved by admin",
        )

    return user


def require_admin_firebase_user(
    authorization: Optional[str] = Header(default=None),
):
    user = require_approved_firebase_user(authorization)
    profile = user.get("profile") or {}

    if profile.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return user