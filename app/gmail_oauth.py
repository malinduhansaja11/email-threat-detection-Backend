import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

# Prevent OAuthLib from crashing if Google returns extra default scopes.
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

BASE_DIR = Path(__file__).resolve().parent
TOKENS_DIR = BASE_DIR / "user_tokens"
TOKENS_DIR.mkdir(exist_ok=True)

STATE_SECRET = os.getenv("OAUTH_STATE_SECRET", "dev-change-this-secret")


def _client_config():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            "Missing GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI in .env"
        )

    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def _safe_uid(uid: str) -> str:
    return "".join(c for c in uid if c.isalnum() or c in ("_", "-"))


def _token_file(uid: str) -> Path:
    return TOKENS_DIR / f"{_safe_uid(uid)}.json"


def make_oauth_state(uid: str) -> str:
    payload = {
        "uid": uid,
        "ts": int(time.time()),
    }

    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("utf-8")

    signature = hmac.new(
        STATE_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return f"{payload_b64}.{signature}"


def read_oauth_state(state: str) -> str:
    try:
        payload_b64, signature = state.rsplit(".", 1)
    except Exception:
        raise RuntimeError("Invalid OAuth state")

    expected_signature = hmac.new(
        STATE_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        raise RuntimeError("Invalid OAuth state signature")

    try:
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode(
            "utf-8"
        )
        payload = json.loads(payload_json)
    except Exception:
        raise RuntimeError("Invalid OAuth state payload")

    uid = payload.get("uid")
    ts = int(payload.get("ts", 0))

    if not uid:
        raise RuntimeError("OAuth state missing uid")

    if int(time.time()) - ts > 10 * 60:
        raise RuntimeError("OAuth state expired")

    return uid


def get_auth_url(state: str) -> str:
    print("USING GMAIL OAUTH SCOPES:", SCOPES)

    flow = Flow.from_client_config(
        _client_config(),
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI"),
    )

    auth_url, _state = flow.authorization_url(
        access_type="offline",
        prompt="consent select_account",
        state=state,
    )

    return auth_url


def exchange_code_for_token(code: str, uid: str) -> Credentials:
    print("EXCHANGING TOKEN FOR UID:", uid)
    print("REQUESTED SCOPES:", SCOPES)

    flow = Flow.from_client_config(
        _client_config(),
        scopes=SCOPES,
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI"),
    )

    flow.fetch_token(code=code)

    creds = flow.credentials

    granted_scopes = set(creds.scopes or [])
    print("GRANTED SCOPES:", granted_scopes)

    if "https://www.googleapis.com/auth/gmail.readonly" not in granted_scopes:
        raise RuntimeError(
            "Google did not grant gmail.readonly scope. "
            "Check Google Cloud Data access scopes, Gmail API enabled status, "
            "test users, and reconnect Gmail after deleting old token files."
        )

    _token_file(uid).write_text(creds.to_json(), encoding="utf-8")

    return creds


def load_credentials(uid: str) -> Optional[Credentials]:
    token_file = _token_file(uid)

    if not token_file.exists():
        return None

    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return creds


def clear_token(uid: str):
    token_file = _token_file(uid)

    if token_file.exists():
        token_file.unlink()