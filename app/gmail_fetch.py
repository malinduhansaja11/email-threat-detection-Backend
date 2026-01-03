import base64
from googleapiclient.discovery import build
from .gmail_oauth import load_credentials

def _header(headers, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""

def _extract_text(payload) -> str:
    # Try text/plain from parts
    parts = payload.get("parts") or []
    for p in parts:
        if p.get("mimeType") == "text/plain":
            data = p.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        # nested parts
        if p.get("parts"):
            nested = _extract_text(p)
            if nested:
                return nested

    # fallback: payload.body.data
    data = payload.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    return ""

def fetch_inbox(max_results: int = 10):
    creds = load_credentials()
    if not creds:
        return None

    service = build("gmail", "v1", credentials=creds)

    res = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX"], maxResults=max_results)
        .execute()
    )

    msgs = res.get("messages", [])
    emails = []

    for m in msgs:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()

        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        sender = _header(headers, "From") or "unknown"
        subject = _header(headers, "Subject") or "(no subject)"
        date = _header(headers, "Date") or ""

        body = _extract_text(payload).strip()

        emails.append({
            "id": msg.get("id"),
            "sender": sender,
            "subject": subject,
            "body": body,
            "received_at": date,
        })

    return emails
