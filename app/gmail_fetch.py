import base64
import re
from typing import Optional

from googleapiclient.discovery import build

from .gmail_oauth import load_credentials


def _decode(data):
    if not data:
        return ""

    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")


def _parse_parts(parts):
    body = ""
    attachments = []

    for part in parts:
        mime = part.get("mimeType", "")
        filename = part.get("filename", "")
        body_data = part.get("body", {}).get("data")

        if mime == "text/plain" and body_data:
            body = _decode(body_data)

        if filename:
            attachments.append(
                {
                    "filename": filename,
                    "mimeType": mime,
                    "attachment_id": part.get("body", {}).get("attachmentId"),
                }
            )

        if part.get("parts"):
            nested_body, nested_attach = _parse_parts(part["parts"])
            body = body or nested_body
            attachments.extend(nested_attach)

    return body, attachments


def fetch_inbox(uid: str, max_results: int = 10):
    creds = load_credentials(uid)

    if not creds:
        return None

    service = build("gmail", "v1", credentials=creds)

    res = (
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=["INBOX"],
            q="category:primary",
            maxResults=max_results,
        )
        .execute()
    )

    emails = []

    for item in res.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=item["id"], format="full")
            .execute()
        )

        payload = msg.get("payload", {})
        raw_headers = payload.get("headers", [])

        headers = {h["name"]: h["value"] for h in raw_headers}

        sender = headers.get("From", "")
        subject = headers.get("Subject", "")
        date = headers.get("Date", "")
        to = headers.get("To", "")
        cc = headers.get("Cc", "")
        bcc = headers.get("Bcc", "")
        reply_to = headers.get("Reply-To", "")
        return_path = headers.get("Return-Path", "")
        message_id = headers.get("Message-ID", "")
        delivered_to = headers.get("Delivered-To", "")
        mime_version = headers.get("MIME-Version", "")
        content_type = headers.get("Content-Type", "")
        auth_results = headers.get("Authentication-Results", "")

        received_headers = [
            h["value"]
            for h in raw_headers
            if h.get("name", "").lower() == "received"
        ]

        domain_match = re.search(r"@([\w.-]+)", sender)
        sender_domain = domain_match.group(1) if domain_match else ""

        auth_lower = auth_results.lower()

        spf = (
            "pass"
            if "spf=pass" in auth_lower
            else "fail"
            if "spf=fail" in auth_lower
            else "unknown"
        )

        dkim = (
            "pass"
            if "dkim=pass" in auth_lower
            else "fail"
            if "dkim=fail" in auth_lower
            else "unknown"
        )

        dmarc = (
            "pass"
            if "dmarc=pass" in auth_lower
            else "fail"
            if "dmarc=fail" in auth_lower
            else "unknown"
        )

        parts = payload.get("parts", [])
        body, attachments = _parse_parts(parts)

        if not body:
            body = _decode(payload.get("body", {}).get("data"))

        emails.append(
            {
                "id": msg["id"],
                "sender": sender,
                "sender_domain": sender_domain,
                "subject": subject,
                "date": date,
                "received_at": date,
                "body": body.strip(),
                "attachments": attachments,
                "spf": spf,
                "dkim": dkim,
                "dmarc": dmarc,
                "to": to,
                "cc": cc,
                "bcc": bcc,
                "reply_to": reply_to,
                "return_path": return_path,
                "message_id": message_id,
                "delivered_to": delivered_to,
                "mime_version": mime_version,
                "content_type": content_type,
                "authentication_results": auth_results,
                "received": received_headers,
                "headers": headers,
            }
        )

    return emails