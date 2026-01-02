from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class EmailItem:
    id: str
    sender: str
    subject: str
    body: str
    received_at: str

class EmailStore:
    def __init__(self):
        self._items: List[EmailItem] = []

    def add(self, sender: str, subject: str, body: str) -> dict:
        item = EmailItem(
            id=str(uuid.uuid4()),
            sender=sender or "unknown@demo",
            subject=subject or "(no subject)",
            body=body or "",
            received_at=datetime.utcnow().isoformat() + "Z",
        )
        self._items.insert(0, item)  # newest first
        return asdict(item)

    def list(self) -> List[dict]:
        return [asdict(x) for x in self._items]

    def get(self, email_id: str) -> Optional[dict]:
        for x in self._items:
            if x.id == email_id:
                return asdict(x)
        return None
