from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PredictRequest(BaseModel):
    body: str


class PredictResponse(BaseModel):
    tokens: List[str]
    labels: List[int]
    scores: List[float]
    obf_tokens: List[str]
    risk_score: int
    meta: Optional[Dict[str, Any]] = None

    