from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    body: str

class PredictResponse(BaseModel):
    tokens: List[str]
    labels: List[int]   # 0/1
    scores: List[float]
    obf_tokens: List[str]
    risk_score: int