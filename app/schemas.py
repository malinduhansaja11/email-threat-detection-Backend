from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    body: str


# OLD response (mock rule-based)
class PredictResponse(BaseModel):
    tokens: List[str]
    labels: List[int]   # 0/1
    scores: List[float]
    obf_tokens: List[str]
    risk_score: int


# NEW response (ML-based)
class AnalyzeResponse(BaseModel):
    tokens: List[str]
    labels: List[str]   # "email" / "url" / "obfuscation_word" / "normal_word"
    scores: List[float]
    emails: List[str]
    urls: List[str]
    obf_tokens: List[str]
    risk_score: int
