# from pydantic import BaseModel
# from typing import List
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any

# class PredictRequest(BaseModel):
#     body: str
#     model_type: Optional[str] = "pkl"   # pkl | safetensors | combined

# # OLD response (mock rule-based)
# class PredictResponse(BaseModel):
#     tokens: List[str]
#     labels: List[int]   # 0/1
#     scores: List[float]
#     obf_tokens: List[str]
#     risk_score: int
#     meta: Optional[Dict[str, Any]] = None

# # NEW response (ML-based)
# class AnalyzeResponse(BaseModel):
#     tokens: List[str]
#     labels: List[str]   # "email" / "url" / "obfuscation_word" / "normal_word"
#     scores: List[float]
#     emails: List[str]
#     urls: List[str]
#     obf_tokens: List[str]
#     risk_score: int

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

    