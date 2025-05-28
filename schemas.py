from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_confidences: Dict[str, float]