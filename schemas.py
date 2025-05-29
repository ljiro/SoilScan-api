from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: str
    properties: List[str]
    color: str
    all_confidences: Dict[str, Dict[str, Union[float, str]]]