from pydantic import BaseModel
from typing import Dict, List, Union  # Added missing imports

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: str
    properties: List[str]  # Now List is properly imported
    color: str
    all_confidences: Dict[str, Dict[str, Union[float, str]]]