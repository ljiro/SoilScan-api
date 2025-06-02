from pydantic import BaseModel
from typing import Dict, List, Union  # Added missing imports

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: str
    properties: List[str]  # Now List is properly imported
    color: str
    all_confidences: Dict[str, Dict[str, Union[float, str]]]
        
        

class FertilizerRequest(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float

    class Config:
        schema_extra = {
            "example": {
                "Temperature": 25,
                "Humidity": 60,
                "Moisture": 25,
                "Soil_Type": "Loamy",
                "Crop_Type": "Maize",
                "Nitrogen": 20,
                "Potassium": 30,
                "Phosphorous": 15
            }
        }