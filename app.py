import json
import os
import tempfile 
import pandas as pd
import sklearn
import torch
from pydantic import BaseModel
import joblib  # for RandomForest model loading
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback
import io
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from models import load_soil_model, transform
from schemas import PredictionResponse
from models import SoilTextureModel
from tensorflow.keras.models import load_model  # TensorFlow's load_model
from typing import Dict  # Add this import
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Initialize models at startup
soil_model = load_soil_model()


# For patching if needed
try:
    import patchify
except ImportError:
    patchify = None

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
        
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

    

    
# Add this dictionary at the top of your file with other constants
SOIL_TEXTURE_INFO = {
    'Alluvial': {
        'description': 'Fertile soil deposited by rivers, rich in minerals',
        'properties': ['High fertility', 'Good water retention', 'Suitable for most crops'],
        'color': '#E1C16E'  # A representative color for alluvial soil
    },
    'Black': {
        'description': 'Volcanic soil with high clay content, very fertile',
        'properties': ['High moisture retention', 'Rich in calcium and magnesium', 'Cracks in dry weather'],
        'color': '#3D3D3D'
    },
    'Cinder': {
        'description': 'Porous volcanic soil with good drainage',
        'properties': ['Low water retention', 'Good aeration', 'Suitable for drought-resistant plants'],
        'color': '#6F6F6F'
    },
    'Clay': {
        'description': 'Fine-grained soil with poor drainage',
        'properties': ['High nutrient content', 'Slow drainage', 'Hard when dry, sticky when wet'],
        'color': '#B66A50'
    },
    'Laterite': {
        'description': 'Reddish soil rich in iron and aluminum',
        'properties': ['Low fertility', 'Good for bricks', 'Common in tropical areas'],
        'color': '#E97451'
    },
    'Loamy': {
        'description': 'Balanced mixture of sand, silt, and clay',
        'properties': ['Ideal for most plants', 'Good drainage and moisture retention', 'Easy to work with'],
        'color': '#C19A6B'
    },
    'Peat': {
        'description': 'Organic soil with high water content',
        'properties': ['High acidity', 'Good water retention', 'Requires drainage for cultivation'],
        'color': '#5F4B32'
    },
    'Red': {
        'description': 'Soil rich in iron oxides',
        'properties': ['Good drainage', 'Low fertility', 'Common in warm climates'],
        'color': '#E35335'
    },
    'Sandy': {
        'description': 'Coarse soil with large particles',
        'properties': ['Fast drainage', 'Low nutrient retention', 'Easy to work with'],
        'color': '#FAD5A5'
    },
    'Yellow': {
        'description': 'Soil with iron oxide hydration',
        'properties': ['Moderate fertility', 'Good drainage', 'Common in humid areas'],
        'color': '#FFD700'
    }
}



# Load your Random Forest model
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the trained model and preprocessing objects
try:
    with open('ExtraTreesClassifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")
    
    
    # Soil and crop types from the training data
soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 
              'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']


ct = joblib.load('column_transformer_v1.pkl')
sc = joblib.load('standard_scaler_v1.pkl')



async def temp(file: UploadFile):
    temp_path = None  # Initialize temp_path outside the with block
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            contents = await file.read()  # Now 'await' is valid here
            tmp_file.write(contents)  # Use await for async write
            temp_path = tmp_file.name  # Get the path to the temporary file

        print(f"DEBUG: Saved uploaded file to temporary location: {temp_path}")

        # Call your predict method with the path to the temporary file
        result = classifier.predict_on_image_resnet50(
            temp_path,
            config_for_resnet50_prediction,
            class_names_for_resnet50_prediction
        )

        # Clean up the temporary file
        os.remove(temp_path)
        print(f"DEBUG: Removed temporary file: {temp_path}")

        # Return the results
        return result
    except Exception as e:
        import traceback
        if temp_path:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"DEBUG: Cleaned up temporary file due to error: {temp_path}")
        return {"error": str(e), "traceback": traceback.format_exc()}
      
      
      
config_for_resnet50_prediction = {
    'input_shape': (150, 150, 3),
    'patch_size': 100,
    'patch_step': 50,
    'use_patching_for_prediction': True,
}


class_names_npy_path = "class_info.npy"


class_names_for_resnet50_prediction = []
num_classes_from_file_load = 0


loaded_data = np.load(class_names_npy_path, allow_pickle=True)
print(f"DEBUG: 0-D array: {loaded_data}")


# --- MODIFIED LOGIC TO HANDLE 0-D ARRAY CONTAINING A DICTIONARY ---
if loaded_data.ndim == 0 and isinstance(loaded_data.item(), dict):
    print(f"✓ Loaded a 0-D array containing a dictionary from '{class_names_npy_path}'.")
    data_dict = loaded_data.item()
    if 'class_names' in data_dict and isinstance(data_dict['class_names'], list):
        class_names_for_resnet50_prediction = [str(name) for name in data_dict['class_names']]
        print(f"  Extracted 'class_names' list with {len(class_names_for_resnet50_prediction)} items.")
    else:
        print(f"  ⚠️ Dictionary in .npy file does not contain a 'class_names' key with a list value.")
elif loaded_data.ndim == 1:
    print(f"✓ Loaded a 1-D array from '{class_names_npy_path}'.")
    class_names_for_resnet50_prediction = list(map(str, loaded_data))
else:
    print(f"⚠️ Loaded an array with unexpected dimensions (shape: {loaded_data.shape}) from '{class_names_npy_path}'. Expected 0-D with dict or 1-D array.")
# --- END OF MODIFIED LOGIC ---


if not class_names_for_resnet50_prediction:
    print(f"⚠️ Class names list is empty after processing '{class_names_npy_path}'.")
else:
    num_classes_from_file_load = len(class_names_for_resnet50_prediction)
    print(f"✓ Successfully processed {num_classes_from_file_load} class names.")

    

# ✅ Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "soil color classifier ready"}

crop_df = pd.read_csv("Crop_recommendation.csv")
CROP_CLASSES = dict(enumerate(pd.Categorical(crop_df["label"]).categories))

@app.post("/predict-crop")
async def predict_crop(data: CropInput):
    try:
        # Convert input to array
        features = [[
            data.N, data.P, data.K,
            data.temperature, data.humidity,
            data.ph, data.rainfall
        ]]

        # Scale input
        scaled = scaler.transform(features)

        # Predict probabilities for all classes
        proba = xgb_model.predict_proba(scaled)[0]

        # Get top-5 predictions
        top_indices = proba.argsort()[-5:][::-1]
        predictions = [{
            "crop_name": CROP_CLASSES.get(idx, "Unknown"),
            "score": float(proba[idx])
        } for idx in top_indices]

        return { "predictions": predictions }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
      
        
#Comment
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: str
    properties: list
    color: str
    all_confidences: dict
        

@app.post("/predict_texture", response_model=PredictionResponse)
async def predict_texture(file: UploadFile = File(...)):
    """Classify soil texture from an image"""
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(soil_model.device)
        
        # Predict
        with torch.no_grad():
            outputs = soil_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Prepare response
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        class_name = str(soil_model.class_names[class_idx]).strip()
        
        texture_info = SOIL_TEXTURE_INFO.get(class_name, {
            'description': 'No description available',
            'properties': [],
            'color': '#FFFFFF'
        })
        
        all_confidences = {
            str(name).strip(): {
                "score": float(probs[i]),
                "color": str(SOIL_TEXTURE_INFO.get(str(name).strip(), {}).get('color', '#FFFFFF'))
            } for i, name in enumerate(soil_model.class_names)
        }
        
        return {
            "predicted_class": class_name,
            "confidence": confidence,
            "description": texture_info['description'],
            "properties": texture_info['properties'],
            "color": texture_info['color'],
            "all_confidences": all_confidences
        }
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
        


@app.post("/predict_fertilizer")
async def predict_fertilizer(request: FertilizerRequest):
    try:
        # Validate soil and crop types
        if request.Soil_Type not in soil_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Soil Type. Must be one of: {', '.join(soil_types)}"
            )
        
        if request.Crop_Type not in crop_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Crop Type. Must be one of: {', '.join(crop_types)}"
            )

        # Create a DataFrame from the input
        input_data = pd.DataFrame([{
            'Temparature': request.Temperature,
            'Humidity': request.Humidity,  # Fixed: Removed extra space
            'Moisture': request.Moisture,
            'Soil Type': request.Soil_Type,
            'Crop Type': request.Crop_Type,
            'Nitrogen': request.Nitrogen,
            'Potassium': request.Potassium,
            'Phosphorous': request.Phosphorous
        }])

        # Apply the same preprocessing as during training
        X_encoded = ct.transform(input_data)
        X_scaled = sc.transform(X_encoded)

        # Make prediction
        prediction = model.predict(X_scaled)
        
        return {"recommended_fertilizer": prediction[0]}
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

        


if __name__ == "__main__":
  print("TEST")
