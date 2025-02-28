from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()

# ✅ Enable CORS to allow requests from Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend (change to specific origin in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("fall_detection_model.pkl")

# Define request structure
class SensorData(BaseModel):
    x: float
    y: float
    z: float

@app.post("/predict")
def predict_fall(data: SensorData):
    # Compute acceleration magnitude
    acc_magnitude = np.sqrt(data.x**2 + data.y**2 + data.z**2)
    
    # Prepare data for model
    features = np.array([[data.x, data.y, data.z, acc_magnitude]])
    
    # Predict fall
    prediction = model.predict(features)
    result = "FALL DETECTED" if prediction[0] == 1 else "No Fall"

    return {"prediction": result}