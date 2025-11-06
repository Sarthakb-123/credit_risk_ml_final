from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Credit Risk Prediction API", version="2.0")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

# --- Load Model Components ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print(f"✅ Models and feature list loaded successfully! ({len(model_features)} features)")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model, scaler, model_features = None, None, []

# --- Input Schema ---
class CreditData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float
    CODE_GENDER: str
    NAME_FAMILY_STATUS: str
    NAME_EDUCATION_TYPE: str

# --- API Root ---
@app.get("/")
def root():
    return {"message": "Welcome to Credit Risk Prediction API 🚀"}

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: CreditData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check model files.")

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # One-hot encode categorical columns (as done in training)
    input_df = pd.get_dummies(input_df)

    # Add missing columns (set to 0)
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[model_features]

    # Scale numerical data (if scaler used)
    try:
        input_scaled = scaler.transform(input_df)
    except Exception:
        input_scaled = input_df  # fallback if scaler not used

    # Predict
    pred_proba = model.predict_proba(input_scaled)[0][1]
    prediction = "High Risk" if pred_proba > 0.5 else "Low Risk"

    return {
        "prediction": prediction,
        "probability": round(float(pred_proba), 4)
    }
