# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sys

# Import your FeatureEngineer so joblib can find it
from feature_engineer import FeatureEngineer

# ⭐ Register FeatureEngineer under __main__ for joblib unpickling
sys.modules["__main__"].FeatureEngineer = FeatureEngineer

MODEL_PATH = "best_model.pkl"

print("🔄 Loading model once at startup...")

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully!")

app = FastAPI()

class PredictRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.data])

    pred = int(model.predict(df)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])

    return {"prediction": pred, "probability": proba}
