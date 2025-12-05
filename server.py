# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from feature_engineer import FeatureEngineer


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
    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])
    else:
        proba = None

    return {"prediction": int(pred), "probability": proba}
