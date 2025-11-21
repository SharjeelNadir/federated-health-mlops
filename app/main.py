import os
from typing import Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fl.model import HealthRiskNet  # reuse the same model architecture

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "global_model.pth")

app = FastAPI(
    title="Federated Health Risk API",
    description="API for predicting health risk using the federated global model",
    version="1.0.0",
)

model: nn.Module | None = None
device = torch.device("cpu")


# --------------------------------------------------------
# Input schema (for citizens/wearables)
# This matches Node 1 features: 8 inputs
# --------------------------------------------------------
class WearableFeatures(BaseModel):
    heart_rate: float
    spo2: float
    steps: float
    sleep_hours: float
    age: float
    smoker: int     # 0 or 1
    chronic: int    # 0 or 1
    aqi: float


# --------------------------------------------------------
# Startup: load the trained global model
# --------------------------------------------------------
@app.on_event("startup")
def load_model() -> None:
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}. "
              f"Train FL and save global_model.pth first.")
        # We still create an untrained model so the app doesn't crash
        model = HealthRiskNet(input_dim=8).to(device)
        return

    print(f"[INFO] Loading global model from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = HealthRiskNet(input_dim=8).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Model loaded and ready for inference.")


# --------------------------------------------------------
# Health check
# --------------------------------------------------------
@app.get("/")
def read_root() -> Dict[str, str]:
    return {
        "message": "Federated Health Risk API is running.",
        "model_loaded": str(model is not None and os.path.exists(MODEL_PATH)),
    }


# --------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------
@app.post("/predict", summary="Predict health risk from wearable + context data")
def predict_risk(features: WearableFeatures) -> Dict[str, float | bool]:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Make sure global_model.pth exists and restart the API.",
        )

    # Convert input to model tensor [1, 8]
    x = torch.tensor(
        [[
            features.heart_rate,
            features.spo2,
            features.steps,
            features.sleep_hours,
            features.age,
            float(features.smoker),
            float(features.chronic),
            features.aqi,
        ]],
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        prob = model(x).item()

    # Simple threshold (can be tuned)
    high_risk = prob >= 0.5

    return {
        "risk_score": round(prob, 4),
        "high_risk": high_risk,
    }
