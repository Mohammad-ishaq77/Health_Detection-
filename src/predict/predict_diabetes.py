import joblib
import pandas as pd
import os
from src.config import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")

# Global cache for model
_model_cache = {}

def _load_model():
    """Lazy load model with caching to avoid multiprocessing issues"""
    global _model_cache
    
    if "model" in _model_cache:
        return _model_cache["model"]
    
    model = joblib.load(MODEL_PATH)
    _model_cache["model"] = model
    return model


def predict_diabetes(data: dict):
    model = _load_model()
    
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        risk = "Diabetic"
    else:
        risk = "Non-Diabetic"

    return {
        "result": risk,
        "probability": float(probability)
    }