import os
import sys
import joblib
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from backend.preprocess import clean
except ImportError:
    from preprocess import clean

MODEL_PATH = BACKEND_DIR / "model.joblib"
_ARTIFACTS = None

def get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        _ARTIFACTS = joblib.load(MODEL_PATH)
    return _ARTIFACTS

def predict(text: str) -> tuple[str, float]:
    artifacts = get_artifacts()
    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]
    
    cleaned = clean(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]  # [P(REAL), P(FAKE)]
    
    # Class 0: REAL, Class 1: FAKE
    prob_fake = float(probs[1])
    prob_real = float(probs[0])
    
    if prob_fake >= 0.5:
        return "FAKE", round(prob_fake, 4)
    else:
        return "REAL", round(prob_real, 4)
