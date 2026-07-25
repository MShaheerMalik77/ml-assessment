from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import clean_text

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"

app = FastAPI(
    title="Customer Query Classification API",
    description="Classifies incoming text into a predicted category.",
    version="1.0.0",
)

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")


class QueryResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Input text is empty after preprocessing.")

    prediction = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = float(max(probabilities))

    return QueryResponse(category=prediction, confidence=confidence)