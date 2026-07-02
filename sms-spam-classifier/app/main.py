"""
main.py
FastAPI application that serves the trained SMS spam classifier.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    GET  /                -> health check
    POST /predict          -> classify a single message
    POST /predict/batch    -> classify multiple messages at once
"""

import re
import string
import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

app = FastAPI(
    title="SMS Spam Classifier API",
    description="A REST API that classifies SMS/text messages as Spam or Ham (not spam) using a TF-IDF + Naive Bayes model.",
    version="1.0.0",
)

# ---------- Load model artifacts once at startup ----------
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


# ---------- Text cleaning (must match training preprocessing) ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Request / Response schemas ----------
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, example="Congratulations! You've won a free prize, call now!")


class BatchMessageRequest(BaseModel):
    messages: List[str] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    message: str
    prediction: str
    spam_probability: float
    ham_probability: float


# ---------- Routes ----------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "SMS Spam Classifier API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    cleaned = clean_text(request.message)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]   # [ham_prob, spam_prob]
    pred = model.predict(vec)[0]

    return PredictionResponse(
        message=request.message,
        prediction="spam" if pred == 1 else "ham",
        spam_probability=round(float(proba[1]), 4),
        ham_probability=round(float(proba[0]), 4),
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchMessageRequest):
    results = []
    for msg in request.messages:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]
        results.append(
            PredictionResponse(
                message=msg,
                prediction="spam" if pred == 1 else "ham",
                spam_probability=round(float(proba[1]), 4),
                ham_probability=round(float(proba[0]), 4),
            )
        )
    return results
