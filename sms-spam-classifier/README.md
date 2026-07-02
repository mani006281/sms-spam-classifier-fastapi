# SMS Spam Classifier — ML + REST API

A machine learning project that classifies SMS/text messages as **Spam** or **Ham** (not spam), trained on the public SMS Spam Collection dataset (5,572 labeled messages) and served through a **FastAPI** REST API.

This project combines a classic NLP/ML pipeline with a backend API layer, so the model can be consumed by any client (web app, mobile app, or another service) via HTTP.

## How it works

1. **Text preprocessing** — lowercasing, URL/number removal, punctuation stripping
2. **Feature extraction** — TF-IDF vectorization (unigrams + bigrams, top 3000 features)
3. **Model training** — compared Multinomial Naive Bayes vs Logistic Regression, selected the best based on F1-score
4. **Serving** — the trained model and vectorizer are saved with `joblib` and loaded once at API startup for fast inference

## Model Performance

Evaluated on a held-out 20% test split (1,115 messages):

| Metric | Score |
|---|---|
| Accuracy | 97.04% |
| Precision (spam) | 99.15% |
| Recall (spam) | 78.52% |
| F1-score (spam) | 87.64% |

High precision means the model rarely flags a real message as spam (few false positives) — important for a spam filter, since blocking a genuine message is worse than letting an occasional spam message through.

## Tech Stack

- **ML:** scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression)
- **Data handling:** pandas
- **API:** FastAPI + Uvicorn
- **Model persistence:** joblib
- **Validation:** Pydantic

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Classify a single message |
| POST | `/predict/batch` | Classify multiple messages at once |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! You have been selected to receive a cash prize! Call now to claim!"}'
```

### Example response

```json
{
  "message": "WINNER! You have been selected to receive a cash prize! Call now to claim!",
  "prediction": "spam",
  "spam_probability": 0.9867,
  "ham_probability": 0.0133
}
```

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model from scratch
python train_model.py

# 3. Start the API server
uvicorn app.main:app --reload

# 4. Open interactive API docs
# http://localhost:8000/docs
```

## Project Structure

```
├── app/
│   └── main.py          # FastAPI application and endpoints
├── model/
│   ├── spam_model.pkl    # Trained classifier
│   ├── vectorizer.pkl    # Fitted TF-IDF vectorizer
│   └── model_info.txt    # Saved evaluation metrics
├── train_model.py        # Data loading, preprocessing, training, evaluation
├── data_raw.csv           # SMS Spam Collection dataset
├── requirements.txt
└── README.md
```

## Dataset

[SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 SMS messages (4,825 ham, 747 spam), UCI Machine Learning Repository / Kaggle.

## Future Improvements

- Add a simple frontend to test predictions interactively
- Experiment with deep learning (LSTM/BERT) for higher recall
- Add authentication for the API
- Deploy to a cloud platform (Render/Railway) for a live demo link
