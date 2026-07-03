# SMS Spam Classifier — ML + REST API

A Machine Learning project that classifies SMS/text messages as **Spam** or **Ham** (Not Spam) using a **TF-IDF + Multinomial Naive Bayes** model, served through a **FastAPI REST API**.

The model is trained on the **SMS Spam Collection Dataset** containing **5,572 labeled SMS messages** and can be accessed through REST endpoints for seamless integration with web, mobile, or backend applications.

---

# 🚀 Live Demo

### API Base URL

https://sms-spam-classifier-fastapi.onrender.com

### Swagger API Documentation

https://sms-spam-classifier-fastapi.onrender.com/docs

---

# 📖 How It Works

1. Text preprocessing
   - Convert text to lowercase
   - Remove URLs
   - Remove numbers
   - Remove punctuation
   - Remove extra spaces

2. Feature Extraction
   - TF-IDF Vectorization
   - Unigrams + Bigrams
   - Top 3000 Features

3. Model Training
   - Compared Logistic Regression and Multinomial Naive Bayes
   - Selected the best model using the F1-score

4. Model Serving
   - Saved the trained model and vectorizer using **Joblib**
   - Loaded once during FastAPI startup for faster predictions

---

# 📊 Model Performance

Evaluated on a held-out **20% test dataset (1,115 messages)**

| Metric | Score |
|---------|-------|
| Accuracy | **97.04%** |
| Precision (Spam) | **99.15%** |
| Recall (Spam) | **78.52%** |
| F1-Score (Spam) | **87.64%** |

The model achieves **97.04% accuracy** with very high precision, meaning it rarely classifies genuine messages as spam.

---

# 🛠 Tech Stack

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Joblib
- Pydantic

---

# 📌 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health Check |
| POST | `/predict` | Predict a Single SMS |
| POST | `/predict/batch` | Predict Multiple SMS Messages |

---

# 📥 Example Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
"message":"WINNER! You have been selected to receive a cash prize! Call now to claim!"
}'
```

---

# 📤 Example Response

```json
{
  "message": "WINNER! You have been selected to receive a cash prize! Call now to claim!",
  "prediction": "spam",
  "spam_probability": 0.9867,
  "ham_probability": 0.0133
}
```

---

# ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani006281/sms-spam-classifier-fastapi.git
cd sms-spam-classifier-fastapi/sms-spam-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the Model

```bash
python train_model.py
```

### 4. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

### 5. Open API Documentation

Local

```
http://localhost:8000/docs
```

Live Deployment

```
https://sms-spam-classifier-fastapi.onrender.com/docs
```

---

# 📁 Project Structure

```text
sms-spam-classifier
│
├── app/
│   └── main.py
│
├── model/
│   ├── spam_model.pkl
│   ├── vectorizer.pkl
│   └── model_info.txt
│
├── train_model.py
├── data_raw.csv
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 📚 Dataset

**SMS Spam Collection Dataset**

- 5,572 SMS Messages
- 4,825 Ham Messages
- 747 Spam Messages

Source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

# 🚀 Future Improvements

- Develop a modern web frontend for real-time predictions
- Improve accuracy using Deep Learning models (LSTM/BERT)
- Add JWT Authentication
- Add Docker support
- Implement CI/CD with GitHub Actions
- Deploy multiple API versions

---

# 👨‍💻 Author

**Mani Kumar Penugonda**

Backend Developer | Python | FastAPI | Machine Learning

**GitHub**
https://github.com/mani006281

**LinkedIn**
https://www.linkedin.com/in/mani-kumar-penugonda-096705363

**Email**
penugondamanikumar99@gmail.com

---

# ⭐ If you found this project useful, please consider giving it a Star on GitHub.