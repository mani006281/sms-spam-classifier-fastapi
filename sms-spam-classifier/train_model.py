"""
train_model.py
Trains a TF-IDF + Multinomial Naive Bayes SMS spam classifier
on the SMS Spam Collection dataset and saves the trained
vectorizer + model to disk for the FastAPI app to load.
"""

import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------- 1. Load data ----------
df = pd.read_csv("data_raw.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df = df.dropna()
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# ---------- 2. Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)

# ---------- 3. Encode labels ----------
df["target"] = df["label"].map({"ham": 0, "spam": 1})

# ---------- 4. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
)

# ---------- 5. TF-IDF vectorization ----------
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- 6. Train models and compare ----------
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

best_model = None
best_f1 = -1
best_name = None

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
print("\nClassification report for best model:")
print(classification_report(y_test, best_model.predict(X_test_vec), target_names=["ham", "spam"]))

# ---------- 7. Save vectorizer + best model ----------
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(best_model, "model/spam_model.pkl")
with open("model/model_info.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, best_model.predict(X_test_vec)):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, best_model.predict(X_test_vec)):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, best_model.predict(X_test_vec)):.4f}\n")
    f.write(f"F1-score: {f1_score(y_test, best_model.predict(X_test_vec)):.4f}\n")

print("\nSaved model/vectorizer.pkl and model/spam_model.pkl")
