"""
train_model.py
----------------
Trains the two AI models used by SupportPilot Milestone 1:

  1. Category classifier   : ticket text -> Department (10 classes)
  2. Severity classifier   : ticket text -> Priority (low / medium / high)

Both use TF-IDF vectorization + Logistic Regression, as specified in the
Milestone 1 deck. Trained on the real IT_Support_Ticket_Data.csv dataset
(~29,650 real support tickets) instead of the toy 10-row example in the
slides, so the reported accuracy is genuine.

Run:
    python train_model.py
"""

import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from classifier import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "IT_Support_Ticket_Data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Body", "Department", "Priority"]).copy()
    df["clean_text"] = df["Body"].apply(preprocess_text)
    # drop empty rows after cleaning
    df = df[df["clean_text"].str.len() > 0]
    return df


def train_category_model(df):
    print("\n=== Training Category (Department) Classifier ===")
    X = df["clean_text"]
    y = df["Department"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), min_df=2,
        stop_words="english", sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Category classification accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "category_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "category_vectorizer.pkl"))
    # also save under the name referenced in the Milestone 1 slides
    joblib.dump(model, os.path.join(MODEL_DIR, "ticket_classifier.pkl"))

    return accuracy


def train_severity_model(df):
    print("\n=== Training Severity (Priority) Classifier ===")
    X = df["clean_text"]
    y = df["Priority"]  # low / medium / high

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), min_df=2,
        stop_words="english", sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Severity prediction accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "severity_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "severity_vectorizer.pkl"))

    return accuracy


def main():
    df = load_data()
    print(f"Loaded {len(df)} cleaned tickets from real dataset.")

    cat_acc = train_category_model(df)
    sev_acc = train_severity_model(df)

    report = {
        "dataset_size": len(df),
        "classification_accuracy": round(cat_acc * 100, 2),
        "classification_target": 90.0,
        "classification_status": "PASS" if cat_acc >= 0.90 else "BELOW TARGET",
        "severity_accuracy": round(sev_acc * 100, 2),
        "severity_target": 85.0,
        "severity_status": "PASS" if sev_acc >= 0.85 else "BELOW TARGET",
    }

    with open(os.path.join(BASE_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Milestone 1 Evaluation Summary ===")
    print(json.dumps(report, indent=2))
    print("\nModels saved to:", MODEL_DIR)
    print("Evaluation report saved to: evaluation_report.json")


if __name__ == "__main__":
    main()
