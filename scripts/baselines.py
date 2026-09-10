import pandas as pd
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_FILE = "data/golden/golden_200.jsonl"


def load_data():
    rows = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


df = load_data()

print(f"Total examples: {len(df)}")

print("\nIntent distribution:")
print(df["intent"].value_counts())


# --------------------------------------------------
# 1. MAJORITY CLASS BASELINE
# --------------------------------------------------

majority_class = df["intent"].value_counts().idxmax()

majority_predictions = [majority_class] * len(df)

majority_accuracy = accuracy_score(
    df["intent"],
    majority_predictions
)

print("\n==============================")
print("MAJORITY CLASS BASELINE")
print("==============================")

print(f"Majority intent: {majority_class}")
print(f"Accuracy: {majority_accuracy:.4f}")


# --------------------------------------------------
# 2. TF-IDF + LOGISTIC REGRESSION
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["intent"],
    test_size=0.2,
    random_state=42,
    stratify=df["intent"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==============================")
print("TF-IDF + LOGISTIC REGRESSION")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)