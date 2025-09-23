import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from ..models.classical_ml import load_model
from config import (
    ARTIFACTS_DIR,
    Y_TEST_PATH,
    X_TEST_TFIDF_PATH,
    LR_MODEL_PATH
)


def load_test_data():
    X_test = sp.load_npz(X_TEST_TFIDF_PATH)
    y_test = np.load(Y_TEST_PATH, allow_pickle=True)
    return X_test, y_test

def evaluate_logistic():
    print("🔹 Loading test data...")
    X_test, y_test = load_test_data()

    print(f"🔹 Loading trained model from {LR_MODEL_PATH}")
    model = load_model(LR_MODEL_PATH)

    print("🔹 Predicting...")
    y_pred = model.predict(X_test)

    print("🔹 Evaluation Metrics:")
    print("F1-micro:", f1_score(y_test, y_pred, average="micro"))
    print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Jaccard Score (samples):", jaccard_score(y_test, y_pred, average="samples"))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
