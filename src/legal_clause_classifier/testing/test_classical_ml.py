import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from ..models.classical_ml import load_model
from utils import (
    ARTIFACTS_DIR,
    Y_TEST_PATH,
    X_TEST_TFIDF_PATH
)

MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "logistic_model.pkl")

def load_test_data():
    X_test = sp.load_npz(X_TEST_TFIDF_PATH)
    y_test = np.load(Y_TEST_PATH, allow_pickle=True)
    return X_test, y_test

def evaluate_logistic():
    print("🔹 Loading test data...")
    X_test, y_test = load_test_data()

    print(f"🔹 Loading trained model from {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)

    print("🔹 Predicting...")
    y_pred = model.predict(X_test)

    print("🔹 Evaluation Metrics:")
    print("F1-micro:", f1_score(y_test, y_pred, average="micro"))
    print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("Jaccard Score (samples):", jaccard_score(y_test, y_pred, average="samples"))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_logistic()