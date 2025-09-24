import os
import joblib
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import f1_score, average_precision_score

from src.legal_clause_classifier.models.advanced_ann import TfidfANNAdvanced
from config import (
    X_TEST_TFIDF_PATH, Y_TEST_PATH,
    ADVANCED_ANN_MODEL_PATH)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_ann_advanced.pt")

def load_data():
    X_test = sp.load_npz(X_TEST_TFIDF_PATH)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
   
    y_test = torch.from_numpy(np.load(Y_TEST_PATH)).float()

    return X_test, y_test, X_test.shape[1], y_test.shape[1]


def test_advanced_ann_model():
    X_test, y_test, input_dim, num_labels = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = TfidfANNAdvanced(input_dim, num_labels).to(device)
    model.load_state_dict(torch.load(ADVANCED_ANN_MODEL_PATH, map_location=device))
    model.eval()

    # Prediction
    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    y_true = y_test.numpy()
    y_pred = (probs > 0.5).astype(int)

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    pr_auc = average_precision_score(y_true, probs, average="micro")

    print(f"Test Results → Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f} | PR-AUC: {pr_auc:.4f}")
