import os
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score

from src.legal_clause_classifier.models.tfidf_ann import ANNClassifier
from config import (
    ARTIFACTS_DIR, 
    X_TEST_TFIDF_PATH, Y_TEST_PATH,
    ANN_MODEL_PATH
)

def evaluate_tfidf_ann_model():
    X_test = sp.load_npz(X_TEST_TFIDF_PATH)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    
    y_test = torch.from_numpy(np.load(Y_TEST_PATH)).float()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]

    model = ANNClassifier(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    model.load_state_dict(torch.load(ANN_MODEL_PATH, map_location="cpu"))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds.append(outputs.numpy())
            labels.append(y_batch.numpy())

    preds = np.vstack(preds)
    labels = np.vstack(labels)

    micro_f1 = f1_score(labels, preds > 0.5, average="micro")
    macro_f1 = f1_score(labels, preds > 0.5, average="macro")
    pr_auc = average_precision_score(labels, preds, average="micro")

    print(f"Test Results → Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f} | PR-AUC: {pr_auc:.4f}")
