import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score
import pickle
from src.legal_clause_classifier.models.lstm import LSTMClassifier
from config import (
    Y_TEST_PATH, LSTM_TOKENIZED_TEST, LSTM_MODEL_PATH, LSTM_VOCAB_PATH)


def load_dataset(tokenized_path, y_path):
    X = np.load(tokenized_path)
    y = np.load(y_path)
    return TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32))


def collate_fn(batch):
    X, y = zip(*batch)
    return torch.stack(X), torch.stack(y)


def evaluate_epoch(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    micro_f1 = f1_score(labels, preds > 0.5, average="micro")
    macro_f1 = f1_score(labels, preds > 0.5, average="macro")
    pr_auc = average_precision_score(labels, preds, average="micro")

    return micro_f1, macro_f1, pr_auc


def test_lstm_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LSTM_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    test_ds = load_dataset(LSTM_TOKENIZED_TEST, Y_TEST_PATH)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(vocab)
    num_labels = np.load(Y_TEST_PATH).shape[1]

    model = LSTMClassifier(vocab_size, embed_dim=128, hidden_dim=256, num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    micro_f1 = f1_score(labels, preds > 0.5, average="micro")
    macro_f1 = f1_score(labels, preds > 0.5, average="macro")
    pr_auc = average_precision_score(labels, preds, average="micro")

    print(f"Test Results | Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f} | PR-AUC: {pr_auc:.4f}")

