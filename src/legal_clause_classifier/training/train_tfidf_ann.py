import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score

from ..models.tfidf_ann import ANNClassifier
from config import (
    ARTIFACTS_DIR, 
    X_TRAIN_TFIDF_PATH, X_VAL_TFIDF_PATH,
    Y_TRAIN_PATH, Y_VAL_PATH,
    ANN_MODEL_PATH
)

# Config
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3


def load_data():
    X_train = torch.from_numpy(np.load(X_TRAIN_TFIDF_PATH, allow_pickle=False)["X"].toarray()).float()
    y_train = torch.from_numpy(np.load(Y_TRAIN_PATH)).float()

    X_val = torch.from_numpy(np.load(X_VAL_TFIDF_PATH, allow_pickle=False)["X"].toarray()).float()
    y_val = torch.from_numpy(np.load(Y_VAL_PATH)).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_train.shape[1], y_train.shape[1]

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            labels.append(y_batch.cpu().numpy())
    preds = np.vstack(preds)
    labels = np.vstack(labels)
    micro_f1 = f1_score(labels, preds > 0.5, average="micro")
    macro_f1 = f1_score(labels, preds > 0.5, average="macro")
    pr_auc = average_precision_score(labels, preds, average="micro")
    return micro_f1, macro_f1, pr_auc



def train_tfidf_ann_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, input_dim, output_dim = load_data()

    model = ANNClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # validation
        micro_f1, macro_f1, pr_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f} | PR-AUC: {pr_auc:.4f}")

    torch.save(model.state_dict(), ANN_MODEL_PATH)
    print(f"Model saved at {ANN_MODEL_PATH}")

