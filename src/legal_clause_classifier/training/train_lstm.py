import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from src.legal_clause_classifier.models.lstm import LSTMClassifier
import wandb
from config import (
    Y_TRAIN_PATH, Y_VAL_PATH,
    LSTM_TOKENIZED_TRAIN, LSTM_TOKENIZED_VAL,
    LSTM_VOCAB_PATH, LSTM_MODEL_PATH
    )


def load_dataset(tokenized_path, y_path):
    X = np.load(tokenized_path)
    y = np.load(y_path)
    return TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32))


def collate_fn(batch):
    X, y = zip(*batch)
    return torch.stack(X), torch.stack(y)


def train_epoch(model, train_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/ len(train_loader)



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


# --- Training ---
def train_lstm_model():

    # Initialize wandb
    wandb.init(
        project="legal-clause-classifier",  
        name="lstm_v1",  
        config={
            "batch_size": 32,
            "epochs": 15,
            "learning_rate": 1e-3,
            "embed_dim": 128,
            "hidden_dim": 256,
            "model": "LSTMClassifier"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LSTM_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    train_ds = load_dataset(LSTM_TOKENIZED_TRAIN, Y_TRAIN_PATH)
    val_ds = load_dataset(LSTM_TOKENIZED_VAL, Y_VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True ,collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(vocab)
    num_labels = np.load(Y_TRAIN_PATH).shape[1]

    model = LSTMClassifier(vocab_size, embed_dim=128, hidden_dim=256, num_labels=num_labels).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)#optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train_epoch(model, train_loader, device, optimizer, criterion)
        micro, macro, prauc = evaluate_epoch(model, val_loader, device)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss,
            "f1_micro": micro,
            "f1_macro": macro,
            "pr_auc": prauc
        })

        print(f"Epoch {epoch+1} | Loss: {loss} "
              f"| Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | PR-AUC: {prauc:.4f}")

    torch.save(model.state_dict(), LSTM_MODEL_PATH)

    wandb.finish()

