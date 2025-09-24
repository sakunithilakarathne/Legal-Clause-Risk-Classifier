import joblib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
import scipy.sparse as sp
from src.legal_clause_classifier.models.advanced_ann import TfidfANNAdvanced, FocalLoss
from config import (
    ADVANCED_ANN_MODEL_PATH,
    X_TRAIN_TFIDF_PATH, Y_TRAIN_PATH,
    X_VAL_TFIDF_PATH, Y_VAL_PATH
)


TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
BEST_VAL_F1 = 0.0
EPOCHS = 15


def load_data():
    X_train = sp.load_npz(X_TRAIN_TFIDF_PATH)
    X_val = sp.load_npz(X_VAL_TFIDF_PATH)

    # Convert to dense torch tensors
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)

    y_train = torch.from_numpy(np.load(Y_TRAIN_PATH)).float()
    y_val = torch.from_numpy(np.load(Y_VAL_PATH)).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=VAL_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_train.shape[1], y_train.shape[1]



def model_evaluate(model, val_loader, device):
# Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(yb.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    micro_f1 = f1_score(all_labels, all_preds > 0.5, average="micro")
    macro_f1 = f1_score(all_labels, all_preds > 0.5, average="macro")
    pr_auc = average_precision_score(all_labels, all_preds, average="micro")

    return micro_f1, macro_f1, pr_auc



def train_advanced_ann_model():
    train_loader, val_loader, input_dim, num_labels = load_data()

    # Model setup
    model = TfidfANNAdvanced(input_dim, num_labels).to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_val_f1 = BEST_VAL_F1
    epochs = EPOCHS

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        micro_f1, macro_f1, pr_auc = model_evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss {total_loss:.4f} | Micro-F1 {micro_f1:.4f} | Macro-F1 {macro_f1:.4f} | PR-AUC {pr_auc:.4f}")

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            torch.save(model.state_dict(), ADVANCED_ANN_MODEL_PATH)
            print(f"✅ Saved new best model (micro_f1={micro_f1:.4f})")









