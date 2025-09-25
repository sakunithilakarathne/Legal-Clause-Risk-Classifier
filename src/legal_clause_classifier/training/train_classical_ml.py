import os
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from ..models.classical_ml import *
from ..utils.logger import get_logger
import wandb
from config import(
    ARTIFACTS_DIR,
    Y_TRAIN_PATH, Y_VAL_PATH,
    X_TRAIN_TFIDF_PATH, X_VAL_TFIDF_PATH,
    LR_MODEL_PATH)


# Configure logging
logger = get_logger("training")


def load_data():

    X_train = sp.load_npz(X_TRAIN_TFIDF_PATH)
    X_val = sp.load_npz(X_VAL_TFIDF_PATH)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True)
    y_val = np.load(Y_VAL_PATH, allow_pickle=True)

    return X_train, X_val, y_train, y_val


def train_logistic_regression_model():

    wandb.init(
        project="legal-clause-classifier",  
        name="logistic_regression_v1",  
        config={
            "model": "LogisticRegression",
            "solver": "saga",  
            "max_iter": 100,  
            "batch_size": None,  
            "learning_rate": None,  
        }
    )

    logger.info("Loading data...")
    X_train, X_val, y_train, y_val = load_data()

    logger.info("Building Logistic Regression model...")
    model =logistic_regression_model()

    logger.info("Training...")
    model.fit(X_train, y_train)

    logger.info("Evaluating on validation set...")

    y_val_pred = model.predict(X_val)
    f1_micro = f1_score(y_val, y_val_pred, average="micro")
    f1_macro = f1_score(y_val, y_val_pred, average="macro")
    logger.info(f"Validation F1-micro: {f1_micro:.4f}, F1-macro: {f1_macro:.4f}")
    print(f"Validation F1-micro: {f1_micro:.4f}, F1-macro: {f1_macro:.4f}")

    # Log metrics to wandb
    wandb.log({
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    })

    logger.info(f"Saving model to {LR_MODEL_PATH}")
    save_model(model, LR_MODEL_PATH)

    wandb.finish()

