import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import f1_score, average_precision_score, accuracy_score

from src.legal_clause_classifier.models.legal_bert import get_legalbert_model
from src.legal_clause_classifier.optimization.class_imbalance_handling import *

from config import (
    ARTIFACTS_DIR, Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH,
    TOKENIZED_TRAIN, TOKENIZED_VAL, TOKENIZED_TEST,
    LABEL_LIST_PATH, BEST_PARAMS_PATH,
    HP_TUNED_MODEL_PATH
)
import json
from datasets import Value
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_from_disk, Sequence, Value



def load_data():
    train_ds = load_from_disk(TOKENIZED_TRAIN)
    val_ds = load_from_disk(TOKENIZED_VAL)
    test_ds = load_from_disk(TOKENIZED_TEST)

    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).astype("float32")
    y_val   = np.load(Y_VAL_PATH, allow_pickle=True).astype("float32")
    y_test  = np.load(Y_TEST_PATH, allow_pickle=True).astype("float32")

    # Add labels (multi-label → sequence of floats)
    train_ds = train_ds.add_column("labels", list(y_train))
    val_ds   = val_ds.add_column("labels", list(y_val))
    test_ds  = test_ds.add_column("labels", list(y_test))

    # Explicitly declare dtype as sequence of float32
    train_ds = train_ds.cast_column("labels", Sequence(Value("float32")))
    val_ds   = val_ds.cast_column("labels", Sequence(Value("float32")))
    test_ds  = test_ds.cast_column("labels", Sequence(Value("float32")))

    return train_ds, val_ds, test_ds


# -------------------- Metrics --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))  # multi-label → sigmoid
    preds = (preds > 0.5).int().numpy()
    labels = labels.astype(float)  # <-- ensure labels are float for multi-label metrics

    # Metrics
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, preds, average="micro")

    # Accuracy (multi-label: exact match ratio / subset accuracy)
    subset_acc = accuracy_score(labels, preds)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc,
        "accuracy": subset_acc,
    }

# -------------------- Custom Collator --------------------
def float_data_collator(features):
    batch = default_data_collator(features)
    if "labels" in batch:
        batch["labels"] = batch["labels"].float()  # force float32 for BCEWithLogitsLoss
        # Debug: print dtype only once
        if not hasattr(float_data_collator, "printed"):
            print(f"[DEBUG] Collator labels dtype: {batch['labels'].dtype}")
            float_data_collator.printed = True
    return batch




def train_legalbert_with_best_params():

    os.makedirs(os.path.dirname(HP_TUNED_MODEL_PATH), exist_ok=True)
    # Load best params from JSON
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    # Initialize WandB run
    run = wandb.init(
        project="legal-clause-classifier",  
        name="legal-bert-with-focal-loss", 
        config={
            "epochs": best_params["epochs"],
            "batch_size": best_params["batch_size"],
            "learning_rate": best_params["learning_rate"],
            "weight_decay": best_params["weight_decay"],
            "warmup_ratio": best_params["warmup_ratio"],
            "dropout": best_params["dropout"],
            "grad_accum": best_params["grad_accum"],
            "loss": "Focal Loss"
        }
    )

    train_ds, val_ds, test_ds = load_data()

    # Load label list for num_labels
    with open(LABEL_LIST_PATH, "r") as f:
        label_list = json.load(f)

    model = get_legalbert_model(num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=HP_TUNED_MODEL_PATH,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        num_train_epochs=best_params["epochs"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        gradient_accumulation_steps=best_params.get("grad_accum", 1),
        logging_dir=os.path.join(HP_TUNED_MODEL_PATH, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="wandb",
        run_name="legal-bert-with-hp"
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=float_data_collator,  # <-- Ensure float labels
    )

    trainer.train()

    # Save model locally
    trainer.save_model(HP_TUNED_MODEL_PATH)
    print(f"Model saved at {HP_TUNED_MODEL_PATH}")

    # Upload model as next version of W&B artifact
    artifact = wandb.Artifact(
        name="legal-bert-v2", 
        type="model",
        description="LegalBERT model fine-tuned with focal loss and hyperparameters.",
        metadata=best_params
    )

    # Add saved model files to the artifact
    artifact.add_file(os.path.join(HP_TUNED_MODEL_PATH, "model.safetensors"))
    artifact.add_file(os.path.join(HP_TUNED_MODEL_PATH, "config.json"))
    artifact.add_file(os.path.join(HP_TUNED_MODEL_PATH, "training_args.bin"))

    # Log the new version of the model
    run.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()