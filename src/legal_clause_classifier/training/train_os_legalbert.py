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
    ARTIFACTS_DIR, OS_Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH,
    TOKENIZED_VAL, TOKENIZED_TEST,
    LABEL_LIST_PATH, TOKENIZED_TRAIN_OS,
    OS_LEGAL_BERT_WITH_FOCAL_LOSS,
    
)
import json
from datasets import Value
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import load_from_disk, Sequence, Value

config = {
    "epochs": 4,
    "batch_size": 8,
    "learning_rate": 4.87e-5,
    "weight_decay": 0.0077,
    "warmup_ratio": 0.005
}

# ====  Parameters of Legal BERT ====
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 4.87e-5
WEIGHT_DECAY = 0.0077
WARMUP_RATIO = 0.005
LOGGING_STEPS = 50
DROPOUT_RATE = 0.245
GRAD_ACCUM = 1
SAVE_STEPS = 500
EVAL_STEPS = 500

def load_data():
    train_ds = load_from_disk(TOKENIZED_TRAIN_OS)
    val_ds = load_from_disk(TOKENIZED_VAL)
    
    y_train = np.load(OS_Y_TRAIN_PATH, allow_pickle=True).astype("float32")
    y_val   = np.load(Y_VAL_PATH, allow_pickle=True).astype("float32")
    

    # Add labels (multi-label → sequence of floats)
    train_ds = train_ds.add_column("labels", list(y_train))
    val_ds   = val_ds.add_column("labels", list(y_val))
    

    # Explicitly declare dtype as sequence of float32
    train_ds = train_ds.cast_column("labels", Sequence(Value("float32")))
    val_ds   = val_ds.cast_column("labels", Sequence(Value("float32")))
   
    return train_ds, val_ds


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



# ------------ Legal BERT with Focal Loss ------------------ #



def train_legalbert_with_focal_loss():
    # Initialize WandB run
    run = wandb.init(
        project="legal-clause-classifier",  
        name="lb-with-oversampling&focalloss", 
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "loss": "Focal Loss"
        }
    )

    train_ds, val_ds = load_data()

    # Load label list for num_labels
    with open(LABEL_LIST_PATH, "r") as f:
        label_list = json.load(f)

    model = get_legalbert_model(num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=OS_LEGAL_BERT_WITH_FOCAL_LOSS,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_dir=os.path.join(ARTIFACTS_DIR, "logs"),
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="wandb",
        run_name="lb-with-os-focalloss"
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
    trainer.save_model(OS_LEGAL_BERT_WITH_FOCAL_LOSS)
    print(f"Model saved at {OS_LEGAL_BERT_WITH_FOCAL_LOSS}")

    # Upload model as next version of W&B artifact
    artifact = wandb.Artifact(
        name="legal-bert-v2",  # SAME name = auto-versioning (v0 → v1)
        type="model",
        description="LegalBERT model with oversampling fine-tuned with focal loss.",
        metadata=config
    )

    # Add saved model files to the artifact
    artifact.add_file(os.path.join(OS_LEGAL_BERT_WITH_FOCAL_LOSS, "model.safetensors"))
    artifact.add_file(os.path.join(OS_LEGAL_BERT_WITH_FOCAL_LOSS, "config.json"))
    artifact.add_file(os.path.join(OS_LEGAL_BERT_WITH_FOCAL_LOSS, "training_args.bin"))

    # Log the new version of the model
    run.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()
    