import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from config import (TOKENIZED_VAL, Y_VAL_PATH, LEGAL_BERT_WITH_FOCAL_LOSS_PATH,
                    THRESHOLDS_PATH, THRESHOLD_METRICS_PATH, WANDB_ARTIFACTS_PATH, HP_TUNED_MODEL_PATH)
import wandb


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
BATCH_SIZE = 16  # adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_val_dataset():
    val_ds = load_from_disk(TOKENIZED_VAL)
    y_val = np.load(Y_VAL_PATH, allow_pickle=True).astype("float32")
    val_ds = val_ds.add_column("labels", list(y_val))
    return val_ds, y_val


def get_dataloader(dataset, batch_size=BATCH_SIZE):
    collator = DataCollatorWithPadding(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def compute_metrics(labels, preds):
    # labels & preds: numpy arrays (num_samples, num_classes)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, preds, average="micro")
    subset_acc = accuracy_score(labels, preds)
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc,
        "subset_accuracy": subset_acc,
    }


def get_best_thresholds(model, dataloader, y_true, num_classes):
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    # Collect predictions
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
    all_logits = torch.cat(all_logits).numpy()

    # Per-class threshold sweep
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_thresholds = np.ones(num_classes) * 0.5

    for class_idx in range(num_classes):
        best_f1 = 0
        for t in thresholds:
            preds = (sigmoid(torch.tensor(all_logits[:, class_idx])) > t).int().numpy()
            f1 = f1_score(y_true[:, class_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[class_idx] = t

    # Apply optimized thresholds to entire predictions
    y_pred = np.zeros_like(y_true)
    for i in range(num_classes):
        y_pred[:, i] = (sigmoid(torch.tensor(all_logits[:, i])) > best_thresholds[i]).int().numpy()

    metrics = compute_metrics(y_true, y_pred)
    return best_thresholds.tolist(), metrics


def run_threshold_optimization():
    # Initialize W&B run
    run = wandb.init(project="legal-clause-classifier", 
               name="threshold-optimization",
               job_type="threshold_opt")
    
    # Use artifact
    artifact = run.use_artifact(
        'scsthilakarathne-nibm/legal-clause-classifier/legal-bert-v2:v8', 
        type='model')
    artifact_dir = artifact.download()
    
    
    # Load validation dataset
    val_ds, y_val = load_val_dataset()
    val_loader = get_dataloader(val_ds)

    # Load trained model
    num_labels = y_val.shape[1]
    model = AutoModelForSequenceClassification.from_pretrained(
        artifact_dir, 
        num_labels=num_labels, 
        problem_type="multi_label_classification")
    model.to(DEVICE)

    # Run threshold optimization
    best_thresholds, val_metrics = get_best_thresholds(model, val_loader, y_val, num_labels)

    # Save thresholds and metrics
    os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(THRESHOLD_METRICS_PATH), exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(best_thresholds, f, indent=4)
    with open(THRESHOLD_METRICS_PATH, "w") as f:
        json.dump(val_metrics, f, indent=4)

    print("Optimized thresholds and validation metrics saved.")
    print("Validation Metrics:", val_metrics)

    # Adding artifacts to wandb
    thresholds_artifact = wandb.Artifact(
        name="thresholds-v2",  
        type="thresholds",    
        description="Optimized decision thresholds for LegalBERT model with focal loss, oversampling and best params",
        metadata={
            "note": "Thresholds optimized on validation set",
            "model_version": "os-legal-bert with focal loss and best hp"
        }
    )

    thresholds_artifact.add_file(THRESHOLDS_PATH)
    thresholds_artifact.add_file(THRESHOLD_METRICS_PATH)

    wandb.log_artifact(thresholds_artifact)
    
    wandb.finish()

