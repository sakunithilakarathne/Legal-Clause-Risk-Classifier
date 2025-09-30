import os
import json
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from config import (TOKENIZED_TEST,Y_TEST_PATH, FINAL_TEST_METRICES)

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
BATCH_SIZE = 16  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_dataloader(dataset, batch_size=BATCH_SIZE):
    collator = DataCollatorWithPadding(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

def compute_metrics(labels, preds):
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

def evaluate_on_test(model, test_dataset, thresholds, batch_size=BATCH_SIZE):
    test_loader = get_dataloader(test_dataset, batch_size)
    sigmoid = torch.nn.Sigmoid()

    all_logits = []
    y_true = np.array(test_dataset["labels"])

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())

    all_logits = torch.cat(all_logits).numpy()

    # Apply thresholds
    y_pred = np.zeros_like(y_true)
    for i, t in enumerate(thresholds):
        y_pred[:, i] = (sigmoid(torch.tensor(all_logits[:, i])) > t).int().numpy()

    metrics = compute_metrics(y_true, y_pred)
    return metrics


def run_test_evaluation():

    run = wandb.init(
        project="legal-clause-classifier", 
        name="test-evaluation", 
        job_type="test_eval")

    # Load thresholds artifact
    threshold_artifact = run.use_artifact(
        'scsthilakarathne-nibm/legal-clause-classifier/thresholds-v2:v0', 
        type='thresholds')
    threshold_artifact_dir = threshold_artifact.download()

    # Load model artifact
    model_artifact = run.use_artifact(
        'scsthilakarathne-nibm/legal-clause-classifier/legal-bert-v2:v8', 
        type='model')
    model_artifact_dir = model_artifact.download()

    with open(os.path.join(threshold_artifact_dir, "optimized_thresholds.json"), "r") as f:
        thresholds = json.load(f)

    # Load test dataset
    test_ds = load_from_disk(TOKENIZED_TEST)
    y_test = np.load(Y_TEST_PATH, allow_pickle=True).astype("float32")
    test_ds = test_ds.add_column("labels", list(y_test))

    # Load model
    num_labels = y_test.shape[1]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_artifact_dir,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(DEVICE).eval()

    # Run evaluation
    test_metrics = evaluate_on_test(model, test_ds, thresholds)
    print("Test Metrics:", test_metrics)

    # Save locally
    with open(FINAL_TEST_METRICES, "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Log to W&B
    test_artifact = wandb.Artifact(
        name="test-metrics-v1",
        type="metrics",
        description="Final test metrics with threshold-optimized LegalBERT"
    )
    test_artifact.add_file(FINAL_TEST_METRICES)
    wandb.log_artifact(test_artifact)

    wandb.finish()

