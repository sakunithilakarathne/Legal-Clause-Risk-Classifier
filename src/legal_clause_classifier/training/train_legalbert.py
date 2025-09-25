import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, average_precision_score

from src.legal_clause_classifier.models.legal_bert import LegalBERTClassifier
from config import(
    TOKENIZED_TRAIN, TOKENIZED_VAL,
    Y_TRAIN_PATH, Y_VAL_PATH, LEGAL_BERT_MODEL_PATH
)


# ==== Training Parameters (easy to tune) ====
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500

# ==== Load datasets ====
train_dataset = load_from_disk(TOKENIZED_TRAIN)
val_dataset = load_from_disk(TOKENIZED_VAL)

y_train = np.load(Y_TRAIN_PATH)
y_val = np.load(Y_VAL_PATH)

# Attach labels into dataset objects
train_dataset = train_dataset.add_column("labels", y_train.tolist())
val_dataset = val_dataset.add_column("labels", y_val.tolist())

num_labels = y_train.shape[1]

# ==== Model ====
model = LegalBERTClassifier(num_labels=num_labels)

# ==== Metrics function ====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)

    micro_f1 = f1_score(labels, preds > 0.5, average="micro")
    macro_f1 = f1_score(labels, preds > 0.5, average="macro")
    pr_auc = average_precision_score(labels, preds, average="micro")

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc
    }

# ==== Training arguments ====
training_args = TrainingArguments(
    output_dir=LEGAL_BERT_MODEL_PATH,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    report_to="none"  # disable wandb/hub reporting unless needed
)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,  # already tokenized
    compute_metrics=compute_metrics
)

# ==== Train ====
trainer.train()

# Save final model
trainer.save_model(LEGAL_BERT_MODEL_PATH)
print(f"Model saved at {LEGAL_BERT_MODEL_PATH}")
