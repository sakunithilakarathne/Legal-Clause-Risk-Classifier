import torch
import numpy as np
import wandb
from transformers import (
    Trainer,
    TrainingArguments
)
import os
from config import (
    ARTIFACTS_DIR, Y_TRAIN_PATH, POS_WEIGHTS_PATH,
    LABEL_LIST_PATH, LEGAL_BERT_WITH_POS_MODEL_PATH
)
import json
from datasets import Value
from transformers import Trainer, TrainingArguments

from src.legal_clause_classifier.models.legal_bert import get_legalbert_model
from src.legal_clause_classifier.training.train_legalbert import load_data , compute_metrics, float_data_collator


# ==== Training Parameters (easy to tune) ====
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500


# Compute positive class weights for BCEWithLogitsLoss.
def compute_class_weights(y_train: np.ndarray):

    n_samples, n_labels = y_train.shape
    pos_counts = y_train.sum(axis=0)
    neg_counts = n_samples - pos_counts

    # Avoid division by zero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    np.save(POS_WEIGHTS_PATH, pos_weights)

    return torch.tensor(pos_weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
# class WeightedTrainer(Trainer):
#     def __init__(self, pos_weight=None, **kwargs):
#         super().__init__(**kwargs)
#         self.pos_weight = pos_weight

#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # Adding BCE with pos_weight for imbalance handling
#         loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
#         loss = loss_fct(logits, labels)

#         return (loss, outputs) if return_outputs else loss


def train_legalbert_with_pos_weight():
    # Initialize WandB
    wandb.init(
        project="legal-clause-classifier",  
        name="legal-bert-with-pos-weight", 
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
        }
    )

    train_ds, val_ds, test_ds = load_data()

    # Load label list for num_labels
    with open(LABEL_LIST_PATH, "r") as f:
        label_list = json.load(f)

    # Compute pos_weight from training set
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).astype("float32")
    pos_weight = compute_class_weights(y_train)

    model = get_legalbert_model(num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=os.path.join(ARTIFACTS_DIR, "legalbert_outputs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(ARTIFACTS_DIR, "logs"),
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to="wandb",
        run_name="legal-bert-class-imbalance-handling"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=float_data_collator,
        pos_weight=pos_weight,  # Imbalance Handling using pos_weight
    )

    trainer.train()

    trainer.save_model(LEGAL_BERT_WITH_POS_MODEL_PATH)
    print(f"Model saved at {LEGAL_BERT_WITH_POS_MODEL_PATH}")

    # Log model to WandB as an artifact
    artifact = wandb.Artifact(
    name="legal-bert-v2",  # Same name to version it
    type="model",
    description="LegalBERT model fine-tuned with class imbalance handling via pos_weight.",
    metadata=wandb.config
    )
    
    
    artifact.add_file("artifacts/legalbert_with_posweights_outputs/model.safetensors")
    artifact.add_file("artifacts/legalbert_with_posweights_outputs/config.json")
    artifact.add_file("artifacts/legalbert_with_posweights_outputs/training_args.bin")

    wandb.log_artifact(artifact)

    wandb.finish()