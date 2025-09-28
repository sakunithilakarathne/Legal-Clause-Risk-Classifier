import os
import json
import optuna
import wandb
import numpy as np
import torch

from datasets import load_from_disk
from datasets import Value, Sequence
from sklearn.metrics import f1_score, accuracy_score, average_precision_score

from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from config import (TOKENIZED_TRAIN, TOKENIZED_VAL, Y_TRAIN_PATH, Y_VAL_PATH, ARTIFACTS_DIR,
                    LABEL_LIST_PATH, LEGAL_BERT_MODEL_PATH, LOGS_DIR, BEST_PARAMS_PATH, HP_ARTIFACTS_PATH)



MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# -------------------- Load Data --------------------
def load_data():
    train_ds = load_from_disk(TOKENIZED_TRAIN)
    val_ds = load_from_disk(TOKENIZED_VAL)

    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).astype("float32")
    y_val   = np.load(Y_VAL_PATH, allow_pickle=True).astype("float32")

    train_ds = train_ds.add_column("labels", list(y_train))
    val_ds   = val_ds.add_column("labels", list(y_val))

    train_ds = train_ds.cast_column("labels", Sequence(Value("float32")))
    val_ds   = val_ds.cast_column("labels", Sequence(Value("float32")))

    return train_ds, val_ds


# -------------------- Metrics --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))
    preds = (preds > 0.5).int().numpy()
    labels = labels.astype(float)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, preds, average="micro")
    subset_acc = accuracy_score(labels, preds)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc,
        "accuracy": subset_acc,
    }


# -------------------- Data Collator --------------------
def float_data_collator(features):
    batch = default_data_collator(features)
    if "labels" in batch:
        batch["labels"] = batch["labels"].float()
    return batch


# -------------------- Objective for Optuna --------------------
def objective(trial):
    train_ds, val_ds = load_data()

    # Load labels
    with open(LABEL_LIST_PATH, "r") as f:
        label_list = json.load(f)

    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    per_device_batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("epochs", 3, 6)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    # W&B logging
    wandb.init(
        project="legal-clause-classifier",
        name=f"optuna-trial-{trial.number}",
        reinit=True,
        config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": per_device_batch_size,
            "epochs": num_train_epochs,
            "warmup_ratio": warmup_ratio,
        }
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(ARTIFACTS_DIR, f"trial_{trial.number}"),
        eval_strategy="epoch",
        save_strategy="no",  # avoid checkpoint clutter during tuning
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        report_to="wandb",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=float_data_collator,
    )

    result = trainer.train()
    metrics = trainer.evaluate()

    wandb.log(result)
    wandb.log(metrics)
    wandb.finish()

    # Optuna maximizes → micro_f1
    return metrics["eval_micro_f1"]


# -------------------- Main --------------------
def hyperparameter_tuning():
    study = optuna.create_study(
        direction="maximize",
        study_name="legalbert_hp_tuning"
    )
    study.optimize(objective, n_trials=10)  # adjust trials as 
    
    best_trial = study.best_trial
    best_params = best_trial.params

    # Save best parameters to JSON
    
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best parameters saved to {BEST_PARAMS_PATH}")
    print("Best trial:", best_params)

    # Save Optimization History
    optuna.visualization.plot_optimization_history(study).show()
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(HP_ARTIFACTS_PATH, "optuna_optimization_history.html"))

    # Save Parameter Importance
    optuna.visualization.plot_param_importances(study).show()
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(HP_ARTIFACTS_PATH, "optuna_param_importances.html"))

    # Save Parallel Coordinate Plot
    optuna.visualization.plot_parallel_coordinate(study).show()
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(HP_ARTIFACTS_PATH, "optuna_parallel_coordinates.html"))

    print(f"[INFO] Optuna visualizations saved to {HP_ARTIFACTS_PATH}")