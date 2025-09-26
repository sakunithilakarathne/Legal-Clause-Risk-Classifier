import os
import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from sklearn.metrics import f1_score, average_precision_score, accuracy_score

from src.legal_clause_classifier.models.legal_bert import get_legalbert_model
from config import (
    ARTIFACTS_DIR, Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH,
    TOKENIZED_TRAIN, TOKENIZED_VAL, TOKENIZED_TEST,
    LABEL_LIST_PATH, LEGAL_BERT_MODEL_PATH
)
import json
from datasets import Value
from transformers import Trainer, TrainingArguments, default_data_collator



# ==== Training Parameters (easy to tune) ====
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500



def load_data():
    train_ds = load_from_disk(TOKENIZED_TRAIN)
    val_ds = load_from_disk(TOKENIZED_VAL)
    test_ds = load_from_disk(TOKENIZED_TEST)

    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).astype("float32")
    y_val = np.load(Y_VAL_PATH, allow_pickle=True).astype("float32")
    y_test = np.load(Y_TEST_PATH, allow_pickle=True).astype("float32")

    # Add labels
    train_ds = train_ds.add_column("labels", list(y_train))
    val_ds = val_ds.add_column("labels", list(y_val))
    test_ds = test_ds.add_column("labels", list(y_test))

    # Force correct dtype
    train_ds = train_ds.cast_column("labels", Value("float32"))
    val_ds = val_ds.cast_column("labels", Value("float32"))
    test_ds = test_ds.cast_column("labels", Value("float32"))

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




# -------------------- Training --------------------
def train_legalbert_model():
    # Initialize WandB
    wandb.init(
        project="legal-clause-classifier",  
        name="legal-bert-v2", 
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
        run_name="legal-bert-v2"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=float_data_collator,  # <-- ensure float labels
    )

    trainer.train()

    # Evaluate on test set
    # test_results = trainer.evaluate(test_ds)
    # print("Test Results:", test_results)

    trainer.save_model(LEGAL_BERT_MODEL_PATH)
    print(f"Model saved at {LEGAL_BERT_MODEL_PATH}")

    wandb.finish()















# # -------------------- Training --------------------
# def train_legalbert_model():
    
#     # Initialize WandB
#     wandb.init(
#         project="legal-clause-classifier",  
#         name="legal-bert-v2", 
#         config={
#             "epochs": EPOCHS,
#             "batch_size": BATCH_SIZE,
#             "learning_rate": LEARNING_RATE,
#             "weight_decay": WEIGHT_DECAY,
#             "warmup_ratio": WARMUP_RATIO,
#         }
#     )

#     train_ds, val_ds, test_ds = load_data()

#     # Load label list for num_labels
#     with open(LABEL_LIST_PATH, "r") as f:
#         label_list = json.load(f)

#     model = get_legalbert_model(num_labels=len(label_list))

#     # Training parameters
#     training_args = TrainingArguments(
#         output_dir=os.path.join(ARTIFACTS_DIR, "legalbert_outputs"),
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=LEARNING_RATE,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=EPOCHS,
#         weight_decay=WEIGHT_DECAY,
#         logging_dir=os.path.join(ARTIFACTS_DIR, "logs"),
#         logging_steps=LOGGING_STEPS,
#         load_best_model_at_end=True,
#         metric_for_best_model="micro_f1",
#         greater_is_better=True,
#         report_to="wandb",
#         run_name="legal-bert-v2"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics
#     )

#     trainer.train()

#     # Evaluate on test set
#     # test_results = trainer.evaluate(test_ds)
#     # print("Test Results:", test_results)

#     trainer.save_model(LEGAL_BERT_MODEL_PATH)
#     print(
#         f"Model saved at {LEGAL_BERT_MODEL_PATH}")

#     wandb.finish()

# # # ==== Training Parameters (easy to tune) ====
# BATCH_SIZE = 8
# EPOCHS = 5
# LEARNING_RATE = 2e-5
# WEIGHT_DECAY = 0.01
# WARMUP_RATIO = 0.1
# LOGGING_STEPS = 50
# SAVE_STEPS = 500
# EVAL_STEPS = 500


# # -------------------- Load Data --------------------
# def load_data():
#     train_ds = load_from_disk(TOKENIZED_TRAIN)
#     val_ds = load_from_disk(TOKENIZED_VAL)
#     test_ds = load_from_disk(TOKENIZED_TEST)

#     y_train = np.load(Y_TRAIN_PATH, allow_pickle=True)
#     y_val = np.load(Y_VAL_PATH, allow_pickle=True)
#     y_test = np.load(Y_TEST_PATH, allow_pickle=True)

#     # Add labels back into HuggingFace datasets
#     train_ds = train_ds.add_column("labels", y_train.tolist())
#     val_ds = val_ds.add_column("labels", y_val.tolist())
#     test_ds = test_ds.add_column("labels", y_test.tolist())

#     return train_ds, val_ds, test_ds


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = torch.sigmoid(torch.tensor(logits))  # multi-label → sigmoid
#     preds = (preds > 0.5).int().numpy()
#     labels = labels.astype(int)

#     # Metrics
#     micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
#     macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
#     pr_auc = average_precision_score(labels, preds, average="micro")

#     # Accuracy (multi-label: exact match ratio OR subset accuracy)
#     subset_acc = accuracy_score(labels, preds)

#     return {
#         "micro_f1": micro_f1,
#         "macro_f1": macro_f1,
#         "pr_auc": pr_auc,
#         "accuracy": subset_acc,
#     }


# # -------------------- Training --------------------
# def train_legalbert_model():
    
#     #Initialize WandB
#     wandb.init(
#         project="legal-clause-classifier",  
#         name="legal-bert-v2", 
#         config={
#             "epochs": EPOCHS,
#             "batch_size": BATCH_SIZE,
#             "learning_rate": LEARNING_RATE,
#             "weight_decay": WEIGHT_DECAY,
#             "warmup_ratio": WARMUP_RATIO,
#         }
#     )

#     train_ds, val_ds, test_ds = load_data()

#     # Load label list for num_labels
#     with open(LABEL_LIST_PATH, "r") as f:
#         label_list = json.load(f)

#     model = get_legalbert_model(num_labels=len(label_list))

#     # Training parameters (easy to tweak later)
#     training_args = TrainingArguments(
#         output_dir=os.path.join(ARTIFACTS_DIR, "legalbert_outputs"),
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=LEARNING_RATE ,
#         per_device_train_batch_size=BATCH_SIZE, 
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs= EPOCHS,
#         weight_decay= WEIGHT_DECAY,
#         logging_dir=os.path.join(ARTIFACTS_DIR, "logs"),
#         logging_steps= LOGGING_STEPS,
#         load_best_model_at_end=True,
#         metric_for_best_model="micro_f1",
#         greater_is_better=True,

#         report_to="wandb",
#         run_name="legal-bert-v2"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics
#     )

#     trainer.train()

#     # Evaluate on test set
#     # test_results = trainer.evaluate(test_ds)
#     # print("Test Results:", test_results)

#     trainer.save_model(LEGAL_BERT_MODEL_PATH)
#     print(f"Model saved at {LEGAL_BERT_MODEL_PATH}")

#     wandb.finish()

    
    # import os
# import numpy as np
# import torch
# from datasets import load_from_disk
# from transformers import Trainer, TrainingArguments
# from sklearn.metrics import f1_score, average_precision_score, accuracy_score

# from src.legal_clause_classifier.models.legal_bert import LegalBERTClassifier
# from transformers import TrainingArguments
# from config import(
#     TOKENIZED_TRAIN, TOKENIZED_VAL,
#     Y_TRAIN_PATH, Y_VAL_PATH, LEGAL_BERT_MODEL_PATH
# )


# import wandb

# # ==== Training Parameters (easy to tune) ====
# BATCH_SIZE = 16
# EPOCHS = 3
# LEARNING_RATE = 2e-5
# WEIGHT_DECAY = 0.01
# WARMUP_RATIO = 0.1
# LOGGING_STEPS = 50
# SAVE_STEPS = 500
# EVAL_STEPS = 500



# def train_legalbert_model():

#     # Initialize WandB
#     wandb.init(
#         project="legal-clause-classifier",  
#         name="legal-bert-v1", 
#         config={
#             "epochs": EPOCHS,
#             "batch_size": BATCH_SIZE,
#             "learning_rate": LEARNING_RATE,
#             "weight_decay": WEIGHT_DECAY,
#             "warmup_ratio": WARMUP_RATIO,
#         }
#     )


#     # ==== Load datasets ====
#     train_dataset = load_from_disk(TOKENIZED_TRAIN)
#     val_dataset = load_from_disk(TOKENIZED_VAL)

#     y_train = np.load(Y_TRAIN_PATH)
#     y_val = np.load(Y_VAL_PATH)

#     # Attach labels into dataset objects
#     train_dataset = train_dataset.add_column("labels", y_train.tolist())
#     val_dataset = val_dataset.add_column("labels", y_val.tolist())

#     num_labels = y_train.shape[1]

#     # ==== Model ====
#     model = LegalBERTClassifier(num_labels=num_labels)

#     # ==== Metrics function ====
#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         preds = torch.sigmoid(torch.tensor(logits)).numpy()
#         labels = np.array(labels)

#         micro_f1 = f1_score(labels, preds > 0.5, average="micro")
#         macro_f1 = f1_score(labels, preds > 0.5, average="macro")
#         pr_auc = average_precision_score(labels, preds, average="micro")

#         # Calculate Accuracy
#         accuracy = accuracy_score(labels, preds > 0.5)

#         # Calculate Loss (using CrossEntropyLoss or another appropriate loss function)
#         loss_fn = torch.nn.BCEWithLogitsLoss()  # If using sigmoid + BCE for multi-label classification
#         loss = loss_fn(torch.tensor(logits), torch.tensor(labels)).item()

#         return {
#             "f1_micro": micro_f1,
#             "f1_macro": macro_f1,
#             "pr_auc": pr_auc,
#             "accuracy": accuracy,
#             "loss": loss
#         }
    
    
#     # ==== Training arguments ====
#     training_args = TrainingArguments(
#         output_dir=LEGAL_BERT_MODEL_PATH,
#         eval_strategy="steps",
#         eval_steps=EVAL_STEPS,
#         save_strategy="steps",
#         save_steps=SAVE_STEPS,
#         logging_steps=LOGGING_STEPS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=EPOCHS,
#         learning_rate=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#         warmup_ratio=WARMUP_RATIO,
#         load_best_model_at_end=True,
#         metric_for_best_model="micro_f1",
#         greater_is_better=True,

#         report_to="wandb",
#         run_name="legal-bert-v1"
#     )

#     # ==== Trainer ====
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=None,  # already tokenized
#         compute_metrics=compute_metrics
#     )

#     # ==== Train ====
#     trainer.train()

#     # Save final model
#     trainer.save_model(LEGAL_BERT_MODEL_PATH)
#     print(f"Model saved at {LEGAL_BERT_MODEL_PATH}")

#     wandb.finish()