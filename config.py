import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw Dataset
RAW_DATA_DIR_PATH = os.path.join(DATA_DIR,"raw")
RAW_DATASET_PATH = os.path.join(RAW_DATA_DIR_PATH,"CUAD_v1.json")

# Processed Dataset
PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR,"processed")
NORMALIZED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"normalized_dataset.csv")
EXTRACTED_CLAUSES_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"extracted_clauses.csv")
FINAL_CLAUSES_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"final_clauses.csv")
TRAIN_CLAUSES_WITH_MULTI_HOT_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"train_with_mh.csv")
TEST_CLAUSES_WITH_MULTI_HOT_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"test_with_mh.csv")
VAL_CLAUSES_WITH_MULTI_HOT_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"val_with_mh.csv")


# File paths
LABEL_LIST_PATH = os.path.join(ARTIFACTS_DIR, "label_list.json")
LABEL_BINARIZER_PATH = os.path.join(ARTIFACTS_DIR, "label_binarizer.pkl")
TFIDF_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")


# Train, Test and Validation sets in parquet form
TRAIN_PARQUET_PATH = os.path.join(ARTIFACTS_DIR, "train.parquet")
TEST_PARQUET_PATH = os.path.join(ARTIFACTS_DIR, "test.parquet")
VAL_PARQUET_PATH = os.path.join(ARTIFACTS_DIR, "val.parquet")
TRAIN_OVERSAMPLED_PATH = os.path.join(ARTIFACTS_DIR, "train_oversampled")

# Y Train, Test and Val sets
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "y_train.npy")
Y_VAL_PATH = os.path.join(ARTIFACTS_DIR, "y_val.npy")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "y_test.npy")

# TFIDF datasets
X_TRAIN_TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "X_train_tfidf.npz")
X_VAL_TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "X_val_tfidf.npz")
X_TEST_TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "X_test_tfidf.npz")

# LSTM Tokenized datasets
LSTM_VOCAB_PATH = os.path.join(ARTIFACTS_DIR,"lstm_vocab.pkl")
LSTM_TOKENIZED_TRAIN = os.path.join(ARTIFACTS_DIR, "X_train_lstm.npy")
LSTM_TOKENIZED_VAL = os.path.join(ARTIFACTS_DIR, "X_val_lstm.npy")
LSTM_TOKENIZED_TEST = os.path.join(ARTIFACTS_DIR, "X_test_lstm.npy")


# Tokenized datasets
TOKENIZED_TRAIN = os.path.join(ARTIFACTS_DIR, "train_tokenized")
TOKENIZED_VAL = os.path.join(ARTIFACTS_DIR, "val_tokenized")
TOKENIZED_TEST = os.path.join(ARTIFACTS_DIR, "test_tokenized")
TOKENIZED_TRAIN_OS = os.path.join(ARTIFACTS_DIR, "train_oversampled_tokenized")

# Logistic Regression Model
LR_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_model.pkl")

# ANN Model
ANN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ann_baseline.pt")

# Advanced ANN Model
ADVANCED_ANN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "advanced_ann.pt")

# LSTM MODEL
LSTM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lstm_model.pt")

# Legal Bert Model
#LEGAL_BERT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "legal_bert_model.pt")

LEGAL_BERT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "legalbert_outputs")

# HYPER PARAMETER TUNING
HP_ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR,"hp_tuning")
BEST_PARAMS_PATH = os.path.join(HP_ARTIFACTS_PATH, "best_params.json")

LEGAL_BERT_WITH_POS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "legalbert_with_posweights_outputs")
POS_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "pos_weight.npy")
LEGAL_BERT_WITH_FOCAL_LOSS_PATH = os.path.join(ARTIFACTS_DIR, "legalbert_with_focalloss_outputs")
LEGAL_BERT_WITH_RESAMPLING_PATH = os.path.join(ARTIFACTS_DIR, "legalbert_with_resampling_outputs")


THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "optimized_thresholds.json")
THRESHOLD_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "validation_metrics.json")

WANDB_ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "wandb_artifacts")

HP_TUNED_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "hp_tuned_model_outputs")
