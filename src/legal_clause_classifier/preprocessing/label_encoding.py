import pandas as pd
import numpy as np
import pickle
import os
import json
from transformers import AutoTokenizer
import torch
from datasets import Dataset


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def multi_hot_encoding(train_df, val_df, test_df):
    with open("artifacts/label_list.json") as f:
        all_labels = json.load(f)

    mlb = MultiLabelBinarizer(classes=all_labels)
    y_train = mlb.fit_transform(train_df["categories_list"])
    y_val = mlb.transform(val_df["categories_list"])
    y_test = mlb.transform(test_df["categories_list"])

    # Save fitted mlb
    with open("artifacts/label_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)

    # Save y arrays (can also merge into parquet if you prefer)
    np.save("artifacts/y_train.npy", y_train)
    np.save("artifacts/y_val.npy", y_val)
    np.save("artifacts/y_test.npy", y_test)

    # Optionally, save inside Parquet as new columns
    train_df["y_multihot"] = list(y_train)
    val_df["y_multihot"] = list(y_val)
    test_df["y_multihot"] = list(y_test)

    return train_df,val_df,test_df


def tfidf_vectorization(train_df, val_df, test_df):
    # Save matrices (scipy sparse -> npz)
    tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2
)

    X_train_tfidf = tfidf.fit_transform(train_df["clause_text"])
    X_val_tfidf = tfidf.transform(val_df["clause_text"])
    X_test_tfidf = tfidf.transform(test_df["clause_text"])

    # Save TF-IDF model
    with open("artifacts/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # Save matrices (scipy sparse -> npz)
    from scipy import sparse
    sparse.save_npz("artifacts/X_train_tfidf.npz", X_train_tfidf)
    sparse.save_npz("artifacts/X_val_tfidf.npz", X_val_tfidf)
    sparse.save_npz("artifacts/X_test_tfidf.npz", X_test_tfidf)

    return X_train_tfidf, X_test_tfidf, X_val_tfidf

    
def transformer_tokenization(train_df, test_df, val_df):
    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # or "roberta-base"
    MAX_LEN = 384

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["clause_text_normalized"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    # Convert DataFrames -> Hugging Face Datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    # Apply tokenization
    train_tokenized = train_ds.map(tokenize_batch, batched=True, remove_columns=train_df.columns)
    val_tokenized = val_ds.map(tokenize_batch, batched=True, remove_columns=val_df.columns)
    test_tokenized = test_ds.map(tokenize_batch, batched=True, remove_columns=test_df.columns)

    # Save as Arrow datasets
    train_tokenized.save_to_disk("artifacts/train_tokenized")
    val_tokenized.save_to_disk("artifacts/val_tokenized")
    test_tokenized.save_to_disk("artifacts/test_tokenized")

    return train_tokenized, val_tokenized, test_tokenized
