import pandas as pd
import numpy as np
import pickle
import os
import json
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import logging
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import pickle

from src.legal_clause_classifier.utils.logger import get_logger

from config import *

logger = get_logger("preprocessing")
run_id = logger.run_id


def multi_hot_encoding(train_df, val_df, test_df, label_path):
    
    logger.info("Starting multi-hot encoding.")

    try:
        with open(label_path) as f:
            all_labels = json.load(f)
        logger.info(f"Loaded {len(all_labels)} labels from {label_path}.")
    except FileNotFoundError:
        logger.error(f"Label list not found at {label_path}.")
        raise

    mlb = MultiLabelBinarizer(classes=all_labels)

    try:
        y_train = mlb.fit_transform(train_df["categories_list"])
        y_val = mlb.transform(val_df["categories_list"])
        y_test = mlb.transform(test_df["categories_list"])
        logger.info("Multi-hot encoding successful.")
    except Exception as e:
        logger.exception("Error during multi-hot encoding.")
        raise e

    try:
        with open(LABEL_BINARIZER_PATH, "wb") as f:
            pickle.dump(mlb, f)
        logger.info(f"Saved label binarizer to {LABEL_BINARIZER_PATH}.")

        np.save(Y_TRAIN_PATH , y_train)
        np.save(Y_VAL_PATH, y_val)
        np.save(Y_TEST_PATH, y_test)
        logger.info("Saved encoded label arrays to .npy files.")
    except Exception as e:
        logger.exception("Failed to save label binarizer or encoded arrays.")
        raise e

    # Optionally add as columns
    train_df["y_multihot"] = list(y_train)
    val_df["y_multihot"] = list(y_val)
    test_df["y_multihot"] = list(y_test)

    logger.info("Appended multi-hot vectors to dataframes.")

    return train_df,val_df,test_df



def tfidf_vectorization(train_df, val_df, test_df):

    logger.info("Starting TF-IDF vectorization.")

    # Save matrices (scipy sparse -> npz)
    tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2
    )
    try:
        X_train_tfidf = tfidf.fit_transform(train_df["clause_text"])
        X_val_tfidf = tfidf.transform(val_df["clause_text"])
        X_test_tfidf = tfidf.transform(test_df["clause_text"])
        logger.info("TF-IDF vectorization completed.")
    except Exception as e:
        logger.exception("Error during TF-IDF transformation.")
        raise e

    try:
        with open(TFIDF_MODEL_PATH , "wb") as f:
            pickle.dump(tfidf, f)
        logger.info(f"Saved TF-IDF model to {TFIDF_MODEL_PATH}")

        sparse.save_npz(X_TRAIN_TFIDF_PATH, X_train_tfidf)
        sparse.save_npz(X_VAL_TFIDF_PATH, X_val_tfidf)
        sparse.save_npz(X_TEST_TFIDF_PATH, X_test_tfidf)
        logger.info("Saved TF-IDF matrices as sparse .npz files.")
    except Exception as e:
        logger.exception("Failed to save TF-IDF model or matrices.")
        raise e
    
    return X_train_tfidf, X_test_tfidf, X_val_tfidf



# def lstm_tokenization(train_df, val_df, test_df):
#     MAX_LEN = 256  # pad/truncate length

#     def build_vocab(texts, min_freq=2):
#         counter = Counter()
#         for text in texts:
#             tokens = word_tokenize(text.lower())
#             counter.update(tokens)
#         return Vocab(counter, min_freq=min_freq, specials=['<unk>', '<pad>'])

#     def encode_text(text, vocab, max_len=MAX_LEN):
#         tokens = word_tokenize(text.lower())
#         ids = [vocab[token] for token in tokens[:max_len]]
#         if len(ids) < max_len:
#             ids += [vocab['<pad>']] * (max_len - len(ids))
#         return np.array(ids)

#     # Build vocab from train only
#     vocab = build_vocab(train_df["clause_text"].tolist(), min_freq=2)
#     with open(LSTM_VOCAB_PATH, "wb") as f:
#         pickle.dump(vocab, f)

#     # Encode datasets
#     train_ids = np.stack([encode_text(t, vocab) for t in train_df["clause_text"]])
#     np.save(LSTM_TOKENIZED_TRAIN, train_ids)
#     logger.info("Saved Tokenized train dataset.")
#     test_ids = np.stack([encode_text(t, vocab) for t in test_df["clause_text"]])
#     np.save(LSTM_TOKENIZED_TEST, test_ids)
#     logger.info("Saved Tokenized test dataset.")
#     val_ids = np.stack([encode_text(t, vocab) for t in val_df["clause_text"]])
#     np.save(LSTM_TOKENIZED_VAL, val_ids)
#     logger.info("Saved Tokenized validation dataset.")



def lstm_tokenization(train_df, val_df, test_df):
    MAX_LEN = 256  # pad/truncate length
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"


    def build_vocab(texts, min_freq=2):
        counter = Counter()
        for text in texts:
            tokens = word_tokenize(text.lower())
            counter.update(tokens)

        # Start vocab with special tokens
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        idx = 2
        for token, freq in counter.items():
            if freq >= min_freq:
                vocab[token] = idx
                idx += 1
        return vocab

    def encode_text(text, vocab, max_len=MAX_LEN):
        tokens = word_tokenize(text.lower())
        ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
        return np.array(ids)

    # Build vocab from training set only
    vocab = build_vocab(train_df["clause_text"].tolist(), min_freq=2)

    # Save vocab for reuse
    with open(LSTM_VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    # Encode splits
    train_ids = np.stack([encode_text(t, vocab) for t in train_df["clause_text"]])
    val_ids = np.stack([encode_text(t, vocab) for t in val_df["clause_text"]])
    test_ids = np.stack([encode_text(t, vocab) for t in test_df["clause_text"]])

    # Save encoded arrays
    np.save(LSTM_TOKENIZED_TRAIN, train_ids)
    np.save(LSTM_TOKENIZED_VAL, val_ids)
    np.save(LSTM_TOKENIZED_TEST, test_ids)


def transformer_tokenization(train_df, test_df, val_df):
    logger.info("Starting transformer tokenization.")

    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
    MAX_LEN = 384

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Loaded tokenizer: {MODEL_NAME}")
    except Exception as e:
        logger.exception(f"Failed to load tokenizer {MODEL_NAME}")
        raise e

    # Tokenization function
    def tokenize_batch(batch):
        return tokenizer(
            batch["clause_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    # Convert DataFrames to Datasets
    try:
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)
        logger.info("Converted DataFrames to Hugging Face Datasets.")
    except Exception as e:
        logger.exception("Failed to convert DataFrames to Datasets.")
        raise e

    # Tokenize datasets
    try:
        train_tokenized = train_ds.map(tokenize_batch, batched=True, remove_columns=list(train_df.columns))
        val_tokenized = val_ds.map(tokenize_batch, batched=True, remove_columns=list(val_df.columns))
        test_tokenized = test_ds.map(tokenize_batch, batched=True, remove_columns=list(test_df.columns))
        logger.info("Applied tokenizer to datasets.")
    except Exception as e:
        logger.exception("Error during tokenization.")
        raise e

    # Save tokenized datasets
    try:
        train_tokenized.save_to_disk(TOKENIZED_TRAIN)
        val_tokenized.save_to_disk(TOKENIZED_VAL)
        test_tokenized.save_to_disk(TOKENIZED_TEST)
        logger.info("Saved tokenized datasets to disk.")
    except Exception as e:
        logger.exception("Failed to save tokenized datasets.")
        raise e

    return train_tokenized, val_tokenized, test_tokenized




    
# def transformer_tokenization(train_df, test_df, val_df):

#     logger.info("Starting transformer tokenization.")

#     MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
#     MAX_LEN = 384

#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         logger.info(f"Loaded tokenizer: {MODEL_NAME}")
#     except Exception as e:
#         logger.exception(f"Failed to load tokenizer {MODEL_NAME}")
#         raise e

#     def tokenize_batch(batch):
#         return tokenizer(
#             batch["clause_text"],
#             padding="max_length",
#             truncation=True,
#             max_length=MAX_LEN
#         )

#     try:
#         train_ds = Dataset.from_pandas(train_df)
#         val_ds = Dataset.from_pandas(val_df)
#         test_ds = Dataset.from_pandas(test_df)
#         logger.info("Converted DataFrames to Hugging Face Datasets.")
#     except Exception as e:
#         logger.exception("Failed to convert DataFrames to Datasets.")
#         raise e

#     try:
#         train_tokenized = train_ds.map(tokenize_batch, batched=True, remove_columns=train_df.columns)
#         val_tokenized = val_ds.map(tokenize_batch, batched=True, remove_columns=val_df.columns)
#         test_tokenized = test_ds.map(tokenize_batch, batched=True, remove_columns=test_df.columns)
#         logger.info("Applied tokenizer to datasets.")
#     except Exception as e:
#         logger.exception("Error during tokenization.")
#         raise e

#     try:
#         train_tokenized.save_to_disk(TOKENIZED_TRAIN)
#         val_tokenized.save_to_disk(TOKENIZED_VAL)
#         test_tokenized.save_to_disk(TOKENIZED_TEST)
#         logger.info("Saved tokenized datasets to disk.")
#     except Exception as e:
#         logger.exception("Failed to save tokenized datasets.")
#         raise e

#     return train_tokenized, val_tokenized, test_tokenized
