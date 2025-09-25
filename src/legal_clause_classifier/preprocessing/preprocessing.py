import json
import pandas as pd
from config import *
import logging

from src.legal_clause_classifier.preprocessing.schema_normalization import *
from src.legal_clause_classifier.preprocessing.clause_extraction import*
from src.legal_clause_classifier.preprocessing.data_cleaning import *
from src.legal_clause_classifier.preprocessing.data_splitting import *
from src.legal_clause_classifier.preprocessing.label_encoding import *
from src.legal_clause_classifier.utils.logger import get_logger

logger = get_logger("preprocessing")


def preprocessing_pipeline(json_file: json):

    normalized_df = normalize_cuad_schema(json_file)
    normalized_df.to_csv(NORMALIZED_DATASET_PATH, index=False, encoding="utf-8")
    logger.info(f"Normalized Dataset saved to {NORMALIZED_DATASET_PATH}.")

    extracted_clauses_df = extract_clauses(normalized_df)
    filtered_clauses_df = filter_short_clauses(extracted_clauses_df)
    filtered_clauses_df.to_csv(EXTRACTED_CLAUSES_DATASET_PATH, index=False, encoding="utf-8")
    logger.info(f"Extarcted clause set saved to {EXTRACTED_CLAUSES_DATASET_PATH}.")

    save_distinct_labels(filtered_clauses_df, LABEL_LIST_PATH)

    final_df = convert_categories_set_to_list(filtered_clauses_df)
    final_df.to_csv(FINAL_CLAUSES_DATASET_PATH, index=False, encoding="utf-8")
    logger.info(f"Extarcted clause set saved to {FINAL_CLAUSES_DATASET_PATH}.")

    train_df, test_df, val_df = split_dataset(final_df)

    train_df.to_parquet(TRAIN_PARQUET_PATH, index=False)
    logger.info(f"Training dataset saved to: {TRAIN_PARQUET_PATH}.")
    val_df.to_parquet(VAL_PARQUET_PATH, index=False)
    logger.info(f"Validating dataset saved to: {VAL_PARQUET_PATH}.")
    test_df.to_parquet(TEST_PARQUET_PATH, index=False)
    logger.info(f"Testing dataset saved to: {TEST_PARQUET_PATH}.")


    train_df, val_df, test_df = multi_hot_encoding(train_df, val_df, test_df, LABEL_LIST_PATH)
    
    train_df.to_csv(TRAIN_CLAUSES_WITH_MULTI_HOT_PATH, index=False, encoding="utf-8")
    logger.info(f"Training dataset saved to {TRAIN_CLAUSES_WITH_MULTI_HOT_PATH}.")
    val_df.to_csv(VAL_CLAUSES_WITH_MULTI_HOT_PATH, index=False, encoding="utf-8")
    logger.info(f"Validating dataset saved to {VAL_CLAUSES_WITH_MULTI_HOT_PATH}.")
    test_df.to_csv(TEST_CLAUSES_WITH_MULTI_HOT_PATH, index=False, encoding="utf-8")
    logger.info(f"Testing dataset saved to {TEST_CLAUSES_WITH_MULTI_HOT_PATH}.")



    tfidf_vectorization(train_df, val_df, test_df)
    lstm_tokenization(train_df, val_df, test_df)
    transformer_tokenization(train_df, test_df, val_df)

    

