import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from src.legal_clause_classifier.utils.logger import get_logger

logger = get_logger("preprocessing", "preprocessing.log")


def split_dataset(dataset):
    logger.info(f"Starting dataset split. Total records: {len(dataset)}")
    
    doc_labels = (
        dataset.groupby("doc_id")["categories_set"]
        .apply(lambda sets: any("no-risk" in s for s in sets))
        .astype(int)  # binary label: has_no_risk_clauses
        .reset_index()
        )
    logger.info(f"Generated binary labels for {len(doc_labels)} documents.")

    try:
        # Split by doc_id to preven data leakage
        train_docs, temp_docs = train_test_split(
            doc_labels,
            test_size=0.2,   # 80/20 first split
            stratify=doc_labels["categories_set"],  # stratify by no-risk presence
            random_state=42
            )
        logger.info(f"Train docs: {len(train_docs)}, Temp docs: {len(temp_docs)}")
        print("Train docs:", train_df["doc_id"].nunique(), "clauses:", len(train_df))

        val_docs, test_docs = train_test_split(
            temp_docs,
            test_size=0.5,   # 10% / 10% final
            stratify=temp_docs["categories_set"],
            random_state=42
            )
        logger.info(f"Validation docs: {len(val_docs)}, Test docs: {len(test_docs)}")

    except ValueError as e:
        logger.error(f"Stratified splitting failed: {e}")
        raise

    train_df = dataset[dataset["doc_id"].isin(train_docs["doc_id"])]
    val_df   = dataset[dataset["doc_id"].isin(val_docs["doc_id"])]
    test_df  = dataset[dataset["doc_id"].isin(test_docs["doc_id"])]

    logger.info(f"Split complete. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train docs: {train_df["doc_id"].nunique()}, Clauses: {len(train_df)}")
    logger.info(f"Test docs: {train_df["doc_id"].nunique()}, Clauses: {len(train_df)}")
    logger.info(f"Train docs: {train_df["doc_id"].nunique()}, Clauses: {len(train_df)}")

    return train_df, test_df, val_df
