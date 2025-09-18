import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


def split_dataset(dataset):
    
    doc_labels = (
        dataset.groupby("doc_id")["categories_set"]
        .apply(lambda sets: any("no-risk" in s for s in sets))
        .astype(int)  # binary label: has_no_risk_clauses
        .reset_index()
        )

    # Split by doc_id to preven data leakage
    train_docs, temp_docs = train_test_split(
        doc_labels,
        test_size=0.2,   # 80/20 first split
        stratify=doc_labels["categories_set"],  # stratify by no-risk presence
        random_state=42
        )

    val_docs, test_docs = train_test_split(
        temp_docs,
        test_size=0.5,   # 10% / 10% final
        stratify=temp_docs["categories_set"],
        random_state=42
        )

    train_df = dataset[dataset["doc_id"].isin(train_docs["doc_id"])]
    val_df   = dataset[dataset["doc_id"].isin(val_docs["doc_id"])]
    test_df  = dataset[dataset["doc_id"].isin(test_docs["doc_id"])]

    return train_df, test_df, val_df
