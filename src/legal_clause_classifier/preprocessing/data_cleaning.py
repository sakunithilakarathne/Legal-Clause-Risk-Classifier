import unicodedata
import pandas as pd
import json
from pathlib import Path
import ast
import re
import logging

from src.legal_clause_classifier.utils.logger import get_logger

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("preprocessing", "preprocessing.log"    )

def normalize_text(text: str) -> str:
    """
    Normalize legal text for consistent processing.
    - Unicode NFC normalization
    - Standardize quotes and dashes
    - Remove common page headers/footers
    - Collapse multiple spaces/newlines into one
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        logger.warning("Received non-string input in normalize_text.")
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize quotes & dashes
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
    )

    # Remove common page headers/footers
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", text, flags=re.I)

    # Remove standalone "Page X" or "Page: X"
    text = re.sub(r"\bPage[:\s]*\d+\b", " ", text, flags=re.I)

    # Collapse multiple spaces/newlines into a single space
    text = " ".join(text.split())

    # Final strip
    return text.strip()


# Remove near duplicates
def deduplicate_clauses(clauses, sim_threshold=0.9):
    logger.info(f"Starting deduplication on {len(clauses)} clauses (threshold={sim_threshold})")

    if not clauses:
        logger.warning("Empty clause list passed to deduplicate_clauses.")
        return []

    texts = [c["clause_text"] for c in clauses]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    sim_matrix = cosine_similarity(vectorizer)
    keep = []
    seen = set()
    for i, row in enumerate(sim_matrix):
        if i in seen: 
            continue
        dupes = {j for j, sim in enumerate(row) if sim > sim_threshold}
        seen |= dupes
        keep.append(clauses[i])
        
    logger.info(f"Deduplication complete. Reduced from {len(clauses)} to {len(keep)} unique clauses.")
    return keep


def filter_short_clauses(df, min_tokens=12):
    return df[df['clause_text'].str.split().str.len() >= min_tokens]


def filter_normalized_schema(dataset: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dataset[dataset["is_impossible"] == False]
    result_df = filtered_df[["id", "doc_id", "category_name", "answer_text"]]

    return result_df

def save_distinct_labels(dataset, output_path):
    all_labels = sorted({label for labels in dataset["categories_set"] for label in labels})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)


def convert_categories_set_to_list(dataset):
    dataset["categories_list"] = dataset["categories_set"].apply(
        lambda s: sorted(list(ast.literal_eval(s))) if isinstance(s, str) else sorted(list(s))
        )
    return dataset
