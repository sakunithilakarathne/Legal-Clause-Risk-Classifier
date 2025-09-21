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

logger = get_logger("preprocessing")
run_id = logger.run_id


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
    logger.info(f"Filtering clauses with fewer than {min_tokens} tokens.")
    
    if 'clause_text' not in df.columns:
        logger.error("Missing 'clause_text' column in DataFrame.")
        raise KeyError("Column 'clause_text' not found in the input DataFrame.")
    
    initial_count = len(df)
    filtered_df = df[df['clause_text'].str.split().str.len() >= min_tokens]
    final_count = len(filtered_df)

    logger.info(f"Filtered {initial_count - final_count} clauses. Remaining: {final_count}")
    return filtered_df
    


def save_distinct_labels(dataset, output_path):
    logger.info("Saving distinct labels to JSON.")
    
    if 'categories_set' not in dataset.columns:
        logger.error("Missing 'categories_set' column in dataset.")
        raise KeyError("Column 'categories_set' not found in the input dataset.")
    
    try:
        all_labels = sorted({label for labels in dataset["categories_set"] for label in labels})
    except TypeError as e:
        logger.exception("Failed to extract labels from 'categories_set'.")
        raise e

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_labels, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(all_labels)} distinct labels to '{output_path}'.")
    except Exception as e:
        logger.exception(f"Error writing labels to {output_path}.")
        raise e



def convert_categories_set_to_list(dataset):

    logger.info("Converting 'categories_set' to 'categories_list'.")

    if 'categories_set' not in dataset.columns:
        logger.error("Missing 'categories_set' column in dataset.")
        raise KeyError("Column 'categories_set' not found in the dataset.")

    def parse_and_sort(s):
        try:
            if isinstance(s, str):
                s = ast.literal_eval(s)
            return sorted(list(s))
        except Exception as e:
            logger.warning(f"Failed to parse entry in 'categories_set': {s}")
            return []

    dataset["categories_list"] = dataset["categories_set"].apply(parse_and_sort)
    logger.info("Successfully created 'categories_list' column.")

    return dataset
