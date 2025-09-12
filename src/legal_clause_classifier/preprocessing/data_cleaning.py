import json
import unicodedata
import re
import pandas as pd
from pathlib import Path

# def normalize_text(text: str, lowercase: bool = True) -> str:

#     if not isinstance(text, str):
#         return ""

#     # Normalize Unicode to composed form (e.g., é is one character, not e + ´)
#     text = unicodedata.normalize("NFC", text)

#     # Replace fancy quotes and dashes with standard ASCII equivalents
#     text = text.replace("’", "'").replace("‘", "'") # Handles right and left single quotes
#     text = text.replace("“", '"').replace("”", '"') # Handles right and left double quotes
#     text = text.replace("–", "-").replace("—", "-") # Handles en-dash and em-dash

#     # Remove any non-printable characters
#     # This regex keeps letters, numbers, basic punctuation, and whitespace.
#     text = re.sub(r'[^\w\s\'\"\-\.\,\(\)\;\:?\!]', '', text)

#     # Collapse all whitespace (spaces, newlines, tabs) to a single space
#     text = " ".join(text.split())
#     text = text.lower()

#     return text.strip()



def remove_duplicates(dataset: pd.DataFrame)-> pd.DataFrame:
    dataset_unique = dataset.drop_duplicates(subset=['answer_text'], keep='first', inplace=False)

    return dataset_unique



def filter_normalized_schema(dataset: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dataset[dataset["is_impossible"] == False]
    result_df = filtered_df[["id", "doc_id", "category_name", "answer_text"]]

    return result_df