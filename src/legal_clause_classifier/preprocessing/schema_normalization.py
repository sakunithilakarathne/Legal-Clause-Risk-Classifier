import json
import re
import pandas as pd
from pathlib import Path
from src.legal_clause_classifier.preprocessing.data_cleaning import *



def load_dataset(file_path: str) -> dict:
    with open(file_path,"r",encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset


def normalize_cuad_schema(data: dict) -> pd.DataFrame:
    """
    Normalize CUAD JSON into flat schema with stable IDs.

    Columns:
    - id: unique identifier for each QA pair (PAIR_xxx)
    - doc_id: numeric running ID (DOC_0001, DOC_0002, …)
    - doc_name: original CUAD title (or 'Unknown' if missing)
    - qa_id: CUAD QA ID string
    - category_name: human-readable category (derived from qa_id suffix)
    - context: normalized contract text
    - question: normalized clause category question
    - answer_text: normalized answer span text
    - start_char: answer start index in context
    - end_char: answer end index in context
    """
    records = []
    pair_counter = 0
    doc_counter = 0
    pattern = r'related to "(.+?)"'

    for contract in data["data"]:
        doc_name = contract.get("title","Unknown")
        doc_counter += 1
        doc_id = f"DOC_{doc_counter:04d}"

        for para in contract["paragraphs"]:
            context = para["context"]

            for qa in para["qas"]:
                match = re.search(pattern, qa['question'])
                if match:
                    clause_types = match.group(1)
                else:
                    clause_types = qa.get['question']
                qa_id = qa.get("id")

                question = qa.get("question", "")
                is_impossible = qa.get("is_impossible", False)

                # Handle multiple answers → expand
                if qa.get("answers"):
                    for ans in qa["answers"]:
                        #answer_text = normalize_text(ans.get("text", ""))
                        answer_text = ans.get("text", "")
                        start_char = ans.get("answer_start", -1)
                        end_char = start_char + len(answer_text) if start_char != -1 else -1

                        pair_counter += 1
                        records.append({
                            "id": f"PAIR_{pair_counter:08d}",
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "qa_id": qa_id,
                            "category_name": clause_types,
                            "context": context,
                            "question": question,
                            "answer_text": answer_text,
                            "start_char": start_char,
                            "end_char": end_char,
                            "is_impossible": is_impossible
                        })
                else:
                    # Unanswerable case
                    pair_counter += 1
                    records.append({
                        "id": f"PAIR_{pair_counter:08d}",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "qa_id": qa_id,
                        "category_name": clause_types,
                        "context": context,
                        "question": question,
                        "answer_text": "",
                        "start_char": -1,
                        "end_char": -1,
                        "is_impossible": True
                    })

    return pd.DataFrame(records)

