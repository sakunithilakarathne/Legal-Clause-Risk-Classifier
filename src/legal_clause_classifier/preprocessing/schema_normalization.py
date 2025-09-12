import json
import unicodedata
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






# def normalize_cuad_schema(data: dict) -> pd.DataFrame:
#     """
#     Normalize CUAD JSON into flat schema with stable IDs.
    
#     Columns:
#     - doc_id: unique document/contract ID
#     - qa_id: original QA ID
#     - category_id: question type index (1..41)
#     - sample_id: synthetic stable row ID
#     - context: normalized contract text
#     - question: normalized clause category question
#     - answer_text: normalized answer span text
#     - start_char: answer start index in context
#     - end_char: answer end index in context
#     - is_impossible: flag if answer not present
#     """
#     records = []
#     sample_counter = 0

#     for doc_index, contract in enumerate(data["data"]):
#         doc_id = f"CUAD_DOC_{doc_index+1:05d}"

#         for para in contract["paragraphs"]:
#             context = normalize_text(para["context"])

#             for q_index, qa in enumerate(para["qas"]):
#                 qa_id = qa.get("id", f"{doc_id}_Q{q_index+1:03d}")
#                 question = normalize_text(qa.get("question", ""))
#                 is_impossible = qa.get("is_impossible", False)

#                 # category_id = map question type (CUAD has 41 categories)
#                 category_id = q_index + 1

#                 # Handle multiple answers → expand
#                 if qa.get("answers"):
#                     for ans in qa["answers"]:
#                         answer_text = normalize_text(ans.get("text", ""))
#                         start_char = ans.get("answer_start", -1)
#                         end_char = start_char + len(answer_text) if start_char != -1 else -1

#                         sample_counter += 1
#                         records.append({
#                             "doc_id": doc_id,
#                             "qa_id": qa_id,
#                             "category_id": category_id,
#                             "sample_id": f"SAMPLE_{sample_counter:08d}",
#                             "context": context,
#                             "question": question,
#                             "answer_text": answer_text,
#                             "start_char": start_char,
#                             "end_char": end_char,
#                             "is_impossible": is_impossible
#                         })
#                 else:
#                     # Unanswerable case
#                     sample_counter += 1
#                     records.append({
#                         "doc_id": doc_id,
#                         "qa_id": qa_id,
#                         "category_id": category_id,
#                         "sample_id": f"SAMPLE_{sample_counter:08d}",
#                         "context": context,
#                         "question": question,
#                         "answer_text": "",
#                         "start_char": -1,
#                         "end_char": -1,
#                         "is_impossible": True
#                     })

#     return pd.DataFrame(records)


# if __name__ == "__main__":
#     file_path = Path(RAW_DATA_DIR/'CUAD_v1.json')
#     raw_data = load_dataset(file_path)
#     print("✅ Dataset Loaded")

    # df = normalize_cuad_schema(raw_data)

    # print("✅ CUAD Dataset Ingested & Normalized")
    # print(df.head(3))
    # print(f"Total samples: {len(df)}")

    # # Save normalized CSV
    # df.to_csv("RAW_DATA_DIR/cuad_normalized.csv", index=False, encoding="utf-8")