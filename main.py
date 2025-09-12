from pathlib import Path
from src.legal_clause_classifier import *
from src.legal_clause_classifier.preprocessing.schema_normalization import *
from src.legal_clause_classifier.preprocessing.data_cleaning import *
from src.legal_clause_classifier.preprocessing.clause_extraction import *


if __name__ == "__main__":
    # file_path = Path(RAW_DATA_DIR/'CUAD_v1.json')
    # raw_data = load_dataset(file_path)
    # print("✅ Dataset Loaded")


    # # Normalize the json dataset
    # normalized_df = normalize_cuad_schema(raw_data)
    # print("✅ CUAD Dataset Ingested & Normalized")
    # print(normalized_df.head(3))
    # print(f"Total samples: {len(normalized_df)}")
    # normalized_df.to_csv(RAW_DATA_DIR/'cuad_normalized.csv', index=False, encoding="utf-8")
    # print("✅ Normalized CUAD Dataset saved to RAW Data Directory ")

    normalized_df = pd.read_csv(RAW_DATA_DIR/'cuad_normalized.csv')
    print("✅ CUAD Dataset read")

    clause_df = extract_clauses(normalized_df)
    print(clause_df.head(3))
    print(f"Total samples: {len(clause_df)}")
    columns = clause_df.columns
    print(f"Columns: {clause_df.columns}" )
    clause_df.to_csv(RAW_DATA_DIR/'paragraphes_clauses_set.csv', index=False, encoding="utf-8")

  

    # final_df = prepare_clauses(normalized_df)
    # final_df.to_csv('clause_category_pairs.csv')
    # print("✅ Clause Category pairs saved to RAW Data Directory ")


    # Filter and Remove Duplicates
    # filtered_df = filter_normalized_schema(normalized_df)
    # filtered_df.to_csv(RAW_DATA_DIR/'cuad_filtered.csv', index=False, encoding="utf-8")
    # print(f"New Total samples: {len(filtered_df)}")
    # print("✅ Filtered Dataset saved")

    # unique_df = remove_duplicates(filtered_df)
    # unique_df.to_csv(RAW_DATA_DIR/'unique_clauses.csv', index=False, encoding="utf-8")
    # print(unique_df.head(3))
    # print(f"New Total samples: {len(unique_df)}")
    # print("✅ Unique Clauses set saved")

    # # filtered_df = pd.read_csv(RAW_DATA_DIR/'filtered_cuad_normalized.csv')
    # # print(f"Total samples: {len(df)}")
    # # print("✅ CUAD Dataset Read")