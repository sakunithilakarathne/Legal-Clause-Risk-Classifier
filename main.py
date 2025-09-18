from pathlib import Path
from src.legal_clause_classifier import *
from src.legal_clause_classifier.preprocessing.schema_normalization import *
from src.legal_clause_classifier.preprocessing.data_cleaning import *
from src.legal_clause_classifier.preprocessing.clause_extraction import *
from src.legal_clause_classifier.preprocessing.data_splitting import *


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


    # Train, Test split
    train_df, test_df, val_df = split_dataset()
    train_df.to_parquet("train.parquet", index=False)
    val_df.to_parquet("val.parquet", index=False)
    test_df.to_parquet("test.parquet", index=False)