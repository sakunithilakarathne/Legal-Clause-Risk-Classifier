from pathlib import Path
from config import *
from src.legal_clause_classifier.preprocessing.schema_normalization import *
from src.legal_clause_classifier.preprocessing.data_cleaning import *
from src.legal_clause_classifier.preprocessing.clause_extraction import *
from src.legal_clause_classifier.preprocessing.data_splitting import *
from src.legal_clause_classifier.preprocessing.preprocessing import *
from src.legal_clause_classifier.training.train_classical_ml import train_logistic_regression_model


def main():
    # raw_data = load_dataset(RAW_DATASET_PATH)
    # X_train_tfidf, X_test_tfidf, X_val_tfidf, train_tokenized, val_tokenized, test_tokenized = preprocessing_pipeline(raw_data)
    train_logistic_regression_model()
    



if __name__ == "__main__":
    main()

    
    