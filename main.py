from pathlib import Path
from config import *
import pandas as pd
from src.legal_clause_classifier.preprocessing.preprocessing import *
from src.legal_clause_classifier.training.train_classical_ml import train_logistic_regression_model
from src.legal_clause_classifier.testing.test_classical_ml import evaluate_logistic_ml_model





def main():
    
    raw_data = load_dataset(RAW_DATASET_PATH)
    preprocessing_pipeline(raw_data)

    train_logistic_regression_model()
    evaluate_logistic_ml_model()



if __name__ == "__main__":
    main()

    
    