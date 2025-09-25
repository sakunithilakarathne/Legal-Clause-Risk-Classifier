from pathlib import Path
from config import *
import pandas as pd
from src.legal_clause_classifier.preprocessing.preprocessing import *
from src.legal_clause_classifier.training.train_classical_ml import train_logistic_regression_model
from src.legal_clause_classifier.testing.test_classical_ml import test_logistic_ml_model
from src.legal_clause_classifier.training.train_tfidf_ann import train_tfidf_ann_model
from src.legal_clause_classifier.testing.test_tfidf_ann import test_tfidf_ann_model
from src.legal_clause_classifier.training.train_adv_ann import train_advanced_ann_model
from src.legal_clause_classifier.testing.test_adv_ann import test_advanced_ann_model
from src.legal_clause_classifier.training.train_lstm import train_lstm_model
from src.legal_clause_classifier.testing.test_lstm import test_lstm_model
from src.legal_clause_classifier.training.train_legalbert import train_legalbert_model
from src.legal_clause_classifier.testing.test_legalbert import test_legalbert_model


def main():
    
    # raw_data = load_dataset(RAW_DATASET_PATH)
    # preprocessing_pipeline(raw_data)

    train_logistic_regression_model()
    test_logistic_ml_model()
    
    # train_tfidf_ann_model()
    # test_tfidf_ann_model()

    # train_advanced_ann_model()
    # test_advanced_ann_model()

    # train_lstm_model()
    # test_lstm_model()

    # train_legalbert_model()
    # test_legalbert_model()



if __name__ == "__main__":
    main()

    
    