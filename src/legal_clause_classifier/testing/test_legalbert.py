# import os
# import numpy as np
# import torch
# from datasets import load_from_disk
# from transformers import Trainer
# from sklearn.metrics import f1_score, average_precision_score

# from src.legal_clause_classifier.models.legal_bert import LegalBERTClassifier
# from config import (Y_TEST_PATH, TOKENIZED_TEST, LEGAL_BERT_MODEL_PATH )



# def test_legalbert_model():

#     # ==== Load dataset ====
#     test_dataset = load_from_disk(TOKENIZED_TEST)
#     y_test = np.load(Y_TEST_PATH)
#     test_dataset = test_dataset.add_column("labels", y_test.tolist())

#     num_labels = y_test.shape[1]
#     model = LegalBERTClassifier(num_labels=num_labels)
#     model.load_state_dict(torch.load(os.path.join(LEGAL_BERT_MODEL_PATH, "pytorch_model.bin"), map_location="cpu"))

#     # ==== Metrics ====
#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         preds = torch.sigmoid(torch.tensor(logits)).numpy()
#         labels = np.array(labels)

#         micro_f1 = f1_score(labels, preds > 0.5, average="micro")
#         macro_f1 = f1_score(labels, preds > 0.5, average="macro")
#         pr_auc = average_precision_score(labels, preds, average="micro")

#         return {
#             "micro_f1": micro_f1,
#             "macro_f1": macro_f1,
#             "pr_auc": pr_auc
#         }

#     # ==== Trainer for eval only ====
#     trainer = Trainer(
#         model=model,
#         tokenizer=None,
#         compute_metrics=compute_metrics
#     )

#     # ==== Evaluate ====
#     results = trainer.evaluate(test_dataset)
#     print("Test Results:", results)
