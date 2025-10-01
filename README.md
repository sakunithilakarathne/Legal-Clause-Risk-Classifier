# ⚖️ Multi Label Legal Clause Classifier + Summarizer
Legal Clause Risk Classifier and Rewrite Suggestions Project trained on CUAD dataset


## 📖 Project Overview

This project builds an **automated system** to classify legal contract clauses into multiple risk categories (e.g. “Termination”, “License Grant”, “No-Risk”, etc.) and provide **explainable insights** (token importances) for users (e.g. lawyers, contract reviewers). In addition to a BERT-based classifier, the system integrates a GPT-based classifier + natural-language explanation side-by-side for comparison.

Key goals:
- Multi-label classification of clauses with up to ~40+ categories.
- Handling severe class imbalance (rare clause categories) via oversampling, focal loss, and threshold tuning.
- Interpretability: token-level explanations using Captum and SHAP.
- Web interface with **Streamlit** for clause input, classification display, and explanation visualization.
- Integration with OpenAI GPT to produce alternative predictions and human-readable explanations.

---

## ✨ Features

- **BERT-based clause classification:** returns top-k predicted categories with probabilities.
- **Explainable AI:** token-level importance overlays via Captum (Integrated Gradients) and SHAP.
- **Threshold tuning per class:** optimal cutoffs tuned on validation to balance precision/recall.
- **Hyperparameter optimization:** using Optuna, pruning, and gradient accumulation for best model.
- **Handling class imbalance:** oversampling of rare classes, combined with focal loss to mitigate bias.
- **GPT-based classification & explanation:** side-by-side OpenAI predictions for comparison.
- **Streamlit web app:** user-friendly UI for input, classification, and explanations.
- **W&B integration:** tracking experiments, model artifacts, metrics, and explainability outputs.

---

## 🧪 Streamlit Usage

- Enter a legal clause in text box.
- Submit → Shows:
    - Top 3 BERT-predicted categories + probabilities
    - Color-coded token importances (red/blue) for the top predicted class
    - GPT’s prediction and explanation
    - Users can compare BERT vs GPT and examine what words influenced BERT’s decision.
---

## 🧭 Tech Stack

| Layer | Technology / Library |
|---|---|
| Model & Training | Python, PyTorch, Hugging Face Transformers, Datasets |
| Imbalance Handling | Oversampling, Focal Loss, Weighted Sampler, Optuna tuning |
| Explainability | Captum (Integrated Gradients) |
| Web App | Streamlit |
| GPT Integration | OpenAI API |
| Experiment Tracking | Weights & Biases (W&B) |

---
## 🛠 Getting Started (Installation & Usage)

### Prerequisites

- Python 3.8+  
- GPU recommended (Google Colab)  
- OpenAI API key (for GPT integration)  
- W&B account (optional but recommended)

### Setup (Conda)

```bash
conda env create -f environment.yml
conda activate legal-clause-classifier
pip install -r requirements.txt

### Train / Fine-Tune BERT Model
python src/legal_clause_classifier/training/train_legalbert.py

### Train oversampled + focal loss model
python src/legal_clause_classifier/training/train_legalbert_os_focal.py

### Hyperparameter Tuning (Optuna)
python src/legal_clause_classifier/optimization/hp_tuning.py

### Threshold Optimization
python src/legal_clause_classifier/explainable_ai/threshold_optimize.py

### Running the Web App
streamlit run run_app.py