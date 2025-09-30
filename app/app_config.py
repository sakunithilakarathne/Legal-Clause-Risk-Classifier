import os

ARTIFACTS_DIR = "./artifacts"

HP_TUNED_OS_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "os_hp_tuned_model_outputs")
LABEL_LIST_PATH = os.path.join(ARTIFACTS_DIR, "label_list.json")


# WandB artifact string (optional; you can leave or change)
WANDB_ARTIFACT = 'scsthilakarathne-nibm/legal-clause-classifier/legal-bert-v2:v8'


# Default tokenizer name
DEFAULT_TOKENIZER = os.environ.get('TOKENIZER_NAME', 'nlpaueb/legal-bert-base-uncased')

# Captum IG steps (tweak for speed/quality)
IG_STEPS = 50

# BERT max tokens default
DEFAULT_MAX_TOKENS = 512

# Device fallback
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'