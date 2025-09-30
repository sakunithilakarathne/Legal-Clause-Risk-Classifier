import os
from typing import Tuple, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# attempt to import wandb only when needed (optional)
try:
    import wandb
except ImportError:
    wandb = None

def load_tokenizer(tokenizer_name: str):
    """Load a HF tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer_name)

def load_model_from_local(model_dir: str, device: str = 'cpu'):
    """Try to load an AutoModelForSequenceClassification from a local directory.
    Supports safetensors (will try a normal load first and fallback to use_safetensors).
    Returns model and resolved model_dir path.
    """
    if not model_dir or not os.path.exists(model_dir):
        raise FileNotFoundError(f"Local model path not found: {model_dir}")
    # try default load
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return model, model_dir
    except Exception as e:
        # try with safetensors flag (some HF versions support use_safetensors param)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
            model.to(device)
            model.eval()
            return model, model_dir
        except Exception as e2:
            raise RuntimeError(f"Failed to load model from {model_dir}: {e} | {e2}")
        

def download_and_load_wandb(artifact_str: str, device: str = 'cpu', cache_root: str = './artifacts') -> Tuple[object, str]:
    if wandb is None:
        raise RuntimeError('wandb is not installed or could not be imported')
    api = wandb.Api()
    artifact = api.artifact(artifact_str)
    local_path = artifact.download(root=cache_root)
    return load_model_from_local(local_path, device=device)

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def chunk_text_by_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Chunk text by tokenizer token ids (returns decoded chunk strings)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

_SOFTMAX = torch.nn.Softmax(dim=-1)


def predict_proba(texts: List[str], model, tokenizer, device: str = 'cpu', max_length: int = 512) -> np.ndarray:
    """Return probabilities for a list of texts."""
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for txt in texts:
            enc = tokenizer(txt, truncation=True, padding='longest', max_length=max_length, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = _SOFTMAX(logits).cpu().numpy()[0]
            all_probs.append(probs)
    return np.vstack(all_probs)

def predict_aggregate(text: str, model, tokenizer, device='cpu', max_length=512, strategy='truncate'):
    """Aggregate predictions for long text using 'truncate' or 'chunk'."""
    if strategy == 'truncate':
        probs = predict_proba([text], model, tokenizer, device=device, max_length=max_length)[0]
        return probs, [text]
    elif strategy == 'chunk':
        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_length)
        chunk_probs = predict_proba(chunks, model, tokenizer, device=device, max_length=max_length)
        avg = chunk_probs.mean(axis=0)
        return avg, chunks
    else:
        raise ValueError('Unknown strategy')