import os
import json
import torch
import wandb
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from config import (TOKENIZED_VAL, OUTPUT_DIR, LABEL_LIST_PATH)

from captum.attr import IntegratedGradients


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_NAME = "nlpaueb/legal-bert-small-uncased" 



def ds_to_tensors(dataset, max_samples=None):
    """Convert HuggingFace dataset to PyTorch tensors on DEVICE"""
    input_ids = torch.tensor(dataset["input_ids"][:max_samples], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor(dataset["attention_mask"][:max_samples], dtype=torch.long).to(DEVICE)
    return input_ids, attention_mask


def captum_explanation(model, input_ids, attention_mask, idx_to_label):
    """Compute token attributions per class using Captum Integrated Gradients"""
    model.eval()

    # embeddings as differentiable inputs
    embeddings = model.get_input_embeddings()(input_ids)

    def forward_func(embeds, attention_mask):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits)

    ig = IntegratedGradients(forward_func)
    attributions_per_class = {}

    for class_idx, label in idx_to_label.items():
        attr, delta = ig.attribute(
            inputs=embeddings,
            additional_forward_args=(attention_mask,),
            target=class_idx,
            return_convergence_delta=True,
            allow_unused=True
        )
        attributions_per_class[label] = attr.detach().cpu().numpy()
        print(f"[Captum] Explained class: {label}")

    return attributions_per_class



def shap_explanation(model, tokenizer, input_ids, attention_mask, idx_to_label):
    """Compute SHAP token importance using KernelExplainer"""
    def f(batch_ids):
        batch_ids = torch.tensor(batch_ids, dtype=torch.long).to(DEVICE)
        batch_mask = (batch_ids != tokenizer.pad_token_id).long().to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
            return torch.sigmoid(logits).cpu().numpy()
    
    explainer = shap.KernelExplainer(f, input_ids[:10].cpu().numpy())
    shap_values = explainer.shap_values(input_ids.cpu().numpy(), nsamples=50)
    print("[SHAP] Explanation completed")

    shap_per_class = {label: shap_val for label, shap_val in zip(idx_to_label.values(), shap_values)}
    return shap_per_class



def visualize_attributions(input_ids, tokenizer, attributions, class_name, output_dir):
    """Visualize token importance for a single class"""
    tokens = [tokenizer.decode([id]) for id in input_ids[0].cpu().numpy()]
    token_attrs = attributions[0].sum(axis=-1)  # sum over embedding dimensions
    plt.figure(figsize=(15, 3))
    plt.bar(range(len(tokens)), token_attrs)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(f"Token importances - {class_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_captum.png"))
    plt.close()



# ------------------- Pipeline -------------------
def run_explainable_ai_pipeline(max_val_samples=50, visualize_top_n=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize WandB
    wandb.init(project="legal-clause-classifier", name="explainable-ai-pipeline")

    # Load model artifact
    artifact = wandb.run.use_artifact('scsthilakarathne-nibm/legal-clause-classifier/legal-bert-v2:v8', type='model')
    artifact_dir = artifact.download()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        artifact_dir, problem_type="multi_label_classification"
    )
    model.to(DEVICE)
    model.eval()

    # Load validation dataset
    val_ds = load_from_disk(TOKENIZED_VAL)
    input_ids, attention_mask = ds_to_tensors(val_ds, max_samples=max_val_samples)

    # Load labels
    with open(LABEL_LIST_PATH, "r") as f:
        label_list = json.load(f)
    idx_to_label = {i: label for i, label in enumerate(label_list)}

    # ---------------- Captum ----------------
    captum_attrs = captum_explanation(model, input_ids, attention_mask, idx_to_label)
    torch.save(captum_attrs, os.path.join(OUTPUT_DIR, "captum_attrs.pt"))

    # Visualize top N classes
    for i, label in list(idx_to_label.items())[:visualize_top_n]:
        visualize_attributions(input_ids, tokenizer, captum_attrs[label], label, OUTPUT_DIR)

    # ---------------- SHAP ----------------
    shap_attrs = shap_explanation(model, tokenizer, input_ids, attention_mask, idx_to_label)
    torch.save(shap_attrs, os.path.join(OUTPUT_DIR, "shap_attrs.pt"))

    # ---------------- Upload artifacts ----------------
    explainable_artifact = wandb.Artifact(
        name="explainable_ai",
        type="model",
        description="Token-level explanations (SHAP + Captum) for LegalBERT"
    )
    explainable_artifact.add_file(os.path.join(OUTPUT_DIR, "captum_attrs.pt"))
    explainable_artifact.add_file(os.path.join(OUTPUT_DIR, "shap_attrs.pt"))
    wandb.log_artifact(explainable_artifact)

    wandb.finish()
    print(f"Explainable AI pipeline completed. Outputs saved in {OUTPUT_DIR}")






