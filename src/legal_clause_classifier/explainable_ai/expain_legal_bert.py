import os
import torch
import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, visualization
from datasets import load_from_disk
from config import (TOKENIZED_VAL, OUTPUT_DIR, LABEL_LIST_PATH)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "nlpaueb/legal-bert-base-uncased"



def ds_to_tensors(dataset, max_samples=None):
    input_ids = torch.tensor(dataset['input_ids'][:max_samples], dtype=torch.long).to(DEVICE)  # ✅ force long
    attention_mask = torch.tensor(dataset['attention_mask'][:max_samples], dtype=torch.long).to(DEVICE)  # ✅ force long
    return input_ids, attention_mask


def captum_explanation(model, input_ids, attention_mask, idx_to_label):
    def forward_func(input_ids, attention_mask):
        
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits)

    ig = IntegratedGradients(forward_func)
    attributions_per_class = {}

    for class_idx, label in idx_to_label.items():
        class_attrs = []
        for i in range(input_ids.shape[0]):
            single_input = input_ids[i].unsqueeze(0)
            single_mask = attention_mask[i].unsqueeze(0)

            attr, delta = ig.attribute(
                inputs=single_input,
                additional_forward_args=(single_mask,),
                target=class_idx,
                return_convergence_delta=True
            )

            class_attrs.append(attr.detach().cpu().numpy())
            torch.cuda.empty_cache()   # free memory each iteration

        attributions_per_class[label] = class_attrs
        print(f"[Captum] Explained class: {label}")

    return attributions_per_class


def shap_explanation(model, tokenizer, input_ids, attention_mask, idx_to_label):
    def f(batch_ids):
        batch_ids = torch.tensor(batch_ids, dtype=torch.long).to(DEVICE)
        batch_mask = (batch_ids != tokenizer.pad_token_id).long().to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
            return torch.sigmoid(logits).cpu().numpy()
    
   
    explainer = shap.KernelExplainer(f, input_ids[:5].cpu().numpy())
    shap_values = explainer.shap_values(input_ids[:5].cpu().numpy(), nsamples=20)
    print("[SHAP] Explanation completed")
    
    shap_per_class = {label: shap_val for label, shap_val in zip(idx_to_label.values(), shap_values)}
    return shap_per_class




def visualize_attributions(input_ids, tokenizer, attributions, class_name, output_dir):
    tokens = [tokenizer.decode([id]) for id in input_ids[0].cpu().numpy()]
    token_attrs = attributions[0].sum(axis=-1)
    plt.figure(figsize=(15,3))
    plt.bar(range(len(tokens)), token_attrs)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(f"Token importances - {class_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name}_captum.png"))
    plt.close()
    

    
    
def run_explainable_ai_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init(
        project="legal-clause-classifier", 
        name="load-artifact-for-explanation")
    
    artifact = wandb.run.use_artifact('scsthilakarathne-nibm/legal-clause-classifier/legal-bert-v2:v6', type='model')
    artifact_dir = artifact.download()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, problem_type="multi_label_classification")
    model.to(DEVICE)
    model.eval()

    val_ds = load_from_disk(TOKENIZED_VAL)

    input_ids, attention_mask = ds_to_tensors(val_ds, max_samples=5)  # limit for speed

    # Load label list from artifact
    with open(os.path.join(LABEL_LIST_PATH), "r") as f:
        label_list = json.load(f)
    idx_to_label = {i: label for i, label in enumerate(label_list)}

    # Captum explanations
    captum_attrs = captum_explanation(model, input_ids, attention_mask, idx_to_label)
    torch.save(captum_attrs, os.path.join(OUTPUT_DIR, "captum_attrs.pt"))

    for i, label in list(idx_to_label.items())[:5]:
        visualize_attributions(input_ids, tokenizer, captum_attrs[label], label, OUTPUT_DIR)

    # SHAP explanations
    shap_attrs = shap_explanation(model, tokenizer, input_ids, attention_mask, idx_to_label)
    torch.save(shap_attrs, os.path.join(OUTPUT_DIR, "shap_attrs.pt"))

    # Upload model as next version of W&B artifact
    artifact = wandb.Artifact(
        name="explainable_ai", 
        type="model",
        description="SHAP and Captum explainability on Legal BERT"
    )

    # Add saved model files to the artifact
    artifact.add_file(os.path.join(OUTPUT_DIR, "shap_attrs.pt"))
    artifact.add_file(os.path.join(OUTPUT_DIR, "captum_attrs.pt"))

    # Log the new version of the model
    wandb.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()

