import numpy as np
import torch
from captum.attr import IntegratedGradients

def compute_attributions_captum(text: str, model, tokenizer, target: int, device: str = 'cpu', n_steps: int = 50):
    """Return (merged_tokens, attributions_array) for a target class index.
    Merges '##' subword tokens into whole words and discards special tokens.
    """
    model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors='pt', truncation=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)
    inputs_embeds.requires_grad_()

    def forward_embeds(embs, attention_mask):
        outputs = model(inputs_embeds=embs, attention_mask=attention_mask)
        return outputs.logits
   
    ig = IntegratedGradients(forward_embeds)
    baseline = torch.zeros_like(inputs_embeds).to(device)

    attributions, delta = ig.attribute(
        inputs=inputs_embeds,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=target,
        return_convergence_delta=True,
        n_steps=n_steps
    )
        
    attributions_sum = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy() # (seq_len,)
    token_ids = input_ids.squeeze(0).detach().cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    merged_tokens = []
    merged_attrs = []
    for tok, attr in zip(tokens, attributions_sum):
        if tok in tokenizer.all_special_tokens:
            continue
        if tok.startswith('##') and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + tok[2:]
            merged_attrs[-1] += float(attr)
        else:
            merged_tokens.append(tok)
            merged_attrs.append(float(attr))
    return merged_tokens, np.array(merged_attrs)


def compute_topk_attributions(text: str, model, tokenizer, topk_indices: list, device='cpu', n_steps=50):
    """Compute attributions for each index in topk_indices. Returns tokens and dict idx->attrs"""
    token_list = None
    attrs_by_class = {}
    for idx in topk_indices:
        tokens, attrs = compute_attributions_captum(text, model, tokenizer, target=int(idx), device=device, n_steps=n_steps)
        if token_list is None:
            token_list = tokens
        else:
        # If mismatch, attempt to align by padding (best-effort)
            if len(tokens) != len(token_list):
                if len(tokens) < len(token_list):
                    tokens = tokens + [''] * (len(token_list) - len(tokens))
                    attrs = np.concatenate([attrs, np.zeros(len(token_list) - len(attrs))])
                else:
                    token_list = tokens
        attrs_by_class[int(idx)] = attrs
    return token_list or [], attrs_by_class


def generate_colored_html(tokens: list, class_attrs: dict, class_names: dict):
    """Create an HTML snippet coloring each token by the class with the largest absolute attribution.
    class_attrs: dict class_idx -> 1D numpy array (len(tokens))
    class_names: class_idx -> string label
    """
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    # map class idxs to colors deterministically
    class_idx_list = list(class_attrs.keys())
    class_colors = {c: base_colors[i % len(base_colors)] for i, c in enumerate(class_idx_list)}
    # compute normalization
    all_vals = np.vstack([np.abs(v) for v in class_attrs.values()]) if class_attrs else np.zeros((1, len(tokens)))
    global_max = float(all_vals.max()) if all_vals.size else 1.0
    if global_max == 0:
        global_max = 1.0
    legend_html = ''.join([
        f"<span style='display:inline-block;padding:4px 8px;margin-right:6px;border-radius:6px;background:{class_colors[c]};color:white;font-size:12px;'>{class_names.get(c, str(c))}</span>"
        for c in class_idx_list
    ])
    html_parts = [f"<div style='margin-bottom:8px'>{legend_html}</div>"]

    for i, tok in enumerate(tokens):
        best_class = None
        best_val = 0.0
        for c_idx, arr in class_attrs.items():
            v = float(arr[i]) if i < len(arr) else 0.0
            if abs(v) > abs(best_val):
                best_val = v
                best_class = c_idx
        alpha = min(0.95, max(0.05, abs(best_val) / global_max)) if global_max > 0 else 0.05
        color_hex = class_colors.get(best_class, '#cccccc')
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        title = f"class: {class_names.get(best_class, str(best_class))} | raw_attr={best_val:.4f}"
        spacer = ' '
        if (tok and tok[0] in '.,;:?!'):
            spacer = ''
        safe_tok = tok.replace('<', '&lt;').replace('>', '&gt;')
        span = f"{spacer}<span title='{title}' style='background: rgba({r}, {g}, {b}, {alpha}); padding:2px 4px; border-radius:4px;'>{safe_tok}</span>"
        html_parts.append(span)
    
    html_combined = ''.join(html_parts)
    html = f"<div style='line-height:1.6; font-family: Inter, Arial; font-size:14px'>{html_combined}</div>"
    return html