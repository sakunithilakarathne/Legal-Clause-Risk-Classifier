import os
import streamlit as st
import numpy as np

from app.app_config import (
    HP_TUNED_OS_MODEL_PATH,
    LABEL_LIST_PATH,
    WANDB_ARTIFACT,
    DEFAULT_TOKENIZER,
    DEFAULT_MAX_TOKENS,
    DEVICE,
    IG_STEPS,
)
from app.model_utils import load_tokenizer, load_model_from_local, download_and_load_wandb, predict_aggregate, count_tokens
from app.explainability import compute_topk_attributions, generate_colored_html
from app.openai_utils import load_labels, classify_clause_gpt
from app.error_utils import check_clause

st.set_page_config(page_title='Legal Clause Classifier (BERT + Captum)', layout='wide')
st.title('Legal Clause Classifier — BERT + Captum explainability')

# Sidebar
with st.sidebar:
    st.header('Model source')
    use_local = st.checkbox('Use local model path', value=True)
    local_model_path = st.text_input('Local model dir', value=HP_TUNED_OS_MODEL_PATH)
    use_wandb = st.checkbox('Download model from Weights & Biases (fallback)', value=False)
    wandb_artifact = st.text_input('WandB artifact', value=WANDB_ARTIFACT)

    tokenizer_name = st.text_input('Tokenizer (HF)', value=DEFAULT_TOKENIZER)
    max_tokens = st.slider('BERT max tokens', 64, 1024, DEFAULT_MAX_TOKENS, step=32)
    strategy = st.selectbox('Long input strategy', ['truncate', 'chunk'])
    st.write('Device detected:', DEVICE)

# Load tokenizer and model (cached as resources)
@st.cache_resource
def init(tokenizer_name, use_local_flag, local_path, use_wandb_flag, wandb_artifact):
    tokenizer = load_tokenizer(tokenizer_name)
    model = None
    model_dir = None
    # prefer local
    if use_local_flag and local_path:
        try:
            model, model_dir = load_model_from_local(local_path, device=DEVICE)
        except Exception as e:
            st.warning(f'Failed to load local model: {e}')
            model = None
    if model is None and use_wandb_flag and wandb_artifact:
        try:
            model, model_dir = download_and_load_wandb(wandb_artifact, device=DEVICE)
        except Exception as e:
            st.warning(f'Failed to download/load wandb artifact: {e}')
            model = None
    return tokenizer, model, model_dir

with st.spinner('Loading tokenizer and model...'):
    tokenizer, model, model_dir = init(tokenizer_name, use_local, local_model_path, use_wandb, wandb_artifact)

if model is None:
    st.error('No model loaded. Please point to a local model directory (with config.json and model.safetensors) or enable W&B artifact download.')

# Load label list
try:
    labels = load_labels(LABEL_LIST_PATH)
except Exception as e:
    st.error(f'Failed to load label list from {LABEL_LIST_PATH}: {e}')
    labels = []

# Main input
clause = st.text_area('Paste legal clause here', height=180)

col1, col2 = st.columns([2, 1])
with col2:
    run_openai = st.checkbox('Also call OpenAI for classification+explanation (optional)')
    run_button = st.button('Classify & Explain')

if run_button:
    if not clause or clause.strip() == '':
        st.warning('Please paste a clause.')
    else:
        preproc = clause.strip()
        ok, msg = check_clause(preproc, min_length=20)
        if not ok:
            st.error(msg)
            st.stop()

        tok_count = count_tokens(preproc, tokenizer)
        st.info(f'Tokens (with special tokens): {tok_count} — BERT max {max_tokens}')

        probs, used_chunks = predict_aggregate(preproc, model, tokenizer, device=DEVICE, max_length=max_tokens, strategy=strategy)
        num_labels = probs.shape[0]

        # Map indices to labels using the provided label list
        if labels and len(labels) >= num_labels:
            id2label = {i: labels[i] for i in range(num_labels)}
        else:
            # fallback: try model.config.id2label
            if hasattr(model, 'config') and getattr(model.config, 'id2label', None):
                id2label = {int(k): v for k, v in model.config.id2label.items()}
            else:
                id2label = {i: f'label_{i}' for i in range(num_labels)}

        # pick top-3
        topk = 3
        topk_idx = np.argsort(probs)[::-1][:topk]
        st.subheader('BERT predictions (top-3)')
        for idx in topk_idx:
            st.write(f"**{id2label.get(int(idx), idx)}** — {probs[int(idx)]:.4f}")
            st.progress(int(probs[int(idx)] * 100))

        # Captum explainability (top-3 only)
        st.subheader('Token importances (Captum Integrated Gradients)')
        with st.spinner('Computing attributions...'):
            token_list, attrs_by_class = compute_topk_attributions(preproc, model, tokenizer, topk_idx.tolist(), device=DEVICE, n_steps=IG_STEPS)
        class_names = {int(i): id2label.get(int(i), str(i)) for i in topk_idx}
        html = generate_colored_html(token_list, attrs_by_class, class_names)
        st.components.v1.html(html, height=320, scrolling=True)

        # Optionally call OpenAI
        if run_openai:
            if not os.getenv('OPENAI_API_KEY'):
                st.error('OPENAI_API_KEY is not set in the environment. Set it to use OpenAI classification.')
            else:
                try:
                    gpt_res = classify_clause_gpt(preproc, categories=labels)
                    st.subheader('OpenAI classification (3 predictions + explanation)')
                    st.json(gpt_res)
                except Exception as e:
                    st.error(f'OpenAI call failed: {e}')

        # chunk info
        if strategy == 'chunk' and len(used_chunks) > 1:
            st.info(f'Input was chunked into {len(used_chunks)} pieces for aggregation.')

        st.success('Done')
