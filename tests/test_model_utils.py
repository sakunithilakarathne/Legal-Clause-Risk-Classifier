import numpy as np
import torch
import pytest

# import the module under test (adjust path if needed)
from app import model_utils


class FakeTokenizer:
    """Minimal tokenizer-like object used for unit tests."""
    def __call__(self, txt, truncation=True, padding='longest', max_length=512, return_tensors='pt'):
        # return 1 sample with 4 token ids and attention mask
        return {'input_ids': torch.tensor([[101, 2, 3, 102]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1]])}

    def encode(self, text, add_special_tokens=True):
        return [101, 2, 3, 102]

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return "decoded chunk"


class MockModel:
    """Mock model that returns fixed logits when call() is invoked."""
    def __init__(self, logits):
        # logits should be a list or array of length = num_labels
        self._logits = torch.tensor([logits], dtype=torch.float)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # returns an object with a logits attribute (same shape as HF output)
        class Out:
            pass
        out = Out()
        out.logits = self._logits
        return out


def test_predict_proba_returns_probs():
    tok = FakeTokenizer()
    model = MockModel([0.0, 10.0])  # large difference -> softmax ~ [0.0, 1.0]
    probs = model_utils.predict_proba(["hello world"], model, tok, device='cpu', max_length=32)
    assert probs.shape == (1, 2)
    # second class should be ~1.0
    assert pytest.approx(probs[0, 1], rel=1e-3) == 1.0


def test_predict_aggregate_chunking(monkeypatch):
    # monkeypatch chunk_text_by_tokens to return three chunks
    monkeypatch.setattr(model_utils, 'chunk_text_by_tokens', lambda text, tokenizer, max_tokens, overlap=50: ['a', 'b', 'c'])

    # monkeypatch predict_proba to return different probabilities for each chunk
    def fake_predict_proba(texts, model, tokenizer, device='cpu', max_length=512):
        # produce 3 rows with different probs
        return np.vstack([[0.1, 0.9], [0.4, 0.6], [0.7, 0.3]])

    monkeypatch.setattr(model_utils, 'predict_proba', fake_predict_proba)

    probs, chunks = model_utils.predict_aggregate("some long text", model=None, tokenizer=None, device='cpu', max_length=10, strategy='chunk')
    # average of [0.1,0.4,0.7] = 0.4 for class0
    assert pytest.approx(probs[0], rel=1e-6) == 0.4
    assert len(chunks) == 3