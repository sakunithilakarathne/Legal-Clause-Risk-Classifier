import types
import pytest
from app import openai_utils

def test_safe_extract_json_basic():
    s = '{"predictions": [], "explanation": "ok"}'
    obj = openai_utils.safe_extract_json(s)
    assert isinstance(obj, dict)
    assert obj["explanation"] == "ok"

def test_classify_clause_gpt_monkeypatched(monkeypatch):
    # create fake response object with .choices[0].message.content
    fake_json = '{"predictions":[{"rank":1,"category":"Confidentiality","confidence":0.9},{"rank":2,"category":"X","confidence":0.05},{"rank":3,"category":"Y","confidence":0.05}],"explanation":"explain"}'
    class FakeResponse:
        class Choice:
            def __init__(self, content):
                self.message = types.SimpleNamespace(content=content)
        def __init__(self, content):
            self.choices = [self.Choice(content)]
    def fake_create(*args, **kwargs):
        return FakeResponse(fake_json)
    # monkeypatch the client method
    monkeypatch.setattr(openai_utils.client.chat.completions, 'create', fake_create)
    cats = ["Confidentiality", "X", "Y"]
    res = openai_utils.classify_clause_gpt("Some clause", cats, model="gpt-test")
    assert "predictions" in res and "explanation" in res