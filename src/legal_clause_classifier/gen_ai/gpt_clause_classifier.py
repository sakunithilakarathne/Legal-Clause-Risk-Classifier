import os
import openai
import json
import re
from config import LABEL_LIST_PATH
# -------------------------
# SETUP
# -------------------------
# Ensure your OpenAI API key is set:
# export OPENAI_API_KEY="your_api_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------
# LOAD LABEL LIST
# -------------------------
def load_labels(label_file: str = LABEL_LIST_PATH):
    """Load categories from JSON file."""
    with open(label_file, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# PROMPT TEMPLATES
# -------------------------
SYSTEM_PROMPT = """You are a legal classification assistant.
You will classify a legal clause into CUAD categories.
Always return JSON only, with exactly three ranked predictions
and one plain-English explanation. Each prediction must include:
- rank (1 to 3)
- category (must be from the provided category list)
- confidence (0 to 1)
"""

FEW_SHOT_EXAMPLES = """
Example:

Clause: "The contractor shall not disclose any confidential information obtained during performance."

JSON:
{
  "predictions": [
    {"rank": 1, "category": "Confidentiality", "confidence": 0.95},
    {"rank": 2, "category": "Assignment", "confidence": 0.10},
    {"rank": 3, "category": "Limitation of Liability", "confidence": 0.05}
  ],
  "explanation": "This clause says that private information cannot be shared, which is a confidentiality obligation."
}
"""

# -------------------------
# HELPERS
# -------------------------
def safe_extract_json(text: str):
    """Try to extract valid JSON from the model's response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                s = match.group(0).replace("'", '"')
                return json.loads(s)
        raise ValueError(f"Could not parse JSON from response: {text[:200]}")

def classify_clause_gpt(clause: str, categories: list, model: str = "gpt-4o-mini"):
    """
    Classify a single legal clause using GPT.
    Returns dict with predictions and explanation.
    """
    user_prompt = f"""
{FEW_SHOT_EXAMPLES}

CATEGORIES:
{json.dumps(categories)}

Clause: "{clause}"

Return JSON only.
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=400,
        n=1
    )
    text = response["choices"][0]["message"]["content"].strip()
    return safe_extract_json(text)

# -------------------------
# MAIN
# -------------------------
def run_gpt_clause_classifier():
    # Load label list
    categories = load_labels(LABEL_LIST_PATH)

    # Take clause input
    clause = input("\nEnter a legal clause: ").strip()

    # Classify
    result = classify_clause_gpt(clause, categories)

    # Display results
    print("\n=== Clause ===")
    print(clause)
    print("\n=== Predictions ===")
    for p in result.get("predictions", []):
        print(f"Rank {p['rank']}: {p['category']} (confidence {p['confidence']})")
    print("\n=== Explanation ===")
    print(result.get("explanation", ""))

