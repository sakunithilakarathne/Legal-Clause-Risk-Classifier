from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# app/utils/error_utils.py
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# make langdetect deterministic
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ["en"]

def check_clause(clause: str, min_length: int = 20) -> tuple[bool, str]:
    """
    Validate a clause for language and length.
    Returns (ok, message). ok=False if invalid.
    """
    clause = clause.strip()

    # length check first
    if len(clause) < min_length:
        return False, f"❌ Clause is too short (min {min_length} characters required)."

    try:
        lang = detect(clause)
        if lang not in SUPPORTED_LANGUAGES:
            return False, f"❌ Clause appears to be in '{lang}', but only English is supported."
    except LangDetectException:
        return False, "❌ Could not detect language. Please enter a longer English clause."

    return True, ""