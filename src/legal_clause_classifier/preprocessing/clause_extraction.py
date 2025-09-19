import re
import random
import spacy
import pandas as pd
import logging
from src.legal_clause_classifier.preprocessing.data_cleaning import *
from src.legal_clause_classifier.utils.logger import get_logger

nlp = spacy.load("en_core_web_sm")


# Configure logging
logger = get_logger("preprocessing", "preprocessing.log")



# Detecting Paragraph beginnings
def detect_paragraphs(text: str):
    logger.info("Detecting paragraph boundaries...")

    """
    Detect paragraph boundaries in the document.
    Each paragraph is treated as a potential clause boundary.
    """
    paragraphs = []
    # Find all double-newline-separated segments
    for match in re.finditer(r"(?:^|\n\n)([^\n].+?)(?=\n\n|$)", text, flags=re.DOTALL):
        para_text = match.group(1).strip()
        if len(para_text) < 4:  # skip tiny fragments
            continue
        start = match.start(1)
        paragraphs.append((start, para_text))

    logger.info(f"Detected {len(paragraphs)} paragraphs.")

    return paragraphs


# Expanding clauses to nearest paragraphs
def snap_to_clause(context, start_char, end_char, paragraphs):

    logger.debug(f"Snapping to clause for start_char={start_char}, end_char={end_char}")

    # Find nearest paragraph above and below
    above, below = None, None
    for (offset, para) in paragraphs:
        if offset <= start_char:
            above = (offset, para)
        elif offset > start_char and below is None:
            below = (offset, para)
            break

    clause_start = above[0] if above else 0
    clause_end = below[0] if below else len(context)

    clause_text = context[clause_start:clause_end].strip()

    # Fallback: sentence splitter if clause is too short
    if len(clause_text.split()) < 12:

        logger.debug("Clause too short, using sentence-level fallback.")

        doc = nlp(context[max(0, start_char-256): start_char+512])
        sents = [s for s in doc.sents]
        for s in sents:
            if s.start_char <= start_char - max(0, start_char-256) <= s.end_char:
                clause_text = s.text
                clause_start = start_char
                clause_end = end_char
                break

    heading = clause_text.split("\n")[0][:100]  
    return clause_text, clause_start, clause_end, heading



# Merge Overlapping clauses
def merge_clauses(clauses):

    logger.info("Merging overlapping clauses...")

    merged = []
    for row in sorted(clauses, key=lambda x: (x["doc_id"], x["start_offset"])):
        if merged and row["doc_id"] == merged[-1]["doc_id"]:
            prev = merged[-1]
            if row["start_offset"] <= prev["end_offset"]:
                # overlap → merge categories + extend span
                prev["end_offset"] = max(prev["end_offset"], row["end_offset"])
                prev["categories_set"].update(row["categories_set"])
                prev["clause_text"] = row["clause_text"]  # refresh text from last span
                continue
        merged.append(row)

    logger.info(f"Merged down to {len(merged)} clauses.")

    return merged




# Generate negative samples
def sample_negatives(context, paragraphs, positive_offsets, doc_id, n_samples=5):

    logger.info(f"Generating up to {n_samples} negative samples for doc_id={doc_id}")

    negatives = []
    for i, (start, para) in enumerate(paragraphs):
        end = paragraphs[i+1][0] if i+1 < len(paragraphs) else len(context)
        if not any(start <= pos <= end for pos in positive_offsets):
            clause_text = context[start:end].strip()
            if len(clause_text.split()) >= 12:
                negatives.append({
                    "id": f"NEG_{doc_id}_{i:05d}",
                    "doc_id": doc_id,
                    "clause_text": clause_text,
                    "start_offset": start,
                    "end_offset": end,  
                    "heading": para.split("\n")[0][:100],  # first line as heading
                    "categories_set": {"no-risk"}
                })
    random.shuffle(negatives)

    logger.info(f"Sampled {len(negatives[:n_samples])} negatives.")

    return negatives[:n_samples]



# Main Clause Extraction 
def extract_clauses(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Starting clause extraction process...")

    results = []
    for doc_id, group in df.groupby("doc_id"):
        context = group["context"].iloc[0]
        headings = detect_paragraphs(context)
        positive_offsets = []

        for _, row in group.iterrows():
            if row["start_char"] == -1:  # skip impossible
                continue

            clause_text, start_offset, end_offset, heading = snap_to_clause(
                context, row["start_char"], row["end_char"], headings
            )

            # Normalize AFTER snap_to_clause
            clause_text = normalize_text(clause_text)

            if len(clause_text.split()) < 12:
                continue

            results.append({
                "id": row["id"],
                "doc_id": doc_id,
                "clause_text": clause_text,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "heading": heading,
                "categories_set": {row["category_name"]}
            })
            positive_offsets.append(start_offset)

        logger.info(f"Extracted {len(positive_offsets)} positive clauses from {doc_id}")

        # Merge overlaps 
        results = merge_clauses(results)

        # Add negatives 
        negs = sample_negatives(context, headings, positive_offsets, doc_id)
        for n in negs:
            n["clause_text"] = normalize_text(n["clause_text"])
        results.extend(negs)
        
    # Remove "no-risk" if clause has a real category
    for r in results:
        if "no-risk" in r["categories_set"] and len(r["categories_set"]) > 1:
            r["categories_set"].discard("no-risk")

    # Deduplicate on normalized text
    results = deduplicate_clauses(results)
    logger.info(f"Final clause count after deduplication: {len(results)}")

    return pd.DataFrame(results)
