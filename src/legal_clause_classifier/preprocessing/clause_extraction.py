import re
import random
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import logging

nlp = spacy.load("en_core_web_sm")



# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# # =======================
# # TEXT NORMALIZATION
# # =======================

# def normalize_text(text: str, lowercase: bool = True) -> str:
#     """
#     Normalize text by standardizing unicode characters, replacing quotes/dashes,
#     removing special characters, and collapsing whitespace.

#     Args:
#         text (str): The input string to normalize.
#         lowercase (bool): Whether to convert text to lowercase.

#     Returns:
#         str: Normalized string.
#     """
#     if not isinstance(text, str):
#         return ""
    
#     text = unicodedata.normalize("NFC", text)
#     text = text.replace("’", "'").replace("‘", "'")
#     text = text.replace("“", '"').replace("”", '"')
#     text = text.replace("–", "-").replace("—", "-")

#     # Remove unwanted characters (allow basic punctuation)
#     text = re.sub(r"[^\w\s\'\"\-\.\,\(\)\;\:?\!]", "", text)

#     # Normalize whitespace
#     text = " ".join(text.split())
#     return text.strip().lower() if lowercase else text.strip()


# # =======================
# # CLAUSE EXPANSION
# # =======================

# def expand_to_clause(context: str, start: int, end: int, max_tokens: int = 512):
#     """
#     Expands a span of text to include the surrounding clause using paragraph or sentence boundaries.

#     Args:
#         context (str): Full document text.
#         start (int): Start character index of the span.
#         end (int): End character index of the span.
#         max_tokens (int): Max word count allowed in a clause.

#     Returns:
#         tuple: (clause_text, clause_start_char, clause_end_char)
#     """
#     para_bounds = [m.start() for m in re.finditer(r"\n\n|;", context)]
#     para_bounds = [0] + para_bounds + [len(context)]

#     # Find paragraph that contains the span
#     start_para = max([b for b in para_bounds if b <= start])
#     end_para = min([b for b in para_bounds if b >= end])

#     clause = context[start_para:end_para].strip()

#     # Fallback to sentence-level span
#     if len(clause.split()) < 5 or len(clause.split()) > max_tokens:
#         doc = nlp(context)
#         for sent in doc.sents:
#             s_start, s_end = sent.start_char, sent.end_char
#             if s_start <= start and s_end >= end:
#                 clause = context[s_start:s_end].strip()
#                 start_para, end_para = s_start, s_end
#                 break

#     return clause, start_para, end_para


# # =======================
# # MERGE CLAUSES
# # =======================

# def merge_clauses(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Merge overlapping or adjacent clauses and union their category labels.

#     Args:
#         df (pd.DataFrame): DataFrame with clause spans and categories.

#     Returns:
#         pd.DataFrame: Merged clauses.
#     """
#     merged_records = []

#     for doc_id, group in df.groupby("doc_id"):
#         spans = [
#             (row.clause_start, row.clause_end, row.category_name, row.new_clause)
#             for _, row in group.iterrows()
#         ]

#         if not spans:
#             continue

#         spans.sort(key=lambda x: x[0])
#         cur_start, cur_end, cur_cats, cur_text = spans[0][0], spans[0][1], {spans[0][2]}, spans[0][3]

#         for s, e, cat, text in spans[1:]:
#             if s <= cur_end:
#                 cur_end = max(cur_end, e)
#                 cur_cats.add(cat)
#             else:
#                 merged_records.append({
#                     "doc_id": doc_id,
#                     "clause_start": cur_start,
#                     "clause_end": cur_end,
#                     "new_clause": normalize_text(cur_text),
#                     "categories": list(cur_cats)
#                 })
#                 cur_start, cur_end, cur_cats, cur_text = s, e, {cat}, text

#         # Append last
#         merged_records.append({
#             "doc_id": doc_id,
#             "clause_start": cur_start,
#             "clause_end": cur_end,
#             "new_clause": normalize_text(cur_text),
#             "categories": list(cur_cats)
#         })

#     return pd.DataFrame(merged_records)


# # =======================
# # EXTRACT NEGATIVE CLAUSES
# # =======================

# def extract_negative_clauses(context: str, positive_spans: list, max_tokens: int = 512):
#     """
#     Identify sentences that are not part of positive spans.

#     Args:
#         context (str): Full document text.
#         positive_spans (list): List of (start, end) tuples for positive clauses.

#     Returns:
#         list: List of tuples (normalized_text, start_char, end_char)
#     """
#     doc = nlp(context)
#     negatives = []

#     for sent in doc.sents:
#         s_start, s_end = sent.start_char, sent.end_char
#         overlap = any(not (s_end <= ps[0] or s_start >= ps[1]) for ps in positive_spans)
#         if not overlap:
#             text = context[s_start:s_end].strip()
#             if len(text.split()) <= max_tokens:
#                 negatives.append((normalize_text(text), s_start, s_end))

#     return negatives


# # =======================
# # MAIN PIPELINE FUNCTION
# # =======================

# def prepare_clauses(df: pd.DataFrame, max_negatives_per_doc: int = 5) -> pd.DataFrame:
#     """
#     Main pipeline to prepare and label contract clauses as positive or negative.
#     Limits negatives per document.

#     Args:
#         df (pd.DataFrame): Input DataFrame with columns: doc_id, context, start_char, end_char, is_impossible, category_name
#         max_negatives_per_doc (int): Max number of negative samples per document.

#     Returns:
#         pd.DataFrame: Labeled and merged clauses (positive and negative).
#     """
#     all_records = []

#     for doc_id, group in df.groupby("doc_id"):
#         logging.info(f"Processing document: {doc_id}")
#         context = group.iloc[0].context
#         positive_spans = []
#         positive_records = []
#         negative_records = []

#         # Positive clauses
#         for _, row in group[group.is_impossible == False].iterrows():
#             clause, c_start, c_end = expand_to_clause(context, row.start_char, row.end_char)
#             positive_spans.append((c_start, c_end))
#             positive_records.append({
#                 "doc_id": doc_id,
#                 "clause_start": c_start,
#                 "clause_end": c_end,
#                 "new_clause": normalize_text(clause),
#                 "category_name": row.category_name
#             })

#         pos_df = pd.DataFrame(positive_records)
#         if not pos_df.empty:
#             pos_df = merge_clauses(pos_df)
#             all_records.extend(pos_df.to_dict("records"))

#         # Negative clauses
#         for _, row in group[group.is_impossible == True].iterrows():
#             all_negatives = extract_negative_clauses(context, positive_spans)
            
#             # Limit to max_negatives_per_doc (randomly sampled)
#             sampled_negatives = random.sample(all_negatives, min(max_negatives_per_doc, len(all_negatives)))

#             for text, s, e in sampled_negatives:
#                 negative_records.append({
#                     "doc_id": doc_id,
#                     "clause_start": s,
#                     "clause_end": e,
#                     "new_clause": text,
#                     "categories": []
#                 })

#         all_records.extend(negative_records)

#         logging.info(f"✅ Completed: {doc_id} ({len(positive_records)} positives, {len(negative_records)} negatives)")

#     return pd.DataFrame(all_records)


def normalize_text(text: str) -> str:
    """
    Normalize legal text for consistent processing.
    - Unicode NFC normalization
    - Standardize quotes and dashes
    - Collapse multiple spaces/newlines into one
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize quotes & dashes
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
    )

    # # Remove common page headers/footers
    # # Example: "Page 3 of 12"
    # text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", text, flags=re.I)

    # # Remove standalone "Page X" or "Page: X"
    # text = re.sub(r"\bPage[:\s]*\d+\b", " ", text, flags=re.I)

    # # Remove company-like headers/footers (heuristic: all-caps lines with >3 words)
    # text = re.sub(r"^[A-Z\s]{5,}$", " ", text, flags=re.MULTILINE)


    # Collapse multiple spaces/newlines into a single space
    text = " ".join(text.split())

    # Final strip
    return text.strip()

####### USING SECTION HEADERS #####################
# ---------- Step 1: Section/Heading Detection ----------
# HEADING_PATTERNS = [
#     re.compile(r"^\s*(\d+(\.\d+)*\s+.+)$"),            # Numbered headings
#     re.compile(r"^[A-Z][A-Z\s\-/&]+$"),                 # ALL CAPS (len>4 check applied later)
#     re.compile(r"^(Section|Article)\s+\d+[A-Za-z]*", re.I)  # Section/Article
# ]

# def detect_headings(text: str):
#     headings = []
#     for match in re.finditer(r".+", text, flags=re.MULTILINE):
#         line = match.group().strip()
#         if len(line) < 4:
#             continue
#         for pat in HEADING_PATTERNS:
#             if pat.match(line):
#                 headings.append((match.start(), line))
#                 break
#     return headings

# # ---------- Step 2: Snap Spans to Nearest Clause ----------
# def snap_to_clause(context, start_char, end_char, headings):
#     # Find nearest heading above and below
#     above, below = None, None
#     for (offset, head) in headings:
#         if offset <= start_char:
#             above = (offset, head)
#         elif offset > start_char and below is None:
#             below = (offset, head)
#             break

#     clause_start = above[0] if above else max(context.rfind("\n\n", 0, start_char), 0)
#     clause_end = below[0] if below else len(context)

#     clause_text = context[clause_start:clause_end].strip()

#     # Fallback: sentence splitter if clause is too short
#     if len(clause_text.split()) < 12:
#         doc = nlp(context[max(0, start_char-256): start_char+512])
#         sents = [s for s in doc.sents]
#         for s in sents:
#             if s.start_char <= start_char - max(0, start_char-256) <= s.end_char:
#                 clause_text = s.text
#                 clause_start = start_char
#                 clause_end = end_char
#                 break

#     heading = above[1] if above else None
#     return clause_text, clause_start, clause_end, heading

# # ---------- Step 3: Merge Overlapping Spans ----------
# def merge_clauses(clauses):
#     merged = []
#     for row in sorted(clauses, key=lambda x: (x["doc_id"], x["start_offset"])):
#         if merged and row["doc_id"] == merged[-1]["doc_id"]:
#             prev = merged[-1]
#             if row["start_offset"] <= prev["end_offset"]:
#                 # overlap → merge categories + extend span
#                 prev["end_offset"] = max(prev["end_offset"], row["end_offset"])
#                 prev["categories_set"].update(row["categories_set"])
#                 prev["clause_text"] = row["clause_text"]  # refresh text from last span
#                 continue
#         merged.append(row)
#     return merged

# # ---------- Step 4: Negative Samples ----------
# def sample_negatives(context, headings, positive_offsets, doc_id, n_samples=5):
#     negatives = []
#     for i, (start, head) in enumerate(headings):
#         end = headings[i+1][0] if i+1 < len(headings) else len(context)
#         if not any(start <= pos <= end for pos in positive_offsets):
#             clause_text = context[start:end].strip()
#             if len(clause_text.split()) >= 12:
#                 negatives.append({
#                     "id": f"NEG_{doc_id}_{i:05d}",
#                     "doc_id": doc_id,
#                     "clause_text": clause_text,
#                     "start_offset": start,
#                     "end_offset": end,
#                     "heading": head,
#                     "categories_set": {"no-category"}
#                 })
#     random.shuffle(negatives)
#     return negatives[:n_samples]

# # ---------- Step 5: Quality Checks (short + dedup) ----------
# def deduplicate_clauses(clauses, sim_threshold=0.9):
#     texts = [c["clause_text"] for c in clauses]
#     vectorizer = TfidfVectorizer().fit_transform(texts)
#     sim_matrix = cosine_similarity(vectorizer)
#     keep = []
#     seen = set()
#     for i, row in enumerate(sim_matrix):
#         if i in seen: 
#             continue
#         dupes = {j for j, sim in enumerate(row) if sim > sim_threshold}
#         seen |= dupes
#         keep.append(clauses[i])
#     return keep



# # ---------- Main Clause Extraction ----------
# def extract_clauses(df: pd.DataFrame) -> pd.DataFrame:
#     results = []
#     for doc_id, group in df.groupby("doc_id"):
#         context = group["context"].iloc[0]
#         headings = detect_headings(context)
#         positive_offsets = []

#         for _, row in group.iterrows():
#             if row["start_char"] == -1:  # skip impossible
#                 continue

#             clause_text, start_offset, end_offset, heading = snap_to_clause(
#                 context, row["start_char"], row["end_char"], headings
#             )

#             # Normalize AFTER snap_to_clause
#             clause_text = normalize_text(clause_text)

#             if len(clause_text.split()) < 12:
#                 continue

#             results.append({
#                 "id": row["id"],
#                 "doc_id": doc_id,
#                 "clause_text": clause_text,
#                 "start_offset": start_offset,
#                 "end_offset": end_offset,
#                 "heading": heading,
#                 "categories_set": {row["category_name"]}
#             })
#             positive_offsets.append(start_offset)

#         # Merge overlaps (on normalized text)
#         results = merge_clauses(results)

#         # Add negatives (also normalize inside sample_negatives)
#         negs = sample_negatives(context, headings, positive_offsets, doc_id)
#         for n in negs:
#             n["clause_text"] = normalize_text(n["clause_text"])
#         results.extend(negs)

#     # Deduplicate on normalized text
#     results = deduplicate_clauses(results)

#     return pd.DataFrame(results)



###### USING PARAGRAPH HEADINGS



# ---------- Step 1: Paragraph Detection (instead of headings) ----------
def detect_paragraphs(text: str):
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
    return paragraphs


# ---------- Step 2: Snap Spans to Nearest Clause ----------
def snap_to_clause(context, start_char, end_char, paragraphs):
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
        doc = nlp(context[max(0, start_char-256): start_char+512])
        sents = [s for s in doc.sents]
        for s in sents:
            if s.start_char <= start_char - max(0, start_char-256) <= s.end_char:
                clause_text = s.text
                clause_start = start_char
                clause_end = end_char
                break

    # Instead of a "heading", we now store the first sentence of the paragraph
    heading = clause_text.split("\n")[0][:100]  # first line as pseudo-heading
    return clause_text, clause_start, clause_end, heading



# ---------- Step 3: Merge Overlapping Spans ----------
def merge_clauses(clauses):
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
    return merged




# ---------- Step 4: Negative Samples ----------
def sample_negatives(context, paragraphs, positive_offsets, doc_id, n_samples=5):
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
    return negatives[:n_samples]


# ---------- Step 5: Quality Checks (short + dedup) ----------
def deduplicate_clauses(clauses, sim_threshold=0.9):
    texts = [c["clause_text"] for c in clauses]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    sim_matrix = cosine_similarity(vectorizer)
    keep = []
    seen = set()
    for i, row in enumerate(sim_matrix):
        if i in seen: 
            continue
        dupes = {j for j, sim in enumerate(row) if sim > sim_threshold}
        seen |= dupes
        keep.append(clauses[i])
    return keep



# ---------- Main Clause Extraction ----------
def extract_clauses(df: pd.DataFrame) -> pd.DataFrame:
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

        # Merge overlaps (on normalized text)
        results = merge_clauses(results)

        # Add negatives (also normalize inside sample_negatives)
        negs = sample_negatives(context, headings, positive_offsets, doc_id)
        for n in negs:
            n["clause_text"] = normalize_text(n["clause_text"])
        results.extend(negs)
        
    # Cleanup: remove "no-risk" if clause has a real category
    for r in results:
        if "no-risk" in r["categories_set"] and len(r["categories_set"]) > 1:
            r["categories_set"].discard("no-risk")

    # Deduplicate on normalized text
    results = deduplicate_clauses(results)

    return pd.DataFrame(results)
