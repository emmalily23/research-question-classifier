"""
RQ Chunking and Template Generation

This script reads research questions and performs chunking to identify:

1. ECs (Entity Chunks): typically noun phrases or adjectives representing key entities.
2. PCs (Process Chunks): typically verb phrases representing actions/processes.

Each detected chunk is replaced with a marker (EC1, EC2, PC1, ...) and
mappings are stored to preserve the original text.

Special handling includes:
- Preserving hyphenated words during tokenization.
- Skipping question words (what, which, who...) at the start of ECs.
- Splitting noun chunks at conjunctions (e.g., "X and Y": EC1, EC2).
- Ignoring auxiliary verbs in PC extraction.
- Correct offset calculations to ensure replacements do not shift subsequent chunks incorrectly.

Adapted from: AgOCQs_Plus https://github.com/AdeebNqo/AgOCQs_Plus

Modifications:
- Updated tokenizer to prevent splitting hyphenated words.
- Refined EC/PC extraction rules for more accurate template generation for this use case.
"""


import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import re

# Load SpaCy language model
nlp = spacy.load("en_core_web_sm")

# Update tokenizer to keep hyphenated words together
# By default, SpaCy splits on "-". We remove that behaviour
infixes = nlp.Defaults.infixes
infixes = [x for x in infixes if '-' not in x]
infix_re = compile_infix_regex(infixes)
nlp.tokenizer = Tokenizer(
    nlp.vocab,
    rules=nlp.Defaults.tokenizer_exceptions,
    prefix_search=nlp.tokenizer.prefix_search,
    suffix_search=nlp.tokenizer.suffix_search,
    infix_finditer=infix_re.finditer,
    token_match=nlp.tokenizer.token_match
)

# EC settings (Entity Chunks)

# Words that usually begin questions
QUESTION_WORDS = {"how", "which", "what", "who", "when", "where", "can", "does", "do", "did"}

# Words/phrases to reject as entity chunks
REJECTING_EC = list(QUESTION_WORDS) + [
    "type", "types", "kinds", "kind", "category", "categories",
    "difference", "differences", "extent", "i", "we", "respect",
    "there", "not", "the main types", "the possible types", "the types",
    "the difference", "the differences", "the main categories",
    "similar", "encode", "that", "our aim"
]

# Helper function to clean text
def clean_text(text):
    # Remove leading "X:" prefixes
    text = re.sub(r'^.*?:\s*', '', text)
    # Remove text in brackets
    text = re.sub(r'\(.*?\)', '', text)
    # Remove quotes
    text = text.replace('"', '').replace("'", '')
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Helper function to replace spans with EC/PC markers
def mark_chunk(cq, spans, chunktype, offset, counter, ec_dict=None):
    """Replace text in question with EC/PC markers and update dictionary"""
    for (start, end) in spans:
        original_text = cq[start - offset:end - offset].strip()
        if not original_text:
            continue
        # Create marker, e.g., EC1, PC2
        marker = f"{chunktype}{counter}"
        if ec_dict is not None:
            ec_dict[marker] = original_text
        # Add space after marker if needed
        next_char = cq[end - offset:end - offset + 1]
        if next_char and not next_char.isspace():
            replacement = marker + " "
        else:
            replacement = marker
        # Replace text with marker
        cq = cq[:start - offset] + replacement + cq[end - offset:]
        offset += (end - start) - len(replacement)
    return cq, offset

# Extract ECs (Entity Chunks)
def extract_EC_chunks(cq):
    """
    Identify EC chunks (nouns or adjectives representing key entities)
    and replace them with markers (EC1, EC2, ...).
    Returns modified question and mapping dictionary.
    """
    cq = clean_text(cq)
    doc = nlp(cq)
    ec_dict = {}   # dictionary mapping EC1 -> text
    counter = 1
    offset = 0

    # Special case: handle "How + ADJ + VERB" questions
    # e.g., "How quick is ..." : mark "quick" as EC
    if len(doc) > 2 and doc[0].text.lower() == 'how' and doc[1].pos_ == 'ADJ' and doc[2].pos_ == 'VERB':
        start = doc[1].idx
        end = start + len(doc[1])
        cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter, ec_dict)
        counter += 1

    # Iterate over noun phrases in the question
    for chunk in doc.noun_chunks:
        start, end = chunk.start_char, chunk.end_char
        span_tokens = list(chunk)
        i = 0
        # Skip question words and small fillers at the start of chunk
        if span_tokens[0].text.lower() in QUESTION_WORDS:
            i = 1
            while i < len(span_tokens) and span_tokens[i].pos_ in ['ADV', 'PRON', 'DET'] \
                  and span_tokens[i].text.lower() not in ['the', 'a', 'an']:
                i += 1
        trimmed_tokens = span_tokens[i:]
        if not trimmed_tokens:
            continue

        # Split noun phrases at "and" into multiple ECs
        ec_splits = []
        temp_start = trimmed_tokens[0].idx
        for j, tok in enumerate(trimmed_tokens):
            if tok.pos_ == 'CCONJ' and tok.text.lower() == 'and':
                temp_end = tok.idx
                ec_splits.append((temp_start, temp_end))
                if j + 1 < len(trimmed_tokens):
                    temp_start = trimmed_tokens[j+1].idx
        temp_end = trimmed_tokens[-1].idx + len(trimmed_tokens[-1])
        ec_splits.append((temp_start, temp_end))

        # Mark valid ECs
        for start_ec, end_ec in ec_splits:
            ec_text = cq[start_ec - offset:end_ec - offset].strip()
            if ec_text.lower() in REJECTING_EC:
                continue
            cq, offset = mark_chunk(cq, [(start_ec, end_ec)], "EC", offset, counter, ec_dict)
            counter += 1

    return cq, ec_dict

# Get PC spans (Process Chunks)
def get_PCs_as_spans(cq):
    doc = nlp(cq)

    # Words to ignore if they start the question
    AUX_VERBS_START = {"do", "does", "given"}

    def _is_auxilary(token, chunk_token_ids):
        return (token.head.i in chunk_token_ids and token.dep_ == 'aux' and token.i not in chunk_token_ids)

    def _get_span(group, doc):
        ids = [int(x.split("::")[0]) for x in group.split(",")]
        aux = None
        for token in doc:
            if _is_auxilary(token, ids):
                aux = token
        main_verb = doc[ids[-1]].text.lower()

        # Reject if main verb is auxiliary and at start of sentence
        if main_verb in AUX_VERBS_START and doc[ids[-1]].i == doc[ids[-1]].sent.start:
            return None

        return (doc[ids[0]].idx, doc[ids[-1]].idx + len(doc[ids[-1]]), aux)

    # Remove subspans that are inside bigger spans
    def _reject_subspans(spans):
        filtered = []
        for span in spans:
            if span is None:
                continue
            if any(span[0] >= other[0] and span[1] <= other[1] for other in spans if span != other):
                continue
            filtered.append(span)
        return filtered

    # Regex over POS tags to find verbs
    pos_text = ",".join([f"{i}::{t.pos_}" for i, t in enumerate(doc)])
    regexes = [r"([0-9]+::(PART|VERB),?)*([0-9]+::VERB)"]
    spans = []
    for regex in regexes:
        for m in re.finditer(regex, pos_text):
            span = _get_span(m.group(), doc)
            if span is not None:
                spans.append(span)
    return _reject_subspans(spans)

# Extract PCs (Process Chunks)
def extract_PC_chunks(cq):
    """
    Extract PC chunks (verbs/verb phrases) but ignore auxiliary verbs.
    Returns the question with PC markers and a mapping dictionary.
    """
    offset = 0
    counter = 1
    pc_dict = {}

    for begin, end, aux in get_PCs_as_spans(cq):
        # Ignore auxiliary verbs completely
        aux = None
        spans = [(begin, end)]
        # Replace with PC marker and update dictionary
        cq, offset = mark_chunk(cq, spans, "PC", offset, counter, pc_dict)
        counter += 1

    return cq, pc_dict
