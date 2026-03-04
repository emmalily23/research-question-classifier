"""
Compute sentence-level BLEU scores between research questions (from test set) and CNL templates.

Usage:
    python bleuEval.py <model>

Example:
    python bleuEval.py bert
    python bleuEval.py scibert

Output:
    Excel file containing:
        RQ: Original research question
        Best Template: Template with highest BLEU-4 match
        BLEU-1, BLEU-2, BLEU-3, BLEU-4: Individual BLEU scores
        Cumulative BLEU: BLEU-4 score used as main evaluation
"""

import sys
import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- Get model type from command line ---
if len(sys.argv) != 2 or sys.argv[1].lower() not in ['bert', 'scibert']:
    print("Usage: python bleuEval.py <bert|scibert>")
    sys.exit(1)

model = sys.argv[1].lower()

# Set file paths based on model
if model == "bert":
    RQ_FILE = "Data/templates/RQs_for_testing-BERT_plain_with_EC_and_PC_chunks.xlsx"
    TEMPLATE_FILE = "Data/templates/RQs_for_cnl-BERT_plain_with_EC_and_PC_chunks.xlsx"
    OUTPUT_FILE = "Data/TemplateEvaluation/bleu_results_BERT.xlsx"
else:  # scibert
    RQ_FILE = "Data/templates/RQs_for_testing-SCIBERT_plain_with_EC_and_PC_chunks.xlsx"
    TEMPLATE_FILE = "Data/templates/RQs_for_cnl-SCIBERT_plain_with_EC_and_PC_chunks.xlsx"
    OUTPUT_FILE = "Data/TemplateEvaluation/bleu_results_SCIBERT.xlsx"

smooth = SmoothingFunction().method1

# load data
rqs_df = pd.read_excel(RQ_FILE)
templates_df = pd.read_excel(TEMPLATE_FILE)

# second column contains text
rqs = rqs_df.iloc[:, 1].dropna().tolist()
templates = templates_df.iloc[:, 1].dropna().tolist()

def clean_text(text):
    """Replace EC1: EC, PC1: PC"""
    return re.sub(r"\b(EC|PC)\d+\b", r"\1", text)

def tokenize(text):
    return text.split()

rqs_tokens = [tokenize(clean_text(rq)) for rq in rqs]
templates_tokens = [tokenize(clean_text(tpl)) for tpl in templates]

# Compute BLEU scores
results = []

for i, rq_tok in enumerate(rqs_tokens):
    best_score = 0
    best_template = None
    best_bleu1 = best_bleu2 = best_bleu3 = best_bleu4 = 0

    for j, tpl_tok in enumerate(templates_tokens):
        # Compute BLEU-1 to BLEU-4 individually
        bleu1 = sentence_bleu([tpl_tok], rq_tok, weights=(1,0,0,0), smoothing_function=smooth)
        bleu2 = sentence_bleu([tpl_tok], rq_tok, weights=(0.5,0.5,0,0), smoothing_function=smooth)
        bleu3 = sentence_bleu([tpl_tok], rq_tok, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
        bleu4 = sentence_bleu([tpl_tok], rq_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

        if bleu4 > best_score:  # use BLEU-4 as cumulative
            best_score = bleu4
            best_template = templates[j]
            best_bleu1, best_bleu2, best_bleu3, best_bleu4 = bleu1, bleu2, bleu3, bleu4

    results.append({
        "RQ": rqs[i],
        "Best Template": best_template,
        "BLEU-1": round(best_bleu1, 4),
        "BLEU-2": round(best_bleu2, 4),
        "BLEU-3": round(best_bleu3, 4),
        "BLEU-4": round(best_bleu4, 4),
        "Cumulative BLEU": round(best_score, 4)
    })

# save to file
results_df = pd.DataFrame(results)
results_df.to_excel(OUTPUT_FILE, index=False)

print(f"Results saved to {OUTPUT_FILE}")
