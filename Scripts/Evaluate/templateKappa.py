"""
Evaluate Cohen's Kappa for BERT and SciBERT Ratings

This script calculates inter-rater agreement
for BERT and SciBERT evaluations using Kuhn's PENS classification
scheme (Precision, Naturalness, Simplicity).

The ratings are stored in an Excel file with:
- Emma's ratings on the first sheet
- Rector's ratings on the second sheet
- BERT values in rows 4-28
- SciBERT values in rows 32-56

Columns are as follows:
- Column B: Precision
- Column C: Naturalness
- Column D: Simplicity

Steps:
1. Load Excel file
2. Extract ratings by sheet and row range
3. Compute Cohen's Kappa for each dimension
4. Print results

Usage:
    python templateKappa.py
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load Excel file
file_path = "Data/TemplateEvaluation/TemplateEvaluation.xlsx"
xl = pd.ExcelFile(file_path)

# Function to extract ratings by row numbers from a specific sheet
def get_ratings_by_rows(sheet_name, start_row, end_row):
    # Parse without headers
    df = xl.parse(sheet_name, header=None)
    # Slice the DataFrame for the given rows (rows start at 1)
    df_slice = df.iloc[start_row-1:end_row]
    # Columns: B=1, C=2, D=3
    precision = df_slice.iloc[:, 1].tolist()
    naturalness = df_slice.iloc[:, 2].tolist()
    simplicity = df_slice.iloc[:, 3].tolist()
    return precision, naturalness, simplicity

# Emma ratings (sheet 1 = Emma)
precision_bert, naturalness_bert, simplicity_bert = get_ratings_by_rows("Emma", 4, 28)
precision_scibert, naturalness_scibert, simplicity_scibert = get_ratings_by_rows("Emma", 32, 56)

# Rector ratings (sheet 2 = Rector)
precision_bert_r, naturalness_bert_r, simplicity_bert_r = get_ratings_by_rows("Rector", 4, 28)
precision_scibert_r, naturalness_scibert_r, simplicity_scibert_r = get_ratings_by_rows("Rector", 32, 56)

# Function to compute Cohen's kappa
def compute_kappa(dim1, dim2, name):
    kappa = cohen_kappa_score(dim1, dim2)
    print(f"Cohen's Kappa for {name}: {kappa:.3f}")

# Evaluate BERT
print("BERT Evaluation:")
compute_kappa(precision_bert, precision_bert_r, "Precision")
compute_kappa(naturalness_bert, naturalness_bert_r, "Naturalness")
compute_kappa(simplicity_bert, simplicity_bert_r, "Simplicity")

# Evaluate SCIBERT
print("\nSCIBERT Evaluation:")
compute_kappa(precision_scibert, precision_scibert_r, "Precision")
compute_kappa(naturalness_scibert, naturalness_scibert_r, "Naturalness")
compute_kappa(simplicity_scibert, simplicity_scibert_r, "Simplicity")

