"""
Extract Top Predicted Research Questions

This script filters scored candidate questions and keeps only those with a predicted RQ score above a threshold.

- Reads predicted scores from 'Data/RQs-MODEL/predicted_rqs-MODEL.jsonl'
- Keeps only questions with predicted_score >= threshold
- Writes top RQs to 'Data/RQs-MODEL/rqs_above_THRESHOLD-MODEL.jsonl'

How to run:
    python extractTopRQs.py [model] [threshold]

Example usage:
    Run default for BERT (threshold 0.85):
        python extractTopRQs.py

    Run SCIBERT with custom threshold:
        python extractTopRQs.py bert 0.9

"""

import json
import argparse
from pathlib import Path

DEFAULT_THRESHOLD = 0.85

def main():
    parser = argparse.ArgumentParser(description="Filter predicted RQs by score threshold")
    parser.add_argument("model", nargs="?", default="bert", choices=["bert", "scibert"],
                        help="Choose model (default: %(default)s)")
    parser.add_argument("threshold", nargs="?", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()
    
    model = args.model.upper()
    threshold = args.threshold

    INPUT_FILE = Path(f"Data/RQs-{model}/predicted_rqs-{model}.jsonl")
    OUTPUT_FILE = Path(f"Data/RQs-{model}/rqs_above_{threshold}-{model}.jsonl")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                score = float(data.get("predicted_score", 0))
                if score >= threshold:
                    outfile.write(json.dumps(data) + "\n")
                    kept += 1
            except json.JSONDecodeError:
                continue

    print(f"Filtered {kept} RQs with score >= {threshold}.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
