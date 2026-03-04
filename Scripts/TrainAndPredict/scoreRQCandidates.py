"""
RQ Candidate Scorer

This script loads a trained RQ classification model and scores unreviewed candidate questions.

Each question is scored based on the model's probability that it is a research question (label=1).
The results are written to a new JSONL file.

How to run:

Run default for BERT:
    python scoreRQCandidates.py
    - Model directory: models/bert-rq-checkpoint
    - Input: Data/RQs-BERT/question_candidates-BERT.jsonl
    - Output: Data/RQs-BERT/predicted_rqs-BERT.jsonl

Run for SciBERT:
    python scoreRQCandidates.py scibert
    - Model directory: models/scibert-rq-checkpoint
    - Input: Data/RQs-SCIBERT/question_candidates-SCIBERT.jsonl
    - Output: Data/RQs-SCIBERT/predicted_rqs-SCIBERT.jsonl
"""

import json
import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Score RQ candidate questions")
    parser.add_argument("model", nargs="?", default="bert", choices=["bert", "scibert"],
                        help="Choose model (default: %(default)s)")
    args = parser.parse_args()
    
    model_name = args.model.lower()
    
    if model_name == "bert":
        MODEL_DIR = "models/bert-rq-checkpoint"
        INPUT_FILE = "Data/RQs-BERT/question_candidates-BERT.jsonl"
        OUTPUT_FILE = "Data/RQs-BERT/predicted_rqs-BERT.jsonl"
    else:  # scibert
        MODEL_DIR = "models/scibert-rq-checkpoint"
        INPUT_FILE = "Data/RQs-SCIBERT/question_candidates-SCIBERT.jsonl"
        OUTPUT_FILE = "Data/RQs-SCIBERT/predicted_rqs-SCIBERT.jsonl"

    # Load Model
    pipe = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, return_all_scores=True)

    # Load Unreviewed Questions
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    unreviewed = [q for q in candidates if q.get("reviewed") == 0]

    # Score Questions
    results = []
    for entry in unreviewed:
        question = entry["question"]
        paper_id = entry["paper_id"]
        try:
            score = pipe(question)[0][1]["score"]  # label 1 = RQ
            results.append({
                "paper_id": paper_id,
                "question": question,
                "predicted_score": round(score, 4)
            })
        except Exception as e:
            print(f"Failed to score question: {question}\n{e}")

    # Write Results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nScored {len(results)} questions.")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
