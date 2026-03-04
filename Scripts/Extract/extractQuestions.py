"""
Question Extractor from Academic Papers

This script extracts question sentences from a collection of academic paper texts.

- Iterates through all '.jsonl' files in 'Data/extracted_texts-MODEL/'
- Identifies valid questions (ending in '?', not URLs, at least 5 words)
- Appends the results to 'Data/RQs-MODEL/question_candidates-MODEL.jsonl'

How to run:
1. Default run (uses BERT paths):
   python extractQuestions.py
   - Input: Data/extracted_texts-BERT/
   - Output: Data/RQs-BERT/question_candidates-BERT.jsonl

2. Custom run (e.g., SciBERT):
   python extractQuestions.py scibert
   - Input: Data/extracted_texts-SCIBERT/
   - Output: Data/RQs-SCIBERT/question_candidates-SCIBERT.jsonl
"""

import json
from pathlib import Path
from tqdm import tqdm
import spacy
import re
import argparse

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def clean_question(text):
    text = text.strip()
    if re.match(r'^RQ[0-9]*[):\s]', text):
        return text
    match = re.search(r'[A-Z][^\n]*\?$', text)
    if match:
        return match.group().strip()
    return ""


def is_valid_question(text):
    return (
        text.endswith("?")
        and len(text.split()) >= 5
        and not text.lower().startswith(('http://', 'https://'))
    )


def extract_questions_from_text(text):
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if is_valid_question(sent_text):
            cleaned = clean_question(sent_text)
            if cleaned:
                questions.append(cleaned)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Extract questions from academic papers")
    parser.add_argument("model", nargs="?", default="bert", choices=["bert", "scibert"],
                        help="Model name (default: %(default)s)")
    args = parser.parse_args()

    model = args.model.upper()
    INPUT_DIR = Path(f"Data/extracted_texts-{model}")
    OUTPUT_DIR = Path(f"Data/RQs-{model}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / f"question_candidates-{model}.jsonl"

    processed = 0
    written = 0
    existing_questions = set()

    # Load existing entries to avoid duplicates
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                try:
                    entry = json.loads(line)
                    key = (entry["paper_id"], entry["question"])
                    existing_questions.add(key)
                except json.JSONDecodeError:
                    continue

    # Open output for appending
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        for file_path in sorted(INPUT_DIR.glob("*.jsonl")):
            print(f"Processing {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in tqdm(infile, desc=f"  Extracting from {file_path.name}"):
                    try:
                        paper = json.loads(line)
                        paper_id = paper.get("paper_id", "")
                        body_text = paper.get("body_text", "")

                        questions = extract_questions_from_text(body_text)

                        for q in questions:
                            key = (paper_id, q)
                            if key not in existing_questions:
                                flat_entry = {
                                    "paper_id": paper_id,
                                    "question": q,
                                    "RQ": 0,
                                    "reviewed": 0
                                }
                                outfile.write(json.dumps(flat_entry) + "\n")
                                existing_questions.add(key)
                                written += 1

                        processed += 1
                    except Exception as e:
                        print(f"Error processing line: {e}")

    print(f"\nDone.")
    print(f"Total papers processed: {processed}")
    print(f"New questions written: {written}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
