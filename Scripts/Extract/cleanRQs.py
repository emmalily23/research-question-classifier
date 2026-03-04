"""
Clean, Split, and Deduplicate Research Questions

This script processes a JSONL file of RQs:
- Cleans introductory phrases, numbering, brackets, and extra spaces
- Splits multiple questions in one string into separate questions
- Deduplicates questions by (paper_id, question)
- Saves output to a cleaned JSONL file

How to run:
    python cleanRQs.py path/to/input_file.jsonl

Example:
    python cleanRQs.py Data/RQs-BERT/rqs_above_0.85-BERT.jsonl
"""

import json
import re
import argparse
from pathlib import Path

# Patterns for removing common introductory phrases from questions
INTRO_PATTERNS = [
    r'^.*?(Research Questions\s*\(.*?\)\s*[:])',
    r'^.*?(we (examine|ask|pose|explore) the question\s*[:])',
    r'^.*?(we aim to answer (the )?question[s]?\s*[:])',
    r'^.*?(The question[s]? (we|this paper) (aims to|seeks to|tries to)? answer[s]?\s*[:])',
    r'^.*?(the following question[s]?\s*[:])',
    r'^.*?(this (paper|study) (investigates|asks|seeks)\s*[:])',
]

# Common prefixes to remove at the start of questions
REMOVE_PREFIXES = [
    r'firstly', r'first', r'secondly', r'second', r'thirdly', r'third', r'lastly',
    r'in other words', r'this begs the question', r'in particular', r'however',
    r'and', r'for instance', r'if so'
]

def clean_rq_text(text):
    """Cleans a raw question string."""
    text = text.strip()
    
    # Remove introductory patterns
    for pattern in INTRO_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove section references like 'Section 2.1'
    text = re.sub(r'^(?:Section\s*\d+(\.\d+)*[\)\.:]?\s*)+', '', text, flags=re.IGNORECASE)
    
    # Keep only text after last colon
    if ':' in text:
        text = text.split(':')[-1].strip()
    
    # Remove RQ or Q numbering patterns at the start
    text = re.sub(r'^\s*\(?(?:RQ\s*\d+|RQ\d+|Q\d+)\)?[\)\.:]?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove numbered lists at start (1), 2., etc.)
    text = re.sub(r'^\(?\d+\)?[\)\.:]?\s*', '', text)
    
    # Remove leading prefixes
    for prefix in REMOVE_PREFIXES:
        text = re.sub(rf'^\s*{prefix}\s*[:,]?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove backslash sequences
    text = re.sub(r'\\\S+', '', text)
    
    # Remove brackets and contents
    text = re.sub(r'\s*[\(\[].*?[\)\]]', '', text)
    
    # Remove quotes
    text = text.replace('"', '').replace('“', '').replace('”', '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Ensure question ends with a question mark
    if text and not text.endswith('?'):
        text += '?'
    
    return text

def split_multiple_questions(text):
    """Splits multiple questions in one string into separate questions."""
    # Patterns for splitting based on punctuation and numbering
    text = re.sub(r'\?\s*;\s*and\s+', '?|||', text, flags=re.IGNORECASE)
    text = re.sub(r'\?\s*and\s*\d+\)', '?|||', text, flags=re.IGNORECASE)
    text = re.sub(r';\s*\d+\)|;\s*\d+\.', '|||', text, flags=re.IGNORECASE)
    text = re.sub(r'\?\s*\d+\)', '?|||', text, flags=re.IGNORECASE)
    text = re.sub(r'\?\s*\d+\.', '?|||', text, flags=re.IGNORECASE)
    
    # Split into separate questions
    parts = text.split('|||')
    clean_parts = []
    for p in parts:
        p = p.strip()
        if p and not p.endswith('?'):
            p += '?'
        if p:
            clean_parts.append(p)
    return clean_parts

def main():
    """Main function to process the input JSONL file."""
    parser = argparse.ArgumentParser(description="Clean RQs from a JSONL file")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    # Output file has '_cleaned' appended
    output_file = input_file.with_name(input_file.stem + "_cleaned.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    seen = set()  # Track duplicates
    kept = 0      # Counter for kept questions

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)
            paper_id = entry.get("paper_id")
            question = entry.get("question", "")

            # Clean and split questions
            cleaned = clean_rq_text(question)
            split_questions = split_multiple_questions(cleaned)

            for q in split_questions:
                key = (paper_id, q.lower())
                if key not in seen:
                    seen.add(key)
                    new_entry = dict(entry)
                    new_entry["question"] = q
                    outfile.write(json.dumps(new_entry) + "\n")
                    kept += 1

    print(f"\nCleaned {kept} questions.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
