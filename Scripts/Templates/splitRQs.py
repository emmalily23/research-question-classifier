"""
Split RQs and save as plain text files for CNL and testing sets.

Usage:
    python splitRQs.py <input_file>

Example:
    python splitRQs.py Data/RQs-BERT/rqs_above_0.85-BERT_cleaned.jsonl

Output:
    - Data/inputText/request/RQs_for_cnl-BERT.txt
    - Data/inputText/request/RQs_for_testing-BERT.txt
"""

import json
import random
import argparse
from pathlib import Path

SPLIT_RATIO = 0.8  # 80% for CNL templates, 20% for testing

def save_text_file(data, txt_file: Path):
    """Save list of questions to a plain text file."""
    with open(txt_file, "w", encoding="utf-8") as f:
        for item in data:
            question = item.get("question", "").strip()
            if question:
                f.write(question + "\n")
    print(f"Saved plain text questions to: {txt_file}")

def main():
    parser = argparse.ArgumentParser(description="Split RQs into CNL (80%) and Test (20%) sets as text files")
    parser.add_argument("input_file", help="Path to input JSONL file")
    args = parser.parse_args()

    input_file = Path(args.input_file)

 # Determine model type from filename
    if "SCIBERT" in input_file.name.upper():
        model = "SCIBERT"
    elif "BERT" in input_file.name.upper():
        model = "BERT"
    else:
        model = "UNKNOWN"


    # Output folder
    output_dir = Path("Data/inputText/request")
    output_dir.mkdir(parents=True, exist_ok=True)

    cnl_txt_file = output_dir / f"RQs_for_cnl-{model}.txt"
    test_txt_file = output_dir / f"RQs_for_testing-{model}.txt"

    # Load input JSONL
    with open(input_file, "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile if line.strip()]

    # Shuffle for unbiased splitting
    random.shuffle(data)

    # Split
    split_index = int(len(data) * SPLIT_RATIO)
    cnl_data = data[:split_index]
    test_data = data[split_index:]

    # Save as text files
    save_text_file(cnl_data, cnl_txt_file)
    save_text_file(test_data, test_txt_file)

    print(f"CNL set: {len(cnl_data)} questions: {cnl_txt_file}")
    print(f"Test set: {len(test_data)} questions: {test_txt_file}")

if __name__ == "__main__":
    main()
