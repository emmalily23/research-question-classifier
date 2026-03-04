"""
RQ Chunking and Template Generation

This script reads research questions and applies EC (noun phrase) and PC (verb phrase) chunking to generate templated questions.

Adapted from: AgOCQs_Plus (https://github.com/AdeebNqo/AgOCQs_Plus)

"""

import pandas as pd
from pathlib import Path
from ChunkingLib import extract_EC_chunks, extract_PC_chunks

def read_rqs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return pd.DataFrame(questions, columns=["OriginalQuestion"])

def add_templated_question(df):
    templated_questions = []
    ec_mappings_list = []
    pc_mappings_list = []

    for q in df["OriginalQuestion"]:
        # Extract EC chunks
        template_ec, ec_map = extract_EC_chunks(q)
        # Extract PC chunks (aux ignored) and get mapping
        template_ec_pc, pc_map = extract_PC_chunks(template_ec)

        templated_questions.append(template_ec_pc)
        ec_mappings_list.append("; ".join([f"{k}={v}" for k, v in ec_map.items()]))
        pc_mappings_list.append("; ".join([f"{k}={v}" for k, v in pc_map.items()]))

    df["TemplatedQuestion"] = templated_questions
    df["EC_Mapping"] = ec_mappings_list
    df["PC_Mapping"] = pc_mappings_list
    return df

def main():
    input_dir = Path("Data/inputText/request")
    output_dir = Path("Data/templates")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.txt"))
    print(f"Found {len(input_files)} input files: {[f.name for f in input_files]}")

    for input_file in input_files:
        df = read_rqs(input_file)
        print(f"Processing {input_file.name}, {len(df)} lines")

        df = add_templated_question(df)

        output_file = output_dir / f"{input_file.stem}_with_EC_and_PC_chunks.xlsx"
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Saved RQ templates with ECs and PCs to {output_file.resolve()}")

if __name__ == "__main__":
    main()
