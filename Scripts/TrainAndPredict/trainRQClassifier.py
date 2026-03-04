"""
This script fine-tunes a BERT-based model to classify whether a sentence is a research question or not.
It uses manually labeled data stored in JSONL format and HuggingFace Transformers for tokenization and model training.

Steps:
1. Load labeled question data
2. Filter for reviewed entries
3. Tokenize using a pretrained BERT tokenizer
4. Fine-tune a binary classification model
5. Evaluate using accuracy, F1, precision, and recall
6. Save the trained model and tokenizer to disk

Usage:
    python trainRQClassifier.py
"""

import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configuration
MODEL_NAME = "bert-base-uncased"  # "allenai/scibert_scivocab_uncased"
OUTPUT_DIR = "models/bert-rq-checkpoint"  # "models/scibert-rq-checkpoint"

LABELED_FILE = "Data/question_manually_labelled.jsonl"
NUM_EPOCHS = 4
BATCH_SIZE = 8

# Load and filter manually labelled data
with open(LABELED_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

# Keep only reviewed entries
reviewed = [d for d in data if d.get("reviewed") == 1]

# Create DataFrame with question and RQ label
df = pd.DataFrame(reviewed)[["question", "RQ"]].rename(columns={"RQ": "label"})

# Train-test split with stratification
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Convert to HuggingFace datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["question"], padding=True, truncation=True)

# Apply tokenizer to datasets
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Define Evaluation Metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

# Training Configuration
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1  # Keep only the most recent checkpoint
)

# Train Model
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save Final Model and Tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to: {OUTPUT_DIR}")
