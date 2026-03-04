# RQ CNL Project

The purpose of this project is to create **Controlled Natural Language (CNL) templates** from a set of research questions (RQs) extracted from the bodies of academic papers.

# Dependency installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Project structure
Data/                       # Datasets and intermediate files
Models/                     # Trained models
Scripts/
  Evaluate/                 # Scripts used for RQ and template evaluation
  Extract/                  # Corpus creation and RQ extraction
    downloadPapers.py       # Download academic papers
    processPapers.py        # Process papers into text
    extractQuestions.py     # Extract candidate questions
    extractTopRQs.py        # Extract top scored research questions
    cleanRQs.py             # Clean extracted research questions
  Templates/                # CNL template creation
    splitRQs.py             # Split research questions into template and test set
    Generate.py             # Generate templates
  TrainAndPredict/          # Model training and prediction
    trainRQClassifier.py    # Train the classifier
    scoreRQCandidates.py    # Score RQ candidates

# Data Preparation

python Scripts/Extract/downloadPapers.py
python Scripts/Extract/processPapers.py
python Scripts/Extract/extractQuestions.py

# Model Training

python Scripts/TrainAndPredict/trainRQClassifier.py

# Score Candidate Questions

python Scripts/TrainAndPredict/scoreRQCandidates.py

# Research Question Extraction

python Scripts/Extract/extractTopRQs.py
python Scripts/Extract/cleanRQs.py

# Template Creation 

python Scripts/Templates/splitRQs.py
python Scripts/Templates/Generate.py


