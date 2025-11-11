# Codegen-350M Fine-tuning for Code Review

This repository contains the code and resources for fine-tuning the Salesforce Codegen-350M-multi model for automated code review comment generation. This project is a crucial part of the larger "Live AI Review Assistant" system, aiming to provide intelligent, context-aware feedback on Python code within a secure AWS environment.

## Project Goal
The primary objective is to adapt a pre-trained code generation model (Codegen-350M) to the specific task of generating human-like code review comments. This involves collecting and preprocessing relevant datasets, implementing a robust fine-tuning pipeline, and evaluating the model's performance on comment generation metrics.

## Key Features

*   **LoRA Fine-tuning**: Efficient adaptation of the Codegen-350M model using Low-Rank Adaptation (LoRA) to minimize computational resources.
*   **Data Preparation Pipeline**: Scripts for fetching, cleaning, and formatting code review datasets into `(prompt, response)` pairs.
*   **MLflow Integration**: Comprehensive tracking of experiments, parameters, metrics, and artifacts for reproducibility and comparison.
*   **Automated Evaluation**: Tools for assessing generated comments using metrics like ROUGE, BERTScore, and custom similarity measures.
*   **AWS S3 Adapter Storage**: Mechanism for versioned storage of trained model adapters, ready for deployment.
*   **GPU Optimized**: Configured for efficient training on GPU (e.g., RTX 2060) with FP16 precision.

## Repository Structure

```
.
├── adapters/                  # Stores trained LoRA adapters and tokenizers
├── configs/                   # Configuration files for training and evaluation
├── data/                      # Raw and prepared datasets
│   ├── prepared/              # Cleaned and processed data for fine-tuning
│   └── raw/                   # Raw datasets fetched from sources
├── outputs/                   # Stores evaluation charts and metrics
├── src/                       # Source code for the fine-tuning pipeline
│   ├── data/                  # Scripts for data fetching and preprocessing
│   │   ├── sources/           # Scripts to fetch data from specific sources (e.g., Kaggle)
│   │   ├── fetch.py           # Fetches raw data
│   │   └── preproc.py         # Preprocesses raw data
│   ├── eval.py                # Model evaluation script
│   ├── resolve.py             # Script for uploading adapters to S3
│   ├── schema.py              # Defines data schema
│   ├── train.py               # Model fine-tuning script
│   └── utils/                 # Utility functions (MLflow integration, helpers)
├── Dockerfile                 # For building Docker image for training environment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Methodology

The fine-tuning process follows a structured machine learning pipeline:

1.  **Data Acquisition & Preparation**:
    *   Code fragments and developer comments are collected from public datasets (e.g., `nutanix/codereview-dataset`, `bulivington/code-review-data-v2`).
    *   Data is cleaned, filtered (Python-only), and formatted into `(prompt, response)` pairs.
    *   Duplicates and overly long comments are handled, and data is split into training (90%) and validation (10%) sets, then saved in JSONL format.

2.  **Model Fine-Tuning (LoRA)**:
    *   The `Salesforce/codegen-350M-multi` base model is loaded.
    *   LoRA (Low-Rank Adaptation) technique is applied to efficiently fine-tune selected attention layers, significantly reducing computational requirements.
    *   Training is performed using PyTorch and Hugging Face Transformers + TRL, with FP16 precision for faster GPU computation.
    *   All training parameters (e.g., `max_seq_len=512`, `learning_rate=2e-4`, `epochs=9`, `seed=42`) and metrics are logged to MLflow.

3.  **Model Evaluation**:
    *   The fine-tuned model generates comments for a validation set of Python code snippets.
    *   Generated comments are compared against original developer feedback using a suite of metrics: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BERTScore (Precision, Recall, F1), average character-level similarity, generation length statistics, and percentage of empty/punctuated responses.
    *   Evaluation results and charts are automatically recorded in MLflow.

4.  **Adapter Storage & Deployment**:
    *   The trained LoRA adapter and tokenizer are saved locally and then uploaded to a versioned S3 bucket with a metadata manifest. This enables easy loading of specific model versions by the "Live AI Review Assistant" worker.


