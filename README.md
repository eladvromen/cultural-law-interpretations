# Immigration Law Fine-Tuning Pipeline

This project implements a Parameter-Efficient Fine-Tuning (PEFT) pipeline for legal language models using the AsyLex dataset, which contains immigration law data from Canada.

## Project Overview

The pipeline focuses on:
1. Data preparation and preprocessing of the AsyLex dataset
2. Baseline model fine-tuning using legal domain models
3. PEFT implementation for efficient fine-tuning
4. Evaluation of model performance and catastrophic forgetting

## Dataset

The AsyLex dataset contains 59,112 refugee status determination documents from Canada (1996-2022), designed for legal entity extraction and judgment prediction tasks. The dataset is available on Hugging Face: [clairebarale/AsyLex](https://huggingface.co/datasets/clairebarale/AsyLex)

## Project Structure

```
immigration_fine_tuning/
├── config/             # Configuration files for models and training
├── data/               # Data processing and loading utilities
├── models/             # Model definitions and handlers
├── trainers/           # Training and evaluation code
└── utils/              # Utility functions
```

## Setup and Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the fine-tuning pipeline
python scripts/run_fine_tuning.py
```

## Experiments

The project includes experiments to:
1. Establish baseline performance with full fine-tuning
2. Implement PEFT methods (LoRA, Adapters, etc.)
3. Evaluate catastrophic forgetting by measuring performance on general legal tasks
4. Document the trade-off between performance on general legal tasks and immigration-specific tasks 