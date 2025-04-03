# Transformer Model Training and Evaluation

This directory contains scripts for training and evaluating transformer-based models for determination classification.

## Core Components

### Training
- `train_classifier.py` - Main script for training transformer models
  - Supports multiple model architectures (e.g., Legal-BERT)
  - Includes data augmentation and custom loss functions
  - Handles class imbalance and training optimization

### Evaluation
- `evaluate_checkpoint.py` - Evaluates individual model checkpoints
- `evaluate_ensemble.py` - Evaluates ensemble of models with majority voting
- `evaluate_hybrid_ensemble.py` - Evaluates hybrid ensemble approaches
- `analyse_ensemble.py` - Analyzes ensemble model performance and misclassifications

## Usage

### Training a Model
```bash
python train_classifier.py --model_name nlpaueb/legal-bert-base-uncased
```

### Evaluating Models
```bash
# Single model evaluation
python evaluate_checkpoint.py --checkpoint_path path/to/checkpoint

# Ensemble evaluation
python evaluate_ensemble.py --config path/to/ensemble_config.json
```

## Data Requirements
- Training data should be in the format produced by the determination pipeline
- Test data should follow the same format as training data
- Checkpoint configurations for ensemble evaluation should specify model paths and thresholds 