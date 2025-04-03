# Determination Pipeline

Quick guide for running determination extraction models and experiments.

## Pipeline Components

### 1. Basic Extraction Pipeline
- `text_processing` - Extracts headers (requires validation_set.csv in pipeline.data)
- `run_sparse_extraction.py` - High precision extraction from all text
- `run_basic_extraction.py` - Header-based extraction with moderate precision
- `run_ngram_extraction.py` - Loose header-based extraction

### 2. Transformer Data Processing
Located in `transformer_data_processor/`:
- `determination_train_set_processor.py` - Processes pipeline output into training data for transformer models
- `determination_test_set_processor.py` - Processes test data for transformer evaluation
- `fine_tuning_data_creator.py` - Creates formatted data for transformer fine-tuning

## Running Models & Experiments

### Option 1: Full Pipeline
Run `pipeline_runner.py`

### Option 2: Direct Model Running
Run any model directly using model runners:
```bash
python basic_runner.py
python ngram_runner.py
python sparse_runner.py
```

### Option 3: Configurable Experiments
From `pipeline/utils/experiment_runner.py`:
```python
config = ExperimentConfig(
    model_name='sparse',  # or 'ngram', 'basic'
    model_params={},
    experiment_name='my_experiment',
    description='Custom experiment'
)
```

## Data Flow
1. Basic pipeline components process raw text and extract determinations
2. Output files from pipeline are processed by transformer_data_processor
3. Processed data is formatted for transformer model training/evaluation

