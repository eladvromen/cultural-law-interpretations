# Determination Pipeline

Quick guide for running determination extraction models and experiments.

## Running Models & Experiments
### main pipeline: 
- text_processing (given you already have a validation_set.csv in pipeline.data) - for extracting headers
- run_sparse_extraction.py - for high prescion extraction from all text
- run_basic_extraction.py - for headers only more loose extraction
- run_ngram_extraction.py - even more loose on headers. 
### Option 1: fill pipeline 
run pipeline_runner.py

### Option 2: Direct Model Running

Run any model directly using model runners
python basic_runner.py
python ngram_runner.py
python sparse_runner.py

### Option 3: Configurable Experiments
From `pipeline/utils/experiment_runner.py`:
```python
# Run experiments with different configurations

```
Example configuration:
```python
config = ExperimentConfig(
    model_name='sparse',  # or 'ngram', 'basic'
    model_params={},
    experiment_name='my_experiment',
    description='Custom experiment'
)
```
with this option, you can get ana evaluation analysis

