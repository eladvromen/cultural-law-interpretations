# Determination Extraction Pipeline

A modular pipeline for extracting legal determinations from immigration case documents.

## Overview

This pipeline combines multiple determination extraction models into a unified framework:

1. **SparseExplicitExtractor**: High-precision model that extracts only the most explicit determination statements
2. **BasicDeterminationExtractor**: Balanced model that uses patterns learned from examples
3. **NgramDeterminationExtractor**: Higher-recall model that uses n-gram matching for flexibility

The pipeline processes documents through these models in sequence with configurable thresholds and prioritizes results based on confidence scores.

## Project Structure

```
pipeline/
├── determination_pipeline.py   # Main pipeline implementation
├── model_loader.py            # Model loading and caching utilities
├── run_pipeline.py            # Script to run the full pipeline
├── test_pipeline.py           # Test script for the pipeline
├── rule_based/
│   ├── base_determination_extractor.py   # Base class for extractors
│   ├── sparse_explicit_extraction.py     # High-precision extractor
│   ├── basic_determination_extraction.py # Balanced extractor 
│   ├── ngram_determination_extraction.py # Flexible n-gram extractor
│   └── text_processing.py               # Text preprocessing utilities
└── model_cache/               # Directory for cached models
```

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Navigate to the directory
cd immigration_fine_tuning/determination

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

To run the pipeline on a dataset:

```bash
python pipeline/run_pipeline.py --input data/dataset.csv --output results/output.csv
```

To run with a specific configuration:

```bash
python pipeline/run_pipeline.py --input data/dataset.csv --output results/output.csv --mode precision --sample 100
```

## Pipeline Modes

The pipeline offers three pre-configured modes:

1. **Standard**: Balanced precision and recall
   ```bash
   python pipeline/run_pipeline.py --input data.csv --output results.csv --mode standard
   ```

2. **Precision**: Optimized for high precision
   ```bash
   python pipeline/run_pipeline.py --input data.csv --output results.csv --mode precision
   ```

3. **Recall**: Optimized for high recall
   ```bash
   python pipeline/run_pipeline.py --input data.csv --output results.csv --mode recall
   ```

## Configuration

You can provide a custom configuration file in JSON format:

```bash
python pipeline/run_pipeline.py --input data.csv --output results.csv --config config.json
```

Example configuration file:

```json
{
  "use_sparse_extractor": true,
  "use_basic_extractor": true,
  "use_ngram_extractor": true,
  "sparse_confidence_threshold": 0.9,
  "basic_confidence_threshold": 0.7,
  "ngram_confidence_threshold": 0.6,
  "use_section_extraction": true,
  "target_sections": [
    "decision_headers_text",
    "determination_headers_text",
    "analysis_headers_text",
    "reasons_headers_text",
    "conclusion_headers_text"
  ],
  "apply_post_processing": true,
  "max_determinations": 3,
  "min_words_per_determination": 3,
  "deduplication_similarity": 0.8
}
```

## Pipeline Process

The pipeline follows these steps:

1. **Text Processing**: Extracts relevant sections from documents
2. **High-Precision Extraction**: Applies SparseExplicitExtractor to the full text
3. **Targeted Analysis**: If needed, applies BasicDeterminationExtractor and NgramDeterminationExtractor to relevant sections
4. **Result Combination**: Combines and ranks results based on confidence
5. **Post-processing**: Filters and enhances results for final output

## Model Details

### SparseExplicitExtractor

A high-precision model focused on extracting only the most explicit determination statements. It uses:

- Exact phrase matching
- Regular expressions for explicit determination patterns
- Sentence bank of known determination statements

This model prioritizes precision over recall and is best for capturing obvious determinations.

### BasicDeterminationExtractor

A balanced model that uses patterns learned from examples. Features:

- Optimized pattern matching
- Scores sentences based on learned patterns
- More flexible than SparseExplicit but still relatively precise

This model provides a good balance between precision and recall.

### NgramDeterminationExtractor

A higher-recall model that uses n-gram matching for flexibility. Features:

- Configurable n-gram sizes (2-4 by default)
- Fuzzy matching with configurable threshold
- More robust to variations in language

This model is best for catching determinations with non-standard phrasing.

## Text Section Extraction

The pipeline uses `text_processing.py` to extract relevant sections based on headers:

- `decision_headers_text`: Sections with decision information
- `determination_headers_text`: Sections with explicit determination statements
- `analysis_headers_text`: Sections with case analysis
- `reasons_headers_text`: Sections with reasoning
- `conclusion_headers_text`: Sections with conclusions

This focused extraction helps improve accuracy by targeting the most relevant parts of documents.

## Advanced Usage

### Custom Pipeline Extensions

You can extend the base pipeline for specialized needs:

```python
from determination_pipeline import DeterminationPipeline

class CustomPipeline(DeterminationPipeline):
    """Custom pipeline with domain-specific logic."""
    
    def __init__(self, config=None):
        custom_config = {
            'sparse_confidence_threshold': 0.85,
            # Custom settings...
        }
        if config:
            custom_config.update(config)
        super().__init__(custom_config)
    
    def _apply_post_processing(self, determinations):
        # Custom post-processing logic
        processed = super()._apply_post_processing(determinations)
        # Domain-specific filtering...
        return processed
```

### Model Caching

Models are cached to improve performance. To force retraining:

```bash
python pipeline/run_pipeline.py --input data.csv --output results.csv --force-retrain
```

## Evaluation

To evaluate against ground truth:

```bash
python pipeline/run_pipeline.py --input data.csv --output results.csv --ground-truth determination_column
```

This will calculate precision, recall, and F1 score based on exact matches with the ground truth column.

## Performance Considerations

- For large datasets, use batch processing:
  ```bash
  python pipeline/run_pipeline.py --input large_data.csv --output results.csv --batch-size 100
  ```

- For quick testing, use sampling:
  ```bash
  python pipeline/run_pipeline.py --input data.csv --output results.csv --sample 50
  ```

## Output Format

The pipeline produces the following outputs:

1. **CSV**: The original data with added columns:
   - `sparse_determinations`: Determinations from SparseExplicitExtractor
   - `basic_determinations`: Determinations from BasicDeterminationExtractor
   - `ngram_determinations`: Determinations from NgramDeterminationExtractor
   - `combined_determinations`: All determinations after combining and ranking
   - `primary_determination`: The highest confidence determination
   - `determination_confidence`: Confidence score of the primary determination
   - `determination_source`: Source model of the primary determination

2. **Metrics JSON**: Evaluation metrics in a JSON file

3. **Samples Text**: A text file with sample determinations

## License

[Your license information] 