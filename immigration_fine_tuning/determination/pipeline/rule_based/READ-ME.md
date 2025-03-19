# Rule-Based Determination Extraction System

This system provides tools for extracting legal determinations from immigration case documents using rule-based approaches. The system consists of three main components:

## 1. Text Processing (`text_processing.py`)

A comprehensive text preprocessing pipeline that handles document-level structure and cleaning:

### Key Components:
- `TextCleaner`: Handles basic text cleaning operations
  - Removes page pointers
  - Filters French content
  - Normalizes whitespace
  - Cleans redacted information
  - Extracts metadata

- `HeaderDetector`: Identifies and structures document sections
  - Detects section headers
  - Maintains header hierarchy
  - Extracts section content

- `OutcomeExtractor`: Extracts decision outcomes
  - Identifies allowed/dismissed/refused decisions
  - Provides confidence scores
  - Extracts outcome text

- `LegalTextPreprocessor`: Orchestrates the preprocessing pipeline
  - Combines cleaning, header detection, and outcome extraction
  - Provides structured document representation

## 2. Section Header Analysis (`analyze_section_headers.py`)

A tool for exploratory data analysis of document section headers:

### Features:
- Identifies potential section headers
- Analyzes header frequency and patterns
- Generates header statistics
- Visualizes header distributions
- Helps understand document structure

### Usage:
```python
analyzer = SectionHeaderAnalyzer()
analyzer.process_dataframe(df)
analyzer.plot_top_headers()
analyzer.get_key_header_stats()
```

## 3. Determination Extractors

Two approaches for extracting determinations from processed documents:

### Basic Determination Extractor (`basic_determination_extraction.py`)
- Pattern-based extraction using regular expressions
- Rule-based scoring system
- Optimized for precision

### N-gram Determination Extractor (`ngram_determination_extraction.py`)
- Uses n-gram matching for flexible pattern recognition
- Configurable n-gram sizes and matching thresholds
- Better recall for variant phrasings

### Base Class (`base_determination_extractor.py`)
- Provides common functionality
- Handles text cleaning and normalization
- Manages training examples

## Usage Example

```python
from text_processing import LegalTextPreprocessor
from ngram_determination_extraction import NgramDeterminationExtractor

# Initialize processors
preprocessor = LegalTextPreprocessor()
extractor = NgramDeterminationExtractor()

# Load training data
extractor.load_training_examples('path/to/train.csv', 'path/to/test.csv')

# Process a document
text = "..."
processed_doc = preprocessor.preprocess(text)
determinations = extractor.process_case(processed_doc['cleaned_text'])
```

## Configuration

The system can be configured through YAML files in the `configs/` directory:
- `experiments.yaml`: Experiment configurations
- `paths.yaml`: File path configurations

## Performance Considerations

- Text cleaning operations are cached for improved performance
- Batch processing available for large datasets
- Configurable thresholds for precision/recall tradeoffs

## Dependencies

- pandas
- numpy
- regex
- pyyaml
- tqdm

## Installation

```bash
pip install -r requirements.txt
```
```

This README provides a good overview of the system's components while remaining concise and practical. Let me know if you'd like me to expand on any section or add additional information!