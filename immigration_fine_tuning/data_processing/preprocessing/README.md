# Immigration Data Preprocessing Pipeline

This directory contains the preprocessing pipeline for the immigration fine-tuning dataset.

## Pipeline Overview

The preprocessing pipeline consists of the following steps:

1. **Initial Preprocessing** (`intial_preprocessing.py`): Loads and converts raw datasets from various formats.
2. **Full Text Initial Processing** (`full_text_initial_processing.py`): Processes case archives and extracts text content.
3. **Data Cleaning** (`data_cleaning.py`): Cleans and filters datasets, removing duplicates and invalid records.
4. **Merging DataFrames** (`merging_dfs.py`): Enriches datasets with metadata by merging multiple sources.

## Running the Pipeline

To run the complete pipeline, execute:

```bash
python run_pipeline.py
```

This will process all steps in sequence and generate the following outputs:

- `data/processed/`: Intermediate processed datasets
- `data/clean_data/`: Cleaned datasets with duplicates removed
- `data/merged/`: Final enriched datasets ready for model training

## Individual Components

Each component can also be run independently:

```bash
python intial_preprocessing.py
python full_text_initial_processing.py
python data_cleaning.py
python merging_dfs.py
```

## Data Flow

Raw data → Initial preprocessing → Full text processing → Data cleaning → Merged datasets

## Merged Dataset Features

The merging process (`merging_dfs.py`) combines several datasets to create enriched versions of the test and train datasets with the following features:

1. **Original Data**: Preserves all original columns from the test/train datasets.
2. **Full Text**: Adds the full text of each case from the case_texts dataset.
3. **Sentence Text**: Adds ordered, concatenated sentences from the all_sentences dataset.
4. **Determination Statements**: 
   - Groups all determination statements for each decision ID into a list
   - Adds a count of determination statements per decision
   - Handles cases with no determination statements by providing empty lists
5. **Entity Information**: Adds entity information from the all_entities_inferred dataset.

The merged datasets are saved in both CSV and pickle formats:
- Pickle format (.pkl) preserves all data types, including lists
- CSV format converts list columns to strings for compatibility

## Recent Updates

- Added sentence text extraction from all_sentences dataset
- Implemented determination statement grouping by decision ID
- Fixed type consistency issues for merging operations
- Added robust error handling for list processing
- Improved CSV export with proper list serialization

## Additional Components

- `determination_removal.py`: Specialized processor for removing determination statements from legal documents. 