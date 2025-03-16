import os
import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, processed_data_dir: str, output_dir: str = None):
        """
        Initialize the data cleaner.
        
        Args:
            processed_data_dir: Directory containing the processed datasets
            output_dir: Directory to save the cleaned datasets
        """
        self.processed_data_dir = Path(processed_data_dir)
        
        if output_dir is None:
            # Create output directories as siblings to processed_data_dir
            self.clean_data_dir = self.processed_data_dir.parent / "clean_data"
            self.unused_data_dir = self.processed_data_dir.parent / "unused_data"
        else:
            base_output_dir = Path(output_dir)
            self.clean_data_dir = base_output_dir / "clean_data"
            self.unused_data_dir = base_output_dir / "unused_data"
        
        # Create output directories
        self.clean_data_dir.mkdir(exist_ok=True, parents=True)
        self.unused_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize dataset containers
        self.datasets = {}
        
    def load_datasets(self) -> None:
        """Load all required datasets from pickle files."""
        dataset_files = {
            'all_sentences': 'all_sentences_anonymized_extracted.pkl',
            'train': 'outcome_train_test_train_dataset_silver.pkl',
            'test': 'outcome_train_test_test_dataset_gold.pkl',
            'determination': 'determination_label_extracted_sentences.pkl',
            'case_cover': 'case_cover_case_cover_entities_and_decision_outcome.pkl',
            'case_texts': 'case_texts_processed.json'
        }
        
        for key, filename in dataset_files.items():
            file_path = self.processed_data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                if filename.endswith('.pkl'):
                    logger.info(f"Loading {key} dataset from {filename}")
                    self.datasets[key] = pd.read_pickle(file_path)
                elif filename.endswith('.json'):
                    logger.info(f"Loading {key} dataset from {filename}")
                    self.datasets[key] = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {filename}")
                    continue
                    
                logger.info(f"Loaded {key} dataset with shape {self.datasets[key].shape}")
            except Exception as e:
                logger.error(f"Error loading {key} dataset: {e}")
    
    def add_sentence_order(self) -> None:
        """
        Add inner order sentence variable to the All Sentences dataset.
        This adds a column indicating the position of each sentence within its document.
        """
        if 'all_sentences' not in self.datasets:
            logger.error("All sentences dataset not loaded")
            return
            
        logger.info("Adding sentence order to All Sentences dataset")
        
        df = self.datasets['all_sentences']
        
        # Check if decisionID column exists
        if 'decisionID' not in df.columns:
            logger.error("decisionID column not found in All Sentences dataset")
            return
            
        # Group by decisionID and add sentence_order column
        df['sentence_order'] = df.groupby('decisionID').cumcount() + 1
        
        logger.info(f"Added sentence_order column to All Sentences dataset")
        self.datasets['all_sentences'] = df
    
    def get_decision_ids(self, dataset_name: str) -> Set[int]:
        """
        Get the set of unique decision IDs from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Set of unique decision IDs
        """
        if dataset_name not in self.datasets:
            logger.error(f"{dataset_name} dataset not loaded")
            return set()
            
        df = self.datasets[dataset_name]
        
        if 'decisionID' not in df.columns:
            logger.error(f"decisionID column not found in {dataset_name} dataset")
            return set()
            
        # Convert to integers if not already
        if not pd.api.types.is_integer_dtype(df['decisionID']):
            df['decisionID'] = pd.to_numeric(df['decisionID'], errors='coerce').fillna(-1).astype(int)
            
        return set(df['decisionID'].unique())
    
    def clean_datasets(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Clean the datasets according to the specified rules.
        
        Returns:
            Tuple of (clean_datasets, unused_data)
        """
        clean_datasets = {}
        unused_data = {}
        
        # 1. Add sentence order to All Sentences dataset
        self.add_sentence_order()
        
        # First, identify duplicated decisionIDs in case_texts dataset
        if 'case_texts' in self.datasets:
            case_texts_df = self.datasets['case_texts']
            
            # Find all decision IDs that appear more than once
            duplicate_ids = case_texts_df[case_texts_df['decisionID'].duplicated(keep=False)]['decisionID'].unique()
            logger.info(f"Found {len(duplicate_ids)} duplicated decision IDs in case_texts dataset")
            
            # Remove all rows where decisionID appears in duplicate_ids (all occurrences)
            clean_case_texts = case_texts_df[~case_texts_df['decisionID'].isin(duplicate_ids)]
            unused_case_texts = case_texts_df[case_texts_df['decisionID'].isin(duplicate_ids)]
            
            logger.info(f"Clean case_texts dataset: {len(clean_case_texts)} rows (removed {len(unused_case_texts)} rows)")
            
            clean_datasets['case_texts'] = clean_case_texts
            unused_data['case_texts'] = unused_case_texts
            
            # Get the set of duplicated IDs to exclude from other datasets
            duplicated_case_text_ids = set(duplicate_ids)
        else:
            duplicated_case_text_ids = set()
            logger.warning("case_texts dataset not loaded, proceeding without duplicate ID filtering")
        
        # Check for duplicates within the test dataset itself
        if 'test' in self.datasets:
            test_df = self.datasets['test']
            
            # Find all decision IDs that appear more than once in the test dataset
            duplicate_test_ids = test_df[test_df['decisionID'].duplicated(keep=False)]['decisionID'].unique()
            logger.info(f"Found {len(duplicate_test_ids)} duplicated decision IDs within the test dataset")
            
            # Combine with duplicated case_text IDs
            all_duplicate_ids = duplicated_case_text_ids.union(set(duplicate_test_ids))
            logger.info(f"Total unique duplicated IDs to remove: {len(all_duplicate_ids)}")
        else:
            all_duplicate_ids = duplicated_case_text_ids
        
        # Process each dataset to remove records with duplicated IDs
        for dataset_name in ['train', 'test', 'all_sentences', 'determination', 'case_cover']:
            if dataset_name in self.datasets:
                df = self.datasets[dataset_name]
                
                if 'decisionID' not in df.columns:
                    logger.warning(f"decisionID column not found in {dataset_name} dataset, skipping")
                    continue
                
                # Remove records with duplicated IDs
                clean_df = df[~df['decisionID'].isin(all_duplicate_ids)]
                unused_df = df[df['decisionID'].isin(all_duplicate_ids)]
                
                original_count = len(df)
                clean_count = len(clean_df)
                removed_count = len(unused_df)
                
                logger.info(f"Clean {dataset_name} dataset: {clean_count} rows (removed {removed_count} rows with duplicated IDs)")
                
                clean_datasets[dataset_name] = clean_df
                unused_data[dataset_name] = unused_df
        
        return clean_datasets, unused_data
    
    def save_datasets(self, clean_datasets: Dict[str, pd.DataFrame], unused_data: Dict[str, pd.DataFrame]) -> None:
        """
        Save the cleaned and unused datasets.
        
        Args:
            clean_datasets: Dictionary of cleaned datasets
            unused_data: Dictionary of unused data
        """
        # Save clean datasets
        for name, df in clean_datasets.items():
            output_path = self.clean_data_dir / f"{name}_clean.csv"
            pickle_path = self.clean_data_dir / f"{name}_clean.pkl"
            
            try:
                # Save as CSV
                df.to_csv(output_path, index=False)
                logger.info(f"Saved clean {name} dataset to {output_path}")
                
                # Save as pickle for faster loading
                df.to_pickle(pickle_path)
                logger.info(f"Saved clean {name} dataset to {pickle_path}")
            except Exception as e:
                logger.error(f"Error saving clean {name} dataset: {e}")
        
        # Save unused data
        for name, df in unused_data.items():
            output_path = self.unused_data_dir / f"{name}_unused.csv"
            pickle_path = self.unused_data_dir / f"{name}_unused.pkl"
            
            try:
                # Save as CSV
                df.to_csv(output_path, index=False)
                logger.info(f"Saved unused {name} data to {output_path}")
                
                # Save as pickle for faster loading
                df.to_pickle(pickle_path)
                logger.info(f"Saved unused {name} data to {pickle_path}")
            except Exception as e:
                logger.error(f"Error saving unused {name} data: {e}")
    
    def print_validation_summary(self, clean_datasets: Dict[str, pd.DataFrame], unused_data: Dict[str, pd.DataFrame]) -> None:
        """
        Print a validation summary of the datasets including shapes and column names.
        
        Args:
            clean_datasets: Dictionary of cleaned datasets
            unused_data: Dictionary of unused data
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        # Original datasets summary
        logger.info("\nORIGINAL DATASETS:")
        for name, df in self.datasets.items():
            logger.info(f"  {name}:")
            logger.info(f"    Shape: {df.shape}")
            logger.info(f"    Columns: {', '.join(df.columns)}")
            logger.info(f"    Unique decision IDs: {df['decisionID'].nunique()}")
        
        # Clean datasets summary
        logger.info("\nCLEAN DATASETS:")
        for name, df in clean_datasets.items():
            logger.info(f"  {name}_clean:")
            logger.info(f"    Shape: {df.shape}")
            logger.info(f"    Columns: {', '.join(df.columns)}")
            logger.info(f"    Unique decision IDs: {df['decisionID'].nunique()}")
        
        # Unused data summary
        logger.info("\nUNUSED DATA:")
        for name, df in unused_data.items():
            logger.info(f"  {name}_unused:")
            logger.info(f"    Shape: {df.shape}")
            logger.info(f"    Unique decision IDs: {df['decisionID'].nunique()}")
        
        # Summary of data reduction
        logger.info("\nDATA REDUCTION SUMMARY:")
        for name in clean_datasets.keys():
            if name in self.datasets and name in clean_datasets:
                original_rows = len(self.datasets[name])
                clean_rows = len(clean_datasets[name])
                reduction_percent = ((original_rows - clean_rows) / original_rows) * 100 if original_rows > 0 else 0
                
                logger.info(f"  {name}:")
                logger.info(f"    Original rows: {original_rows}")
                logger.info(f"    Clean rows: {clean_rows}")
                logger.info(f"    Reduction: {original_rows - clean_rows} rows ({reduction_percent:.2f}%)")
        
        logger.info("="*80)
    
    def run_pipeline(self) -> None:
        """Run the complete data cleaning pipeline."""
        logger.info("Starting data cleaning pipeline")
        
        # Load datasets
        self.load_datasets()
        
        # Clean datasets
        clean_datasets, unused_data = self.clean_datasets()
        
        # Print validation summary
        self.print_validation_summary(clean_datasets, unused_data)
        
        # Save datasets
        self.save_datasets(clean_datasets, unused_data)
        
        logger.info("Data cleaning pipeline completed")


def main():
    """Main function to run the data cleaning pipeline."""
    # Define paths - Fix the directory references
    # The script is in immigration_fine_tuning/data_processing/preprocessing/
    # The data is in immigration_fine_tuning/data/processed/
    
    # Get the absolute path to the project root
    base_dir = Path(__file__).resolve().parents[2]  # Go up 2 levels from preprocessing to immigration_fine_tuning
    
    # Define the processed data directory
    processed_data_dir = base_dir / "data" / "processed"
    
    # Define the output directories
    output_dir = base_dir / "data"
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create and run the data cleaner with explicit output directory
    cleaner = DataCleaner(processed_data_dir, output_dir)
    cleaner.run_pipeline()


if __name__ == "__main__":
    main()
