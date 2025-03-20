#!/usr/bin/env python3
"""
Determination Pipeline - An integrated framework for running multiple determination extractors.
"""

import os
import sys
import time
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import extractors
from rule_based.sparse_explicit_extraction import SparseExplicitExtractor
from rule_based.basic_determination_extraction import BasicDeterminationExtractor
from rule_based.ngram_determination_extraction import NgramDeterminationExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("determination_pipeline.log")
    ]
)
logger = logging.getLogger("DeterminationPipeline")

class DeterminationPipeline:
    """
    Pipeline for extracting determinations using different models in sequence.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the determination pipeline.
        
        Args:
            config: Configuration parameters for the pipeline
        """
        self.config = config or {}
        self.processors = {}
        self.performance_stats = {
            'total_time': 0,
            'processed_cases': 0,
            'processor_stats': {}
        }
        
        # Initialize processors based on configuration
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize processor models based on configuration."""
        # Always initialize the sparse extractor
        logger.info("Initializing SparseExplicitExtractor")
        self.processors['sparse'] = SparseExplicitExtractor()
        
        # Optionally initialize other processors based on config
        if self.config.get('use_basic_extractor', True):
            logger.info("Initializing BasicDeterminationExtractor")
            self.processors['basic'] = BasicDeterminationExtractor()
            
        if self.config.get('use_ngram_extractor', True):
            logger.info("Initializing NgramDeterminationExtractor")
            min_ngram = self.config.get('min_ngram_size', 2)
            max_ngram = self.config.get('max_ngram_size', 4)
            threshold = self.config.get('ngram_threshold', 0.75)
            self.processors['ngram'] = NgramDeterminationExtractor(
                min_ngram_size=min_ngram,
                max_ngram_size=max_ngram,
                match_threshold=threshold
            )
    
    def load_training_data(self, train_path: str, test_path: str, force_retrain: bool = False):
        """
        Load training data for all processors.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            force_retrain: Whether to force retraining
        """
        logger.info(f"Loading training data from {train_path} and {test_path}")
        
        for name, processor in self.processors.items():
            logger.info(f"Loading training data for {name} processor")
            processor.load_training_examples(train_path, test_path)
    
    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """
        Process the entire dataframe with all processors.
        
        Args:
            df: Input dataframe
            batch_size: Size of batches for processing
            
        Returns:
            Processed dataframe with extraction results
        """
        start_time = time.time()
        total_rows = len(df)
        logger.info(f"Processing {total_rows} records with batch size {batch_size}")
        
        # Process in batches
        results_dfs = []
        
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            logger.info(f"Processing batch {batch_start}-{batch_end}...")
            
            # Get batch
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            # Apply sparse extractor first (on cleaned_text)
            self._apply_sparse_extractor(batch_df)
            
            # Apply other extractors if configured
            if 'basic' in self.processors:
                self._apply_basic_extractor(batch_df)
                
            if 'ngram' in self.processors:
                self._apply_ngram_extractor(batch_df)
            
            results_dfs.append(batch_df)
        
        # Combine results
        result_df = pd.concat(results_dfs, ignore_index=False)
        
        # Update performance stats
        self.performance_stats['total_time'] = time.time() - start_time
        self.performance_stats['processed_cases'] = total_rows
        
        for name, processor in self.processors.items():
            self.performance_stats['processor_stats'][name] = processor.get_performance_stats()
        
        logger.info(f"Processed {total_rows} records in {self.performance_stats['total_time']:.2f}s")
        
        return result_df
    
    def _apply_sparse_extractor(self, df: pd.DataFrame):
        """Apply sparse extractor to dataframe."""
        processor = self.processors.get('sparse')
        if not processor:
            return
            
        logger.info("Applying sparse extractor...")
        
        # Process each row
        sparse_results = []
        
        for _, row in df.iterrows():
            text = row.get('cleaned_text', '')
            if pd.isna(text) or not text:
                sparse_results.append(None)
                continue
                
            result = processor.process_case(text)
            sparse_results.append(result.get('extracted_determinations', []))
        
        # Add results to dataframe
        df['sparse_explicit_extraction'] = sparse_results
    
    def _apply_basic_extractor(self, df: pd.DataFrame):
        """Apply basic extractor to dataframe."""
        processor = self.processors.get('basic')
        if not processor:
            return
            
        logger.info("Applying basic extractor...")
        
        # Process each row using relevant text sections
        basic_results = []
        
        for _, row in df.iterrows():
            # Combine relevant text sections
            text_sections = []
            
            # Add header sections when available
            for field in ['decision_headers_text', 'determination_headers_text', 
                         'analysis_headers_text', 'conclusion_headers_text']:
                if field in row and not pd.isna(row[field]):
                    text_sections.append(row[field])
            
            # Fall back to cleaned_text if no sections available
            if not text_sections and 'cleaned_text' in row and not pd.isna(row['cleaned_text']):
                text_sections.append(row['cleaned_text'])
            
            if not text_sections:
                basic_results.append(None)
                continue
                
            combined_text = "\n\n".join(text_sections)
            result = processor.process_case(combined_text)
            basic_results.append(result.get('extracted_determinations', []))
        
        # Add results to dataframe
        df['basic_determination_extraction'] = basic_results
    
    def _apply_ngram_extractor(self, df: pd.DataFrame):
        """Apply ngram extractor to dataframe."""
        processor = self.processors.get('ngram')
        if not processor:
            return
            
        logger.info("Applying ngram extractor...")
        
        # Process each row using relevant text sections
        ngram_results = []
        
        for _, row in df.iterrows():
            # Combine relevant text sections
            text_sections = []
            
            # Add header sections when available
            for field in ['decision_headers_text', 'determination_headers_text', 
                         'analysis_headers_text', 'conclusion_headers_text']:
                if field in row and not pd.isna(row[field]):
                    text_sections.append(row[field])
            
            # Fall back to cleaned_text if no sections available
            if not text_sections and 'cleaned_text' in row and not pd.isna(row['cleaned_text']):
                text_sections.append(row['cleaned_text'])
            
            if not text_sections:
                ngram_results.append(None)
                continue
                
            combined_text = "\n\n".join(text_sections)
            result = processor.process_case(combined_text)
            ngram_results.append(result.get('extracted_determinations', []))
        
        # Add results to dataframe
        df['ngram_determination_extraction'] = ngram_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pipeline."""
        return self.performance_stats


def run_simple_pipeline(input_file: str, output_file: str, 
                      train_path: Optional[str] = None, 
                      test_path: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None):
    """
    Run a simple pipeline that applies only the sparse extractor.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        train_path: Path to training data (optional)
        test_path: Path to test data (optional)
        config: Pipeline configuration (optional)
    """
    logger.info(f"Starting simple pipeline on {input_file}")
    
    # Create the pipeline with only sparse extractor
    pipeline_config = config or {'use_basic_extractor': False, 'use_ngram_extractor': False}
    pipeline = DeterminationPipeline(pipeline_config)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    # Set default training and test paths if not provided
    if train_path is None:
        train_path = str(Path(__file__).parent / "rule_based" / "training_data.csv")
        logger.info(f"Using default training path: {train_path}")
    
    if test_path is None:
        test_path = str(Path(__file__).parent / "rule_based" / "validation_set.csv")
        logger.info(f"Using default test path: {test_path}")
    
    # Load training data
    pipeline.load_training_data(train_path, test_path)
    
    # Process data
    logger.info("Processing data...")
    results_df = pipeline.process_dataframe(df)
    
    # Save results
    logger.info(f"Saving results to {output_file}")
    results_df.to_csv(output_file, index=False)
    
    # Log performance stats
    stats = pipeline.get_performance_stats()
    logger.info(f"Pipeline stats: Processed {stats['processed_cases']} records in {stats['total_time']:.2f}s")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run determination extraction pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--train", "-t", default=None, help="Training data path")
    parser.add_argument("--test", "-v", default=None, help="Test/validation data path")
    
    args = parser.parse_args()
    
    run_simple_pipeline(
        input_file=args.input,
        output_file=args.output,
        train_path=args.train,
        test_path=args.test
    ) 