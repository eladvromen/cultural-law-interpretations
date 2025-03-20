#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Run basic extraction model on validation dataset.
"""

import os
import sys
from pathlib import Path
import logging
import datetime
import pandas as pd
import json
import ast

from core.determination_pipeline import DeterminationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("BasicExtraction")

def parse_list_column(value):
    """Parse a column value that contains a list representation."""
    if pd.isna(value):
        return []
    
    if isinstance(value, list):
        return value
    
    # Try to parse as a list
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (SyntaxError, ValueError):
        return []

class SectionBasedBasicExtractor:
    """Process specific text sections with the basic extractor and store counts."""
    
    def __init__(self, pipeline, min_score=5.0):
        """
        Initialize with a pipeline instance that has a basic extractor.
        
        Args:
            pipeline: DeterminationPipeline instance
            min_score: Minimum score threshold for extractions (default: 5.0)
        """
        self.pipeline = pipeline
        self.basic_extractor = pipeline.processors.get('basic')
        self.min_score = min_score
        
        logger.info(f"Using minimum score threshold: {self.min_score}")
        
        if not self.basic_extractor:
            raise ValueError("Basic extractor not initialized in pipeline")
    
    def filter_extractions(self, extractions):
        """Filter extractions to keep only those with scores >= min_score."""
        if not extractions or not isinstance(extractions, list):
            return []
            
        return [ext for ext in extractions if ext.get('score', 0) >= self.min_score]
    
    def process_dataframe(self, df):
        """Process dataframe with section-specific extraction."""
        # First get the latest sparse extraction results if present
        logger.info("Processing with SectionBasedBasicExtractor...")
        
        # Parse sparse extraction column if it exists
        if 'sparse_explicit_extraction' in df.columns:
            logger.info("Parsing sparse extraction column...")
            # Parse the sparse extraction column
            df['sparse_explicit_extraction'] = df['sparse_explicit_extraction'].apply(parse_list_column)
            
            # Add count column
            df['sparse_explicit_extraction_count'] = df['sparse_explicit_extraction'].apply(len)
            
            # Log basic statistics
            total_extractions = df['sparse_explicit_extraction_count'].sum()
            docs_with_extractions = (df['sparse_explicit_extraction_count'] > 0).sum()
            logger.info(f"Found {total_extractions} sparse extractions in {docs_with_extractions} documents")
        
        # Define the sections to process
        sections = [
            'decision_headers_text', 
            'analysis_headers_text', 
            'reasons_headers_text',  # Note: corrected from 'reasoning_headers_text'
            'conclusion_headers_text'
        ]
        
        # Track extraction statistics
        stats = {
            'total_extractions': 0,
            'filtered_extractions': 0,
            'section_stats': {}
        }
        
        # Process each section
        for section in sections:
            if section not in df.columns:
                logger.warning(f"Section {section} not found in dataframe, skipping")
                continue
                
            logger.info(f"Processing section: {section}")
            section_key = section.replace('_headers_text', '')
            
            # Initialize section stats
            stats['section_stats'][section_key] = {
                'total': 0,
                'filtered': 0,
                'documents_with_extractions': 0
            }
            
            # Process section
            output_column = f"{section_key}_basic_extraction"
            count_column = f"{output_column}_count"
            raw_count_column = f"{output_column}_raw_count"
            
            # Apply extractor to each row
            section_results = []
            raw_counts = []
            
            for _, row in df.iterrows():
                text = row.get(section, '')
                if pd.isna(text) or not text:
                    section_results.append([])
                    raw_counts.append(0)
                    continue
                    
                # Get extractions
                result = self.basic_extractor.process_case(text)
                raw_extractions = result.get('extracted_determinations', [])
                
                # Count before filtering
                raw_count = len(raw_extractions)
                raw_counts.append(raw_count)
                stats['total_extractions'] += raw_count
                stats['section_stats'][section_key]['total'] += raw_count
                
                # Apply score filtering
                filtered_extractions = self.filter_extractions(raw_extractions)
                
                # Count after filtering
                filtered_count = len(filtered_extractions)
                stats['filtered_extractions'] += filtered_count
                stats['section_stats'][section_key]['filtered'] += filtered_count
                
                # Track documents with extractions
                if filtered_count > 0:
                    stats['section_stats'][section_key]['documents_with_extractions'] += 1
                
                section_results.append(filtered_extractions)
            
            # Add results and counts to dataframe
            df[output_column] = section_results
            df[count_column] = df[output_column].apply(len)
            df[raw_count_column] = raw_counts
        
        # Log filtering statistics
        if stats['total_extractions'] > 0:
            retention_rate = stats['filtered_extractions'] / stats['total_extractions']
            logger.info(f"Extraction filtering statistics:")
            logger.info(f"  Total extractions: {stats['total_extractions']}")
            logger.info(f"  Retained after filtering: {stats['filtered_extractions']} ({retention_rate:.1%})")
            
            for section_key, section_stats in stats['section_stats'].items():
                if section_stats['total'] > 0:
                    section_retention = section_stats['filtered'] / section_stats['total']
                    logger.info(f"  {section_key.capitalize()} section:")
                    logger.info(f"    Total extractions: {section_stats['total']}")
                    logger.info(f"    Retained after filtering: {section_stats['filtered']} ({section_retention:.1%})")
                    logger.info(f"    Documents with extractions: {section_stats['documents_with_extractions']}")
        
        return df, stats


def find_latest_sparse_run():
    """Find the latest sparse extraction run directory."""
    base_dir = Path(__file__).parent.parent.parent.parent  # go up to immigration_fine_tuning
    pipeline_dir = base_dir / "determination" / "pipeline"
    pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
    
    if not pipeline_stages_dir.exists():
        return None
    
    # Find directories that start with sparse_extraction
    sparse_runs = [d for d in pipeline_stages_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("sparse_extraction_")]
    
    if not sparse_runs:
        return None
    
    # Sort by name (which includes timestamp) and return latest
    return sorted(sparse_runs)[-1]


def main():
    """Run basic extraction pipeline on the validation dataset."""
    # Find latest sparse extraction run
    latest_sparse_run = find_latest_sparse_run()
    if not latest_sparse_run:
        logger.error("No previous sparse extraction run found. Please run sparse extraction first.")
        return
        
    logger.info(f"Found latest sparse extraction run: {latest_sparse_run}")
    
    # Find sparse extraction output file
    sparse_output_files = list(latest_sparse_run.glob("validation_with_sparse_extraction.csv"))
    if not sparse_output_files:
        logger.error(f"No sparse extraction output file found in {latest_sparse_run}")
        return
    
    input_file = sparse_output_files[0]
    logger.info(f"Using sparse extraction results: {input_file}")
    
    # Define base paths
    base_dir = Path(__file__).parent.parent.parent.parent  # go up to immigration_fine_tuning
    pipeline_dir = base_dir / "determination" / "pipeline"
    
    # Create pipeline_stages directory if it doesn't exist
    pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
    pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pipeline_stages_dir / f"basic_extraction_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Set output file path in the new directory
    output_path = run_dir / "validation_with_basic_extraction.csv"
    
    # Add log file to the run directory
    file_handler = logging.FileHandler(run_dir / "basic_extraction.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Find training data paths
    train_path = base_dir / "data" / "merged" / "train_enriched.csv"
    test_path = base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Check if paths exist
    if not train_path.exists():
        logger.warning(f"Train data not found at {train_path}")
        # Try windows-specific path
        train_path = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/train_enriched.csv")
        test_path = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/test_enriched.csv")
    
    logger.info(f"Using training data: {train_path}")
    logger.info(f"Using test data: {test_path}")
    logger.info(f"Results will be saved to: {output_path}")
    
    # Load data from previous stage
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    # Create pipeline with only basic extractor
    config = {'use_basic_extractor': True, 'use_ngram_extractor': False}
    pipeline = DeterminationPipeline(config)
    
    # Load training data
    pipeline.load_training_data(str(train_path), str(test_path))
    
    # Create section processor with score threshold of 5.0
    section_processor = SectionBasedBasicExtractor(pipeline, min_score=5.0)
    
    # Process data
    logger.info("Processing data with section-based basic extractor...")
    results_df, stats = section_processor.process_dataframe(df)
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Save a config file with information about this run
    config_info = {
        "run_timestamp": timestamp,
        "previous_stage": str(latest_sparse_run),
        "extractors_used": ["basic_determination_extraction"],
        "min_score_threshold": 5.0,
        "sections_processed": [
            "decision_headers_text", 
            "analysis_headers_text", 
            "reasons_headers_text", 
            "conclusion_headers_text"
        ],
        "filtering_stats": stats,
        "input_file": str(input_file),
        "output_file": str(output_path),
        "train_data": str(train_path),
        "test_data": str(test_path)
    }
    
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Log summary of extraction counts
    logger.info("Extraction count summary:")
    if 'sparse_explicit_extraction_count' in results_df.columns:
        total_sparse = results_df['sparse_explicit_extraction_count'].sum()
        avg_sparse = results_df['sparse_explicit_extraction_count'].mean()
        docs_with_sparse = (results_df['sparse_explicit_extraction_count'] > 0).sum()
        
        logger.info(f"  Sparse explicit extraction:")
        logger.info(f"    Total extractions: {total_sparse}")
        logger.info(f"    Avg per document: {avg_sparse:.2f}")
        logger.info(f"    Documents with extractions: {docs_with_sparse} ({docs_with_sparse/len(results_df):.1%})")
    
    for section in ['decision', 'analysis', 'reasons', 'conclusion']:
        count_col = f"{section}_basic_extraction_count"
        raw_count_col = f"{section}_basic_extraction_raw_count"
        
        if count_col in results_df.columns and raw_count_col in results_df.columns:
            total_count = results_df[count_col].sum()
            raw_count = results_df[raw_count_col].sum()
            avg_count = results_df[count_col].mean()
            docs_with_extractions = (results_df[count_col] > 0).sum()
            
            logger.info(f"  {section.capitalize()} basic extraction:")
            logger.info(f"    Raw extractions: {raw_count}")
            logger.info(f"    Filtered extractions: {total_count}")
            logger.info(f"    Avg per document: {avg_count:.2f}")
            logger.info(f"    Documents with extractions: {docs_with_extractions} ({docs_with_extractions/len(results_df):.1%})")
    
    logger.info(f"Completed processing. Results saved to {output_path}")
    logger.info(f"Run information saved to {run_dir}")

if __name__ == "__main__":
    main() 

