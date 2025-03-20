#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Run n-gram extraction model on validation dataset.
"""

from pathlib import Path
import logging
import datetime
import pandas as pd
import json
import ast
import numpy as np

from core.determination_pipeline import DeterminationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("NgramExtraction")

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

def convert_to_serializable(obj):
    """Convert NumPy/pandas types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

class SectionBasedNgramExtractor:
    """Process specific text sections with the n-gram extractor and store counts."""
    
    def __init__(self, pipeline, min_score=15.0):
        """
        Initialize with a pipeline instance that has an n-gram extractor.
        
        Args:
            pipeline: DeterminationPipeline instance
            min_score: Minimum score threshold for extractions (default: 15.0)
        """
        self.pipeline = pipeline
        self.ngram_extractor = pipeline.processors.get('ngram')
        self.min_score = min_score
        
        logger.info(f"Using minimum score threshold: {self.min_score}")
        
        if not self.ngram_extractor:
            raise ValueError("N-gram extractor not initialized in pipeline")
    
    def filter_extractions(self, extractions):
        """Filter extractions to keep only those with scores >= min_score."""
        if not extractions or not isinstance(extractions, list):
            return []
            
        return [ext for ext in extractions if ext.get('score', 0) >= self.min_score]
    
    def process_dataframe(self, df):
        """Process dataframe with section-specific extraction."""
        logger.info("Processing with SectionBasedNgramExtractor...")
        
        # Parse previous extraction columns
        for col_name in ['sparse_explicit_extraction', 'decision_basic_extraction', 
                        'analysis_basic_extraction', 'reasons_basic_extraction', 
                        'conclusion_basic_extraction']:
            if col_name in df.columns:
                logger.info(f"Parsing column: {col_name}")
                df[col_name] = df[col_name].apply(parse_list_column)
                df[f"{col_name}_count"] = df[col_name].apply(len)
        
        # Define the sections to process
        sections = [
            'decision_headers_text', 
            'analysis_headers_text', 
            'reasons_headers_text',
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
            output_column = f"{section_key}_ngram_extraction"
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
                result = self.ngram_extractor.process_case(text)
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


def find_latest_basic_run():
    """Find the latest basic extraction run directory."""
    # Use relative path from the script location
    base_dir = Path(__file__).parent.parent.parent.parent
    pipeline_stages_dir = base_dir / "determination" / "pipeline" / "results" / "pipeline_stages"
    
    if not pipeline_stages_dir.exists():
        return None
    
    # Find directories that start with basic_extraction
    basic_runs = [d for d in pipeline_stages_dir.iterdir() 
                 if d.is_dir() and d.name.startswith("basic_extraction_")]
    
    if not basic_runs:
        return None
    
    # Sort by name (which includes timestamp) and return latest
    return sorted(basic_runs)[-1]


def main():
    """Run n-gram extraction pipeline on the validation dataset."""
    # Find latest basic extraction run
    latest_basic_run = find_latest_basic_run()
    if not latest_basic_run:
        logger.error("No previous basic extraction run found. Please run basic extraction first.")
        return
        
    logger.info(f"Found latest basic extraction run: {latest_basic_run}")
    
    # Find basic extraction output file
    basic_output_files = list(latest_basic_run.glob("validation_with_basic_extraction.csv"))
    if not basic_output_files:
        logger.error(f"No basic extraction output file found in {latest_basic_run}")
        return
    
    input_file = basic_output_files[0]
    logger.info(f"Using basic extraction results: {input_file}")
    
    # Define base paths using relative paths
    base_dir = Path(__file__).parent.parent.parent.parent
    pipeline_dir = base_dir / "determination" / "pipeline"
    
    # Create pipeline_stages directory if it doesn't exist
    pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
    pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pipeline_stages_dir / f"ngram_extraction_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Set output file path in the new directory
    output_path = run_dir / "validation_with_ngram_extraction.csv"
    
    # Add log file to the run directory
    file_handler = logging.FileHandler(run_dir / "ngram_extraction.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Find training data paths
    train_path = base_dir / "data" / "merged" / "train_enriched.csv"
    test_path = base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Remove the Windows-specific fallback paths
    if not train_path.exists():
        logger.warning(f"Train data not found at {train_path}")
        return
    
    logger.info(f"Using training data: {train_path}")
    logger.info(f"Using test data: {test_path}")
    logger.info(f"Results will be saved to: {output_path}")
    
    # Load data from previous stage
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")
    
    # Create pipeline with only ngram extractor
    config = {
        'use_basic_extractor': False, 
        'use_ngram_extractor': True,
        'min_ngram_size': 2,
        'max_ngram_size': 4,
        'ngram_threshold': 0.65  # Default threshold in the model
    }
    pipeline = DeterminationPipeline(config)
    
    # Load training data
    pipeline.load_training_data(str(train_path), str(test_path))
    
    # Create section processor with score threshold of 15.0
    # N-gram scores are typically higher than basic scores
    section_processor = SectionBasedNgramExtractor(pipeline, min_score=15.0)
    
    # Process data
    logger.info("Processing data with section-based ngram extractor...")
    results_df, stats = section_processor.process_dataframe(df)
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Save a config file with information about this run
    config_info = {
        "run_timestamp": timestamp,
        "previous_stage": str(latest_basic_run),
        "extractors_used": ["ngram_determination_extraction"],
        "min_score_threshold": 15.0,
        "ngram_config": {
            "min_ngram_size": config['min_ngram_size'],
            "max_ngram_size": config['max_ngram_size'],
            "match_threshold": config['ngram_threshold']
        },
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
    
    # Create summary of all extractors
    extractor_summary = {
        "sparse": {"total": 0, "docs_with_extractions": 0},
        "basic": {section: {"total": 0, "docs_with_extractions": 0} for section in ['decision', 'analysis', 'reasons', 'conclusion']},
        "ngram": {section: {"total": 0, "docs_with_extractions": 0} for section in ['decision', 'analysis', 'reasons', 'conclusion']}
    }
    
    # Calculate sparse stats
    if 'sparse_explicit_extraction_count' in results_df.columns:
        extractor_summary["sparse"]["total"] = results_df['sparse_explicit_extraction_count'].sum()
        extractor_summary["sparse"]["docs_with_extractions"] = (results_df['sparse_explicit_extraction_count'] > 0).sum()
        
        logger.info(f"  Sparse explicit extraction:")
        logger.info(f"    Total extractions: {extractor_summary['sparse']['total']}")
        logger.info(f"    Documents with extractions: {extractor_summary['sparse']['docs_with_extractions']} "
                  f"({extractor_summary['sparse']['docs_with_extractions']/len(results_df):.1%})")
    
    # Calculate basic stats
    for section in ['decision', 'analysis', 'reasons', 'conclusion']:
        basic_count_col = f"{section}_basic_extraction_count"
        if basic_count_col in results_df.columns:
            extractor_summary["basic"][section]["total"] = results_df[basic_count_col].sum()
            extractor_summary["basic"][section]["docs_with_extractions"] = (results_df[basic_count_col] > 0).sum()
            
            logger.info(f"  {section.capitalize()} basic extraction:")
            logger.info(f"    Total extractions: {extractor_summary['basic'][section]['total']}")
            logger.info(f"    Documents with extractions: {extractor_summary['basic'][section]['docs_with_extractions']} "
                      f"({extractor_summary['basic'][section]['docs_with_extractions']/len(results_df):.1%})")
    
    # Calculate ngram stats
    for section in ['decision', 'analysis', 'reasons', 'conclusion']:
        ngram_count_col = f"{section}_ngram_extraction_count"
        ngram_raw_count_col = f"{section}_ngram_extraction_raw_count"
        
        if ngram_count_col in results_df.columns and ngram_raw_count_col in results_df.columns:
            extractor_summary["ngram"][section]["total"] = results_df[ngram_count_col].sum()
            extractor_summary["ngram"][section]["total_raw"] = results_df[ngram_raw_count_col].sum()
            extractor_summary["ngram"][section]["docs_with_extractions"] = (results_df[ngram_count_col] > 0).sum()
            
            logger.info(f"  {section.capitalize()} ngram extraction:")
            logger.info(f"    Raw extractions: {extractor_summary['ngram'][section]['total_raw']}")
            logger.info(f"    Filtered extractions: {extractor_summary['ngram'][section]['total']}")
            logger.info(f"    Documents with extractions: {extractor_summary['ngram'][section]['docs_with_extractions']} "
                      f"({extractor_summary['ngram'][section]['docs_with_extractions']/len(results_df):.1%})")
    
    # Add summary to config
    config_info["extraction_summary"] = extractor_summary
    
    # Convert all values to serializable Python types
    serializable_config = convert_to_serializable(config_info)
    
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    logger.info(f"Completed processing. Results saved to {output_path}")
    logger.info(f"Run information saved to {run_dir}")

if __name__ == "__main__":
    main() 

