#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from typing import Dict, Any, List, Optional, Union

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
    
    def process_dataframe(self, df, sections=None):
        """
        Process dataframe with section-specific extraction.
        
        Args:
            df: Input dataframe
            sections: List of text sections to process (default: standard sections)
        """
        logger.info("Processing with SectionBasedNgramExtractor...")
        
        # Parse previous extraction columns
        for col_name in ['sparse_explicit_extraction', 'decision_basic_extraction', 
                        'analysis_basic_extraction', 'reasons_basic_extraction', 
                        'conclusion_basic_extraction']:
            if col_name in df.columns:
                logger.info(f"Parsing column: {col_name}")
                df[col_name] = df[col_name].apply(parse_list_column)
                df[f"{col_name}_count"] = df[col_name].apply(len)
        
        # Use default sections if none provided
        if sections is None:
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

class NgramExtractionRunner:
    """Runner for the n-gram extraction pipeline stage."""
    
    def find_latest_basic_run(self):
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
    
    def run(self, input_file: Union[str, Path], train_data: Union[str, Path], 
            test_data: Union[str, Path], timestamp: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None, sections: Optional[List[str]] = None,
            min_score: float = 15.0, min_ngram_size: int = 2, max_ngram_size: int = 4,
            ngram_threshold: float = 0.65) -> Path:
        """
        Run n-gram extraction pipeline.
        
        Args:
            input_file: Path to input file 
            train_data: Path to training data
            test_data: Path to test data
            timestamp: Timestamp for this run (default: generate new)
            config: Configuration dictionary
            sections: List of sections to process
            min_score: Minimum score threshold for extractions
            min_ngram_size: Minimum n-gram size
            max_ngram_size: Maximum n-gram size
            ngram_threshold: N-gram threshold
            
        Returns:
            Path to output file
        """
        # Convert paths to Path objects
        input_file = Path(input_file)
        train_data = Path(train_data)
        test_data = Path(test_data)
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config = config or {}
        
        # Extract configuration values with fallbacks to parameters
        min_score = config.get('min_score', min_score)
        min_ngram_size = config.get('min_ngram_size', min_ngram_size)
        max_ngram_size = config.get('max_ngram_size', max_ngram_size)
        ngram_threshold = config.get('ngram_threshold', ngram_threshold)
        
        # Set default sections if none provided
        if sections is None:
            sections = [
                'decision_headers_text', 
                'analysis_headers_text', 
                'reasons_headers_text',
                'conclusion_headers_text'
            ]
        
        # Define base paths
        base_dir = Path(__file__).parent.parent.parent.parent
        pipeline_dir = base_dir / "determination" / "pipeline"
        
        # Create pipeline_stages directory if it doesn't exist
        pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
        pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamped folder for this run
        run_dir = pipeline_stages_dir / f"ngram_extraction_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Set output file path in the new directory
        output_path = run_dir / "validation_with_ngram_extraction.csv"
        
        # Add log file to the run directory
        file_handler = logging.FileHandler(run_dir / "ngram_extraction.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Find input file - either directly specified or from latest basic run
        previous_stage = None
        if not input_file.exists():
            latest_basic_run = self.find_latest_basic_run()
            if latest_basic_run:
                previous_stage = str(latest_basic_run)
                basic_output_files = list(latest_basic_run.glob("validation_with_basic_extraction.csv"))
                if basic_output_files:
                    input_file = basic_output_files[0]
                    logger.info(f"Using basic extraction results: {input_file}")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return None
            
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Using training data: {train_data}")
        logger.info(f"Using test data: {test_data}")
        logger.info(f"Results will be saved to: {output_path}")
        logger.info(f"Processing sections: {sections}")
        logger.info(f"Using minimum score threshold: {min_score}")
        logger.info(f"Using n-gram configuration: min_size={min_ngram_size}, max_size={max_ngram_size}, threshold={ngram_threshold}")
        
        # Load data from previous stage
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Create pipeline with only ngram extractor
        pipeline_config = {
            'use_basic_extractor': False, 
            'use_ngram_extractor': True,
            'min_ngram_size': min_ngram_size,
            'max_ngram_size': max_ngram_size,
            'ngram_threshold': ngram_threshold
        }
        pipeline_config.update(config.get('pipeline_config', {}))
        
        pipeline = DeterminationPipeline(pipeline_config)
        
        # Load training data
        pipeline.load_training_data(str(train_data), str(test_data))
        
        # Create section processor with configured score threshold
        section_processor = SectionBasedNgramExtractor(pipeline, min_score=min_score)
        
        # Process data with configured sections
        logger.info("Processing data with section-based ngram extractor...")
        results_df, stats = section_processor.process_dataframe(df, sections=sections)
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        results_df.to_csv(output_path, index=False)
        
        # Save a config file with information about this run
        config_info = {
            "run_timestamp": timestamp,
            "previous_stage": previous_stage,
            "extractors_used": ["ngram_determination_extraction"],
            "min_score_threshold": min_score,
            "ngram_config": {
                "min_ngram_size": min_ngram_size,
                "max_ngram_size": max_ngram_size,
                "match_threshold": ngram_threshold
            },
            "sections_processed": sections,
            "filtering_stats": stats,
            "input_file": str(input_file),
            "output_file": str(output_path),
            "train_data": str(train_data),
            "test_data": str(test_data),
            "config": config
        }
        
        # Create summary of all extractors
        extractor_summary = {
            "sparse": {"total": 0, "docs_with_extractions": 0},
            "basic": {section.replace('_headers_text', ''): {"total": 0, "docs_with_extractions": 0} 
                     for section in sections},
            "ngram": {section.replace('_headers_text', ''): {"total": 0, "docs_with_extractions": 0} 
                     for section in sections}
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
        for section in [s.replace('_headers_text', '') for s in sections]:
            basic_count_col = f"{section}_basic_extraction_count"
            if basic_count_col in results_df.columns:
                extractor_summary["basic"][section]["total"] = results_df[basic_count_col].sum()
                extractor_summary["basic"][section]["docs_with_extractions"] = (results_df[basic_count_col] > 0).sum()
                
                logger.info(f"  {section.capitalize()} basic extraction:")
                logger.info(f"    Total extractions: {extractor_summary['basic'][section]['total']}")
                logger.info(f"    Documents with extractions: {extractor_summary['basic'][section]['docs_with_extractions']} "
                          f"({extractor_summary['basic'][section]['docs_with_extractions']/len(results_df):.1%})")
        
        # Calculate ngram stats
        for section in [s.replace('_headers_text', '') for s in sections]:
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
        
        # Remove the file handler to avoid duplicate logs in future pipeline stages
        logger.removeHandler(file_handler)
        
        logger.info(f"Completed processing. Results saved to {output_path}")
        logger.info(f"Run information saved to {run_dir}")
        
        return output_path

def main():
    """Run n-gram extraction pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run n-gram extraction pipeline")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--train", "-t", help="Training data path")
    parser.add_argument("--test", "-e", help="Test data path")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    parser.add_argument("--min-score", "-m", type=float, default=15.0, help="Minimum score threshold")
    parser.add_argument("--min-ngram", type=int, default=2, help="Minimum n-gram size")
    parser.add_argument("--max-ngram", type=int, default=4, help="Maximum n-gram size")
    parser.add_argument("--threshold", type=float, default=0.65, help="N-gram threshold")
    parser.add_argument("--sections", "-s", help="Comma-separated list of sections to process")
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Set default paths if not provided
    base_dir = Path(__file__).parent.parent.parent.parent
    input_file = args.input or "auto"  # Will auto-detect from latest basic run
    train_data = args.train or base_dir / "data" / "merged" / "train_enriched.csv"
    test_data = args.test or base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Parse sections if provided
    sections = None
    if args.sections:
        sections = args.sections.split(",")
    
    # Update config with command line arguments
    if args.min_score is not None:
        config['min_score'] = args.min_score
    if args.min_ngram is not None:
        config['min_ngram_size'] = args.min_ngram
    if args.max_ngram is not None:
        config['max_ngram_size'] = args.max_ngram
    if args.threshold is not None:
        config['ngram_threshold'] = args.threshold
    
    # Run pipeline
    runner = NgramExtractionRunner()
    output_path = runner.run(
        input_file=input_file,
        train_data=train_data,
        test_data=test_data,
        config=config,
        sections=sections,
        min_score=args.min_score,
        min_ngram_size=args.min_ngram,
        max_ngram_size=args.max_ngram,
        ngram_threshold=args.threshold
    )
    
    if output_path:
        print(f"Output saved to: {output_path}")
    else:
        print("Pipeline execution failed")

if __name__ == "__main__":
    main() 

