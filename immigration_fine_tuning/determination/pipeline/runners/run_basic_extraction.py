#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Run basic extraction model on validation dataset.
"""

from pathlib import Path
import logging
import datetime
import pandas as pd
import json
import ast
from typing import Dict, Any, List, Optional, Union

from core.determination_pipeline import DeterminationPipeline
from utils.text_cleaning import clean_text

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
    
    def process_dataframe(self, df, sections=None):
        """
        Process dataframe with section-specific extraction.
        
        Args:
            df: Input dataframe
            sections: List of text sections to process (default: standard sections)
        """
        # First get the latest sparse extraction results if present
        logger.info("Processing with SectionBasedBasicExtractor...")
        
        # Use default sections if none provided
        if sections is None:
            sections = [
                'decision_headers_text',
                'analysis_headers_text',
                'reasons_headers_text',
                'conclusion_headers_text'
            ]
        
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
                    
                # Clean text before processing
                cleaned_text = clean_text(text)
                    
                # Get extractions
                result = self.basic_extractor.process_case(cleaned_text)
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

class BasicExtractionRunner:
    """Runner for the basic extraction pipeline stage."""
    
    def find_latest_sparse_run(self):
        """Find the latest sparse extraction run directory."""
        base_dir = Path(__file__).parent.parent.parent.parent
        pipeline_stages_dir = base_dir / "determination" / "pipeline" / "results" / "pipeline_stages"
        
        if not pipeline_stages_dir.exists():
            return None
        
        # Find directories that start with sparse_extraction
        sparse_runs = [d for d in pipeline_stages_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("sparse_extraction_")]
        
        if not sparse_runs:
            return None
        
        # Sort by name (which includes timestamp) and return latest
        return sorted(sparse_runs)[-1]
    
    def run(self, input_file: Union[str, Path], train_data: Union[str, Path], 
            test_data: Union[str, Path], timestamp: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None, sections: Optional[List[str]] = None,
            min_score: float = 5.0) -> Path:
        """
        Run basic extraction pipeline.
        
        Args:
            input_file: Path to input file 
            train_data: Path to training data
            test_data: Path to test data
            timestamp: Timestamp for this run (default: generate new)
            config: Configuration dictionary
            sections: List of sections to process
            min_score: Minimum score threshold for extractions
            
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
        min_score = config.get('min_score', min_score)
        
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
        run_dir = pipeline_stages_dir / f"basic_extraction_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Set output file path in the new directory
        output_path = run_dir / "validation_with_basic_extraction.csv"
        
        # Add log file to the run directory
        file_handler = logging.FileHandler(run_dir / "basic_extraction.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Find input file - either directly specified or from latest sparse run
        previous_stage = None
        if not input_file.exists():
            latest_sparse_run = self.find_latest_sparse_run()
            if latest_sparse_run:
                previous_stage = str(latest_sparse_run)
                sparse_output_files = list(latest_sparse_run.glob("validation_with_sparse_extraction.csv"))
                if sparse_output_files:
                    input_file = sparse_output_files[0]
                    logger.info(f"Using sparse extraction results: {input_file}")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return None
            
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Using training data: {train_data}")
        logger.info(f"Using test data: {test_data}")
        logger.info(f"Results will be saved to: {output_path}")
        logger.info(f"Processing sections: {sections}")
        logger.info(f"Using minimum score threshold: {min_score}")
        
        # Load data from previous stage
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records")
        
        # Create pipeline with only basic extractor
        pipeline_config = {'use_basic_extractor': True, 'use_ngram_extractor': False}
        pipeline_config.update(config.get('pipeline_config', {}))
        
        pipeline = DeterminationPipeline(pipeline_config)
        
        # Load training data
        pipeline.load_training_data(str(train_data), str(test_data))
        
        # Create section processor with configured score threshold
        section_processor = SectionBasedBasicExtractor(pipeline, min_score=min_score)
        
        # Process data with configured sections
        logger.info("Processing data with section-based basic extractor...")
        results_df, stats = section_processor.process_dataframe(df, sections=sections)
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        results_df.to_csv(output_path, index=False)
        
        # Save a config file with information about this run
        run_config = {
            "run_timestamp": timestamp,
            "previous_stage": previous_stage,
            "extractors_used": ["basic_determination_extraction"],
            "min_score_threshold": min_score,
            "sections_processed": sections,
            "filtering_stats": stats,
            "input_file": str(input_file),
            "output_file": str(output_path),
            "train_data": str(train_data),
            "test_data": str(test_data),
            "config": config
        }
        
        with open(run_dir / "run_config.json", 'w') as f:
            json.dump(run_config, f, indent=2)
        
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
        
        for section in [s.replace('_headers_text', '') for s in sections]:
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
        
        # Remove the file handler to avoid duplicate logs in future pipeline stages
        logger.removeHandler(file_handler)
        
        logger.info(f"Completed processing. Results saved to {output_path}")
        logger.info(f"Run information saved to {run_dir}")
        
        return output_path

def main():
    """Run basic extraction pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run basic extraction pipeline")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--train", "-t", help="Training data path")
    parser.add_argument("--test", "-e", help="Test data path")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    parser.add_argument("--min-score", "-m", type=float, default=5.0, help="Minimum score threshold")
    parser.add_argument("--sections", "-s", help="Comma-separated list of sections to process")
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Set default paths if not provided
    base_dir = Path(__file__).parent.parent.parent.parent
    input_file = args.input or "auto"  # Will auto-detect from latest sparse run
    train_data = args.train or base_dir / "data" / "merged" / "train_enriched.csv"
    test_data = args.test or base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Parse sections if provided
    sections = None
    if args.sections:
        sections = args.sections.split(",")
    
    # Update config with command line arguments
    if args.min_score is not None:
        config['min_score'] = args.min_score
    
    # Run pipeline
    runner = BasicExtractionRunner()
    output_path = runner.run(
        input_file=input_file,
        train_data=train_data,
        test_data=test_data,
        config=config,
        sections=sections,
        min_score=args.min_score
    )
    
    if output_path:
        print(f"Output saved to: {output_path}")
    else:
        print("Pipeline execution failed")

if __name__ == "__main__":
    main() 

