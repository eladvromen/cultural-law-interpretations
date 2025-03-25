#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Run sparse extraction model on validation dataset.
"""

from pathlib import Path
import logging
import datetime
import json
from typing import Dict, Any, Optional, Union

from core.determination_pipeline import run_simple_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("SparseExtraction")

class SparseExtractionRunner:
    """Runner for the sparse extraction pipeline stage."""
    
    def run(self, input_file: Union[str, Path], train_data: Union[str, Path], 
            test_data: Union[str, Path], timestamp: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Run sparse extraction pipeline.
        
        Args:
            input_file: Path to input file
            train_data: Path to training data
            test_data: Path to test data
            timestamp: Timestamp for this run (default: generate new)
            config: Configuration dictionary
            
        Returns:
            Path to output file
        """
        # Handle inputs
        input_file = Path(input_file)
        train_data = Path(train_data)
        test_data = Path(test_data)
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config = config or {}
        
        # Define base paths
        base_dir = Path(__file__).parent.parent.parent.parent
        pipeline_dir = base_dir / "determination" / "pipeline"
        
        # Create pipeline_stages directory if it doesn't exist
        pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
        pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamped folder for this run
        run_dir = pipeline_stages_dir / f"sparse_extraction_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Set output file path in the new directory
        output_path = run_dir / "validation_with_sparse_extraction.csv"
        
        # Save log file to the run directory
        file_handler = logging.FileHandler(run_dir / "sparse_extraction.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Using training data: {train_data}")
        logger.info(f"Using test data: {test_data}")
        logger.info(f"Processing validation set: {input_file}")
        logger.info(f"Results will be saved to: {output_path}")
        
        # Configure pipeline
        pipeline_config = {'use_basic_extractor': False, 'use_ngram_extractor': False}
        pipeline_config.update(config.get('pipeline_config', {}))
        
        # Run pipeline
        run_simple_pipeline(
            input_file=str(input_file),
            output_file=str(output_path),
            train_path=str(train_data),
            test_path=str(test_data),
            config=pipeline_config
        )
        
        # Save a config file with information about this run
        run_config = {
            "run_timestamp": timestamp,
            "extractors_used": ["sparse_explicit_extraction"],
            "input_file": str(input_file),
            "output_file": str(output_path),
            "train_data": str(train_data),
            "test_data": str(test_data),
            "config": config
        }
        
        with open(run_dir / "run_config.json", 'w') as f:
            json.dump(run_config, f, indent=2)
        
        logger.info(f"Completed processing. Results saved to {output_path}")
        logger.info(f"Run information saved to {run_dir}")
        
        # Remove the file handler to avoid duplicate logs in future pipeline stages
        logger.removeHandler(file_handler)
        
        return output_path

def main():
    """Run sparse extraction pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sparse extraction pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--train", "-t", help="Training data path")
    parser.add_argument("--test", "-e", help="Test data path")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Set default paths if not provided
    base_dir = Path(__file__).parent.parent.parent.parent
    train_data = args.train or base_dir / "data" / "merged" / "train_enriched.csv"
    test_data = args.test or base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Run pipeline
    runner = SparseExtractionRunner()
    output_path = runner.run(
        input_file=args.input,
        train_data=train_data,
        test_data=test_data,
        config=config
    )
    
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main() 

