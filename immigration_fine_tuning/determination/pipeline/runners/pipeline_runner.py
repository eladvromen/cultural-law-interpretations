#!/usr/bin/env python3
"""
Pipeline runner that executes all processing steps in sequence:
1. Text Processing
2. Sparse Extraction
3. Basic Extraction
4. N-gram Extraction
"""

import os
import sys
from pathlib import Path
import logging
import datetime
import time

# Import runners
from text_processing import process_validation_set
from run_sparse_extraction import main as run_sparse_extraction
from run_basic_extraction import main as run_basic_extraction
from run_ngram_extraction import main as run_ngram_extraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("PipelineRunner")

def setup_run_directory() -> tuple[Path, Path]:
    """Create timestamped run directory and setup logging."""
    # Define base paths using relative paths
    base_dir = Path(__file__).parent.parent.parent.parent
    pipeline_dir = base_dir / "determination" / "pipeline"
    
    # Create pipeline_runs directory if it doesn't exist
    pipeline_runs_dir = pipeline_dir / "results" / "pipeline_runs"
    pipeline_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pipeline_runs_dir / f"full_pipeline_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Add log file to the run directory
    log_path = run_dir / "pipeline_run.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return base_dir, run_dir

def run_pipeline():
    """Run the complete pipeline in sequence."""
    start_time = time.time()
    logger.info("Starting full pipeline run")
    
    try:
        # Setup run directory
        base_dir, run_dir = setup_run_directory()
        logger.info(f"Pipeline run directory: {run_dir}")
        
        # Step 1: Text Processing
        logger.info("\n=== Starting Text Processing ===")
        validation_path = base_dir / "determination" / "pipeline" / "data" / "validation_set.csv"
        preprocessed_path = base_dir / "determination" / "pipeline" / "data" / "preprocessed_validation_set.csv"
        
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation file not found at {validation_path}")
        
        process_validation_set(validation_path, preprocessed_path, sample_size=0)
        logger.info("Text processing completed")
        
        # Step 2: Sparse Extraction
        logger.info("\n=== Starting Sparse Extraction ===")
        run_sparse_extraction()
        logger.info("Sparse extraction completed")
        
        # Step 3: Basic Extraction
        logger.info("\n=== Starting Basic Extraction ===")
        run_basic_extraction()
        logger.info("Basic extraction completed")
        
        # Step 4: N-gram Extraction
        logger.info("\n=== Starting N-gram Extraction ===")
        run_ngram_extraction()
        logger.info("N-gram extraction completed")
        
        # Calculate total runtime
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logger.info("\n=== Pipeline Run Complete ===")
        logger.info(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logger.info(f"Results saved in: {run_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point."""
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 