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

from core.determination_pipeline import run_simple_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sparse_extraction.log")
    ]
)
logger = logging.getLogger("SparseExtraction")

def main():
    """Run sparse extraction pipeline on the validation dataset."""
    # Use:
    base_dir = Path(__file__).parent.parent.parent.parent  # go up to immigration_fine_tuning
    pipeline_dir = base_dir / "determination" / "pipeline"
    validation_path = pipeline_dir / "data" / "preprocessed_validation_set.csv"
    
    # Create pipeline_stages directory if it doesn't exist
    pipeline_stages_dir = pipeline_dir / "results" / "pipeline_stages"
    pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pipeline_stages_dir / f"sparse_extraction_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Set output file path in the new directory
    output_path = run_dir / "validation_with_sparse_extraction.csv"
    
    # Also save log file to the run directory
    file_handler = logging.FileHandler(run_dir / "sparse_extraction.log")
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
    logger.info(f"Processing validation set: {validation_path}")
    logger.info(f"Results will be saved to: {output_path}")
    
    # Run pipeline
    run_simple_pipeline(
        input_file=str(validation_path),
        output_file=str(output_path),
        train_path=str(train_path),
        test_path=str(test_path),
        config={'use_basic_extractor': False, 'use_ngram_extractor': False}
    )
    
    # Save a config file with information about this run
    config_info = {
        "run_timestamp": timestamp,
        "extractors_used": ["sparse_explicit_extraction"],
        "input_file": str(validation_path),
        "output_file": str(output_path),
        "train_data": str(train_path),
        "test_data": str(test_path)
    }
    
    import json
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)
    
    logger.info(f"Completed processing. Results saved to {output_path}")
    logger.info(f"Run information saved to {run_dir}")

if __name__ == "__main__":
    main() 

