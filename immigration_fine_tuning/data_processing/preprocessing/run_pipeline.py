import os
import sys
import logging
from pathlib import Path
import time
import importlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Runner for the complete preprocessing pipeline that executes all steps in order:
    1. Initial preprocessing (loading and converting datasets)
    2. Full text initial processing (processing case archives)
    3. Data cleaning (cleaning and filtering datasets)
    4. Merging dataframes (enriching datasets with metadata)
    """
    
    def __init__(self):
        # Get the absolute path to the project root
        self.base_dir = Path(__file__).resolve().parents[2]  # Go up 2 levels from preprocessing to immigration_fine_tuning
        
        # Define the data directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "asylex_data"
        self.processed_dir = self.data_dir / "processed"
        self.clean_data_dir = self.data_dir / "clean_data"
        self.merged_dir = self.data_dir / "merged"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.clean_data_dir.mkdir(exist_ok=True, parents=True)
        self.merged_dir.mkdir(exist_ok=True, parents=True)
        
        # Import modules dynamically
        self.modules = {}
        
    def import_modules(self):
        """Import all required modules for the pipeline."""
        logger.info("Importing preprocessing modules...")
        
        try:
            # Add the parent directory to sys.path to allow imports
            parent_dir = str(Path(__file__).resolve().parent.parent)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            # Import the modules
            import preprocessing.intial_preprocessing
            import preprocessing.full_text_initial_processing
            import preprocessing.data_cleaning
            import preprocessing.merging_dfs
            
            self.modules = {
                "initial_preprocessing": preprocessing.intial_preprocessing,
                "full_text_initial_processing": preprocessing.full_text_initial_processing,
                "data_cleaning": preprocessing.data_cleaning,
                "merging_dfs": preprocessing.merging_dfs
            }
            
            logger.info("Successfully imported all modules")
        except ImportError as e:
            logger.error(f"Error importing modules: {e}")
            raise
    
    def run_initial_preprocessing(self):
        """Run the initial preprocessing step."""
        logger.info("=" * 80)
        logger.info("STEP 1: INITIAL PREPROCESSING")
        logger.info("=" * 80)
        
        try:
            # Get the module
            module = self.modules["initial_preprocessing"]
            
            # Create DatasetProcessor and run it
            data_dir = str(self.raw_data_dir)
            output_dir = str(self.processed_dir)
            
            logger.info(f"Running initial preprocessing with data_dir={data_dir}, output_dir={output_dir}")
            
            # Call the main function from the module
            if hasattr(module, "main"):
                # Temporarily redirect sys.argv
                original_argv = sys.argv
                sys.argv = [sys.argv[0]]  # Keep only the script name
                
                # Run the main function
                module.main()
                
                # Restore sys.argv
                sys.argv = original_argv
            else:
                logger.warning("No main function found in initial_preprocessing module")
                
            logger.info("Initial preprocessing completed")
        except Exception as e:
            logger.error(f"Error in initial preprocessing: {e}", exc_info=True)
            raise
    
    def run_full_text_initial_processing(self):
        """Run the full text initial processing step."""
        logger.info("=" * 80)
        logger.info("STEP 2: FULL TEXT INITIAL PROCESSING")
        logger.info("=" * 80)
        
        try:
            # Get the module
            module = self.modules["full_text_initial_processing"]
            
            logger.info("Running full text initial processing")
            
            # Call the main function from the module
            if hasattr(module, "main"):
                # Temporarily redirect sys.argv
                original_argv = sys.argv
                sys.argv = [sys.argv[0]]  # Keep only the script name
                
                # Run the main function
                module.main()
                
                # Restore sys.argv
                sys.argv = original_argv
            else:
                logger.warning("No main function found in full_text_initial_processing module")
                
            logger.info("Full text initial processing completed")
        except Exception as e:
            logger.error(f"Error in full text initial processing: {e}", exc_info=True)
            raise
    
    def run_data_cleaning(self):
        """Run the data cleaning step."""
        logger.info("=" * 80)
        logger.info("STEP 3: DATA CLEANING")
        logger.info("=" * 80)
        
        try:
            # Get the module
            module = self.modules["data_cleaning"]
            
            logger.info("Running data cleaning")
            
            # Call the main function from the module
            if hasattr(module, "main"):
                # Temporarily redirect sys.argv
                original_argv = sys.argv
                sys.argv = [sys.argv[0]]  # Keep only the script name
                
                # Run the main function
                module.main()
                
                # Restore sys.argv
                sys.argv = original_argv
            else:
                logger.warning("No main function found in data_cleaning module")
                
            logger.info("Data cleaning completed")
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}", exc_info=True)
            raise
    
    def run_merging_dfs(self):
        """Run the merging dataframes step."""
        logger.info("=" * 80)
        logger.info("STEP 4: MERGING DATAFRAMES")
        logger.info("=" * 80)
        
        try:
            # Get the module
            module = self.modules["merging_dfs"]
            
            logger.info("Running merging dataframes")
            
            # Call the merge_datasets function from the module
            if hasattr(module, "merge_datasets"):
                test_enriched, train_enriched = module.merge_datasets()
                logger.info(f"Merged test dataset shape: {test_enriched.shape}")
                logger.info(f"Merged train dataset shape: {train_enriched.shape}")
            else:
                logger.warning("No merge_datasets function found in merging_dfs module")
                
            logger.info("Merging dataframes completed")
        except Exception as e:
            logger.error(f"Error in merging dataframes: {e}", exc_info=True)
            raise
    
    def check_required_files(self):
        """Check if all required input files exist."""
        logger.info("Checking for required input files...")
        
        # Define required files
        required_files = [
            (self.raw_data_dir / "all_sentences_anonymized.tar.xz", "All sentences archive"),
            (self.raw_data_dir / "cases_anonymized_txt_raw.tar.gz", "Case texts archive"),
            # Add other critical files here
        ]
        
        missing_files = []
        for file_path, description in required_files:
            if not file_path.exists():
                missing_files.append((str(file_path), description))
        
        if missing_files:
            logger.warning("Missing required input files:")
            for file_path, description in missing_files:
                logger.warning(f"  - {description}: {file_path}")
            
            return False
        
        logger.info("All required input files found")
        return True
    
    def generate_summary_report(self):
        """Generate a summary report of the preprocessing results."""
        logger.info("Generating summary report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {}
        }
        
        # Check processed datasets
        for dataset_name in ["all_sentences", "train", "test", "case_texts", "determination"]:
            processed_path = self.processed_dir / f"{dataset_name}_processed.pkl"
            clean_path = self.clean_data_dir / f"{dataset_name}_clean.pkl"
            
            dataset_info = {
                "processed_exists": processed_path.exists(),
                "clean_exists": clean_path.exists(),
                "processed_size": processed_path.stat().st_size if processed_path.exists() else 0,
                "clean_size": clean_path.stat().st_size if clean_path.exists() else 0,
            }
            
            # Try to get row counts
            if clean_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_pickle(clean_path)
                    dataset_info["clean_rows"] = len(df)
                    dataset_info["clean_columns"] = list(df.columns)
                except Exception as e:
                    logger.error(f"Error reading {clean_path}: {e}")
            
            report["datasets"][dataset_name] = dataset_info
        
        # Check merged datasets
        for dataset_name in ["test", "train"]:
            merged_path = self.merged_dir / f"{dataset_name}_enriched.csv"
            
            if merged_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(merged_path)
                    report["datasets"][f"{dataset_name}_enriched"] = {
                        "exists": True,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "size": merged_path.stat().st_size
                    }
                except Exception as e:
                    logger.error(f"Error reading {merged_path}: {e}")
        
        # Save the report
        report_path = self.base_dir / "data" / "preprocessing_report.json"
        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
        
        return report
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        start_time = time.time()
        
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Processed data directory: {self.processed_dir}")
        logger.info(f"Clean data directory: {self.clean_data_dir}")
        logger.info(f"Merged data directory: {self.merged_dir}")
        
        try:
            # Check required files
            if not self.check_required_files():
                logger.error("Missing required input files. Please check the logs and provide the necessary files.")
                return False
            
            # Import modules
            self.import_modules()
            
            # Run each step
            self.run_initial_preprocessing()
            self.run_full_text_initial_processing()
            self.run_data_cleaning()
            self.run_merging_dfs()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info(f"PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds")
            logger.info("=" * 80)
            
            # Generate summary report
            self.generate_summary_report()
            
            return True
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error("=" * 80)
            logger.error(f"PREPROCESSING PIPELINE FAILED after {duration:.2f} seconds")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            
            return False


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        sys.exit(1) 