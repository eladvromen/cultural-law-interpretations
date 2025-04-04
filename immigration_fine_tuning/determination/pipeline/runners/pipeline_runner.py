#!/usr/bin/env python3
"""
Configurable pipeline runner for determination extraction.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import csv # Import csv module for quoting constants

# Import runner modules
from run_sparse_extraction import SparseExtractionRunner
from run_basic_extraction import BasicExtractionRunner
from run_ngram_extraction import NgramExtractionRunner
from run_transformer_extraction import TransformerExtractionRunner
from text_processing import process_validation_set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("PipelineRunner")

class PipelineRunner:
    """
    Configurable pipeline runner that orchestrates the execution of extraction stages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline runner with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.pipeline_dir = self.base_dir / "determination" / "pipeline"
        
        # Create results directory if it doesn't exist
        self.pipeline_stages_dir = self.pipeline_dir / "results" / "pipeline_stages"
        self.pipeline_stages_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline runs directory for storing configurations
        self.pipeline_runs_dir = self.pipeline_dir / "results" / "pipeline_runs" 
        self.pipeline_runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data directory if it doesn't exist
        self.data_dir = self.pipeline_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use run name if provided, otherwise use timestamp
        run_name = config.get("run_name", f"pipeline_run_{self.timestamp}")
        
        # Ensure run name is filename-safe by replacing problematic characters
        safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
        
        # Create a run-specific directory for this pipeline run
        if safe_run_name != f"pipeline_run_{self.timestamp}":
            # If custom name provided, include timestamp to ensure uniqueness
            self.run_dir = self.pipeline_runs_dir / f"{safe_run_name}_{self.timestamp}"
        else:
            self.run_dir = self.pipeline_runs_dir / safe_run_name
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to the run directory
        file_handler = logging.FileHandler(self.run_dir / "pipeline.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Extract common configuration
        self.raw_input_file = self._resolve_path(config.get("input_file", "validation_set.csv"))
        self.train_data = self._resolve_path(config.get("train_data", "data/merged/train_enriched.csv"))
        self.test_data = self._resolve_path(config.get("test_data", "data/merged/test_enriched.csv"))
        
        # Extract stages configuration
        self.stages_config = config.get("stages", {})
        self.active_stages = config.get("active_stages", ["text_processing", "sparse", "basic", "ngram", "transformer"])
        
        # Track stage outputs
        self.stage_outputs = {}
        
        # Save the configuration to the run directory
        with open(self.run_dir / "pipeline_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Initialized pipeline with config: {json.dumps(config, indent=2)}")
        logger.info(f"Pipeline run directory: {self.run_dir}")
    
    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path string to a Path object."""
        path = Path(path_str)
        if not path.is_absolute():
            # Try relative to pipeline directory
            pipeline_relative = self.pipeline_dir / path
            if pipeline_relative.exists():
                return pipeline_relative
                
            # Try relative to base directory
            base_relative = self.base_dir / path
            if base_relative.exists():
                return base_relative
        
        return path
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the full extraction pipeline with the configured stages.
        
        Returns:
            Dictionary with results from each stage
        """
        logger.info("Starting pipeline run")
        results = {}
        
        current_input = self.raw_input_file
        preprocessed_file = None
        
        # --- Function to log DataFrame shape ---
        def log_df_shape(file_path, stage_name):
            try:
                if file_path.exists() and str(file_path).lower().endswith('.csv'):
                    # Use robust reading parameters, similar to transformer runner
                    df_check = pd.read_csv(
                        file_path, 
                        encoding='utf-8',
                        # Use default engine first, but be mindful of potential issues
                        # engine='python', 
                        on_bad_lines='warn', 
                        low_memory=False # Try to avoid dtype warnings if possible
                        # quoting=csv.QUOTE_ALL, # May not be needed if writer used QUOTE_ALL
                        # escapechar='\\'
                    )
                    logger.info(f"DataFrame shape after stage '{stage_name}': {df_check.shape} (read from {file_path.name})")
                elif not str(file_path).lower().endswith('.csv'):
                     logger.info(f"Output after stage '{stage_name}' is not a CSV: {file_path.name}")
                else:
                     logger.warning(f"Output file for stage '{stage_name}' not found at {file_path}")
            except Exception as e:
                logger.error(f"Could not read or get shape of output from stage '{stage_name}' at {file_path}: {e}")
        # --- End function ---

        # Run each stage
        for stage in self.active_stages:
            logger.info(f"Running stage: {stage}")
            
            # Get stage-specific config
            stage_config = self.stages_config.get(stage, {})
            stage_succeeded = False # Flag to check if stage produced output

            if stage == "text_processing":
                # Text processing stage
                logger.info("Running text processing stage")
                output_file = self.data_dir / f"preprocessed_validation_{self.timestamp}.csv"
                
                # Process the validation set
                preprocessed_df = process_validation_set(
                    validation_path=current_input,
                    output_path=output_file,
                    sample_size=stage_config.get("sample_size", 0),  # 0 means process all
                    batch_size=stage_config.get("batch_size", 50)
                )
                
                if output_file.exists():
                    logger.info(f"Text processing completed. Output saved to {output_file}")
                    self.stage_outputs["text_processing"] = output_file
                    preprocessed_file = output_file
                    current_input = output_file
                    stage_succeeded = True
                else:
                    logger.error("Text processing failed to produce output file")
                    pass # Error already logged and returned
                    
            elif stage == "sparse":
                # Make sure we use preprocessed file if text_processing was run
                if "text_processing" in self.active_stages and preprocessed_file is not None:
                    current_input = preprocessed_file
                
                # Sparse extraction stage
                runner = SparseExtractionRunner()
                output_file = runner.run(
                    input_file=current_input,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    timestamp=self.timestamp,
                    config=stage_config
                )
                
                if output_file:
                    self.stage_outputs["sparse"] = output_file
                    current_input = output_file
                    stage_succeeded = True
                else:
                    logger.error("Sparse extraction failed")
                    pass # Error already logged and returned
                
            elif stage == "basic":
                # Basic extraction stage with its own section configuration
                runner = BasicExtractionRunner()
                # Get sections specific to basic extraction
                sections = stage_config.get("sections", [
                    'decision_headers_text', 
                    'analysis_headers_text', 
                    'reasons_headers_text',
                    'conclusion_headers_text'
                ])
                
                output_file = runner.run(
                    input_file=current_input,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    timestamp=self.timestamp,
                    config=stage_config,
                    sections=sections,
                    min_score=stage_config.get("min_score", 5.0)
                )
                
                if output_file:
                    self.stage_outputs["basic"] = output_file
                    current_input = output_file
                    stage_succeeded = True
                else:
                    logger.error("Basic extraction failed")
                    pass # Error already logged and returned
                
            elif stage == "ngram":
                # N-gram extraction stage with its own section configuration
                runner = NgramExtractionRunner()
                # Get sections specific to ngram extraction
                sections = stage_config.get("sections", [
                    'decision_headers_text', 
                    'analysis_headers_text', 
                    'reasons_headers_text',
                    'conclusion_headers_text'
                ])
                
                output_file = runner.run(
                    input_file=current_input,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    timestamp=self.timestamp,
                    config=stage_config,
                    sections=sections,
                    min_score=stage_config.get("min_score", 15.0),
                    min_ngram_size=stage_config.get("min_ngram_size", 2),
                    max_ngram_size=stage_config.get("max_ngram_size", 4),
                    ngram_threshold=stage_config.get("ngram_threshold", 0.65)
                )
                
                if output_file:
                    self.stage_outputs["ngram"] = output_file
                    current_input = output_file
                    stage_succeeded = True
                else:
                    logger.error("Ngram extraction failed")
                    pass # Error already logged and returned

            # --- Added Transformer Stage Handling ---
            elif stage.startswith("transformer"): 
                # Handle transformer stages (allows for names like "transformer_bert", "transformer_roberta")
                logger.info(f"Running transformer stage: {stage}")
                
                # Ensure model_path is specified in the stage config
                model_path = stage_config.get("model_path")
                if not model_path:
                    logger.error(f"Missing 'model_path' in configuration for stage '{stage}'")
                    pass # Error already logged and returned
                
                # Resolve the model path relative to base/pipeline dirs if needed
                resolved_model_path = self._resolve_path(model_path)
                if not resolved_model_path.exists():
                     logger.error(f"Model path not found for stage '{stage}': {resolved_model_path}")
                     pass # Error already logged and returned

                # --- Get Column Identifier ---
                # Use 'name' from config if provided, otherwise fall back to the stage key
                column_suffix_id = stage_config.get("name", stage) 
                logger.info(f"Using identifier '{column_suffix_id}' for output columns for stage '{stage}'")
                # --- End Get Column Identifier ---

                # Get other parameters from stage config
                batch_size = stage_config.get("batch_size", 32)
                threshold = stage_config.get("threshold", 0.5)
                sections = stage_config.get("sections", [ # Default sections for transformer
                    'decision_headers_text', 
                    'analysis_headers_text', 
                    'reasons_headers_text',
                    'conclusion_headers_text'
                ])

                # Instantiate the runner for this specific transformer stage
                try:
                    runner = TransformerExtractionRunner(model_path=str(resolved_model_path))
                except Exception as e:
                     logger.error(f"Failed to initialize TransformerExtractionRunner for stage '{stage}': {e}")
                     pass # Error already logged and returned

                output_file = runner.run(
                    input_file=current_input,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    timestamp=self.timestamp,
                    config=stage_config, # Pass the stage-specific config
                    sections=sections,
                    batch_size=batch_size,
                    threshold=threshold,
                    # Pass the determined identifier for column naming
                    column_suffix_id=column_suffix_id  # <<< Changed from stage_name
                )

                if output_file:
                    self.stage_outputs[stage] = output_file # Use the specific stage name as key for tracking stage output file
                    current_input = output_file
                    stage_succeeded = True
                else:
                    logger.error(f"Transformer extraction stage '{stage}' failed")
                    pass # Error already logged and returned
            # --- End Transformer Stage Handling ---
            
            # --- Log shape after successful stage completion ---
            if stage_succeeded:
                log_df_shape(Path(current_input), stage) 
            # --- End logging shape ---

            results[stage] = {
                "output_file": str(current_input) # Store the path regardless of logging success
            }
            
            logger.info(f"Completed stage: {stage}")
        
        # Create final pipeline results
        final_output = {
            "timestamp": self.timestamp,
            "run_name": self.config.get("run_name", f"pipeline_run_{self.timestamp}"),
            "config": self.config,
            "stage_outputs": {k: str(v) for k, v in self.stage_outputs.items()},
            "final_output": str(current_input)
        }
        
        # Save final results summary to both the run directory and pipeline_stages directory
        results_file_run = self.run_dir / "pipeline_results.json"
        results_file_stages = self.pipeline_stages_dir / f"pipeline_results_{self.timestamp}.json"
        
        with open(results_file_run, 'w') as f:
            json.dump(final_output, f, indent=2)
            
        with open(results_file_stages, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        # Copy the final output to the run directory
        if Path(current_input).exists():
            import shutil
            final_output_copy = self.run_dir / Path(current_input).name
            shutil.copy2(current_input, final_output_copy)
            logger.info(f"Copied final output to run directory: {final_output_copy}")
        
        logger.info(f"Pipeline completed. Results saved to {results_file_run}")
        logger.info(f"Final output: {current_input}")
        
        # --- Log shape of final output ---
        log_df_shape(Path(current_input), "final")
        # --- End logging final shape ---

        return results

def load_config(config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use defaults.
    
    Args:
        config_file: Path to configuration JSON file
        
    Returns:
        Configuration dictionary
    """
    # Generate a default timestamp-based run name
    default_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = f"pipeline_run_{default_timestamp}"
    
    # default_config = {
    #     "run_name": "determination_extraction_pipeline_train",
    #     "input_file": "C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/train_1_half.csv",
    #     "train_data": "data/merged/train_enriched.csv",
    #     "test_data": "data/merged/test_enriched.csv",
    #     "active_stages": ["text_processing", "sparse", "basic", "ngram", "transformer"],
    #     "stages": {
    #         "text_processing": {
    #             "sample_size": 0,  # 0 means process all
    #             "batch_size": 50
    #         },
    #         "sparse": {},
    #         "basic": {
    #             "min_score": 5.0,
    #             "sections": [
    #                 'decision_headers_text', 
    #                 'analysis_headers_text', 
    #                 'reasons_headers_text',
    #                 'conclusion_headers_text',
    #                 'suspected_last_case_paragraph'
    #             ]
    #         },
    #         "ngram": {
    #             "min_score": 10.0,
    #             "min_ngram_size": 2,
    #             "max_ngram_size": 4,
    #             "ngram_threshold": 0.65,
    #             "sections": [
    #                 'decision_headers_text', 
    #                 'analysis_headers_text', 
    #                 'reasons_headers_text',
    #                 'conclusion_headers_text'
    #             ]
    #         },
    #         "transformer": {
    #             "model_path": "path/to/your/default/transformer/model",
    #             "batch_size": 32,
    #             "threshold": 0.5,
    #             "sections": [
    #                 'decision_headers_text', 
    #                 'analysis_headers_text', 
    #                 'reasons_headers_text',
    #                 'conclusion_headers_text'
    #             ]
    #         }
    #     }
    # }
    default_config = {
            "run_name": "pipeline_with_all_models_final",
            "input_file": "/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/preprocessed_determination_extraction_set.csv",
            "active_stages": ["sparse", "basic","transformer_bert", "transformer_roberta"],
            "stages": {
                "text_processing": {
                "sample_size": 0
                },
                "sparse": {},
                "basic": {
                    "min_score": 5.0,
                    "sections": [
                        'cleaned_text',
                        'decision_headers_text', 
                        'analysis_headers_text', 
                        'reasons_headers_text',
                        'conclusion_headers_text',
                        'suspected_last_case_paragraph'
                    ]
                },
                "ngram": {
                    "min_score": 10.0,
                    "min_ngram_size": 2,
                    "max_ngram_size": 4,
                    "ngram_threshold": 0.65,
                    "sections": [
                        'cleaned_text',
                        'decision_headers_text', 
                        'analysis_headers_text', 
                        'reasons_headers_text',
                        'conclusion_headers_text'
                    ]
                },
                "transformer_bert": {
                "name": "bert_anomaly",
                "model_path": "/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/anomaly_run_20250330_171154/best_model",
                "batch_size": 32,
                "threshold": 0.29,
                "sections": ["cleaned_text", "decision_headers_text", "determination_headers_text", "analysis_headers_text", "reasons_headers_text", "conclusion_headers_text"]
                },
                "transformer_roberta": {
                "name": "roberta_balanced",
                "model_path": "/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/balanced_legal_roberta_base_20250331_162411/best_model",
                "batch_size": 32,
                "threshold": 0.23,
                "sections": ["cleaned_text", "decision_headers_text", "determination_headers_text", "analysis_headers_text", "reasons_headers_text", "conclusion_headers_text"]
                }
            }
            }
    if not config_file:
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Merge user config with defaults
        merged_config = default_config.copy()
        for key, value in user_config.items():
            if key == "stages" and isinstance(value, dict) and "stages" in default_config:
                # For stages, merge with defaults for each stage
                for stage, stage_config in value.items():
                    if stage in merged_config["stages"]:
                        merged_config["stages"][stage].update(stage_config)
                    else:
                        merged_config["stages"][stage] = stage_config
            else:
                merged_config[key] = value
        
        return merged_config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return default_config

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Run configurable determination extraction pipeline")
    parser.add_argument("--config", "-c", help="Path to configuration JSON file")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--stages", "-s", help="Comma-separated list of stages to run")
    parser.add_argument("--name", "-n", help="Name for this pipeline run")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.input:
        config["input_file"] = args.input
    
    if args.stages:
        config["active_stages"] = args.stages.split(",")
    
    if args.name:
        config["run_name"] = args.name
    
    # Run pipeline
    runner = PipelineRunner(config)
    runner.run_pipeline()

if __name__ == "__main__":
    main() 