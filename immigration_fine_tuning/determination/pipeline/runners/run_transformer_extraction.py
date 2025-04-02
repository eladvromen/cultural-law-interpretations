#!/usr/bin/env python3
# Add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Run transformer-based extraction model on validation dataset.
"""

from pathlib import Path
import logging
import datetime
import pandas as pd
import json
import ast
from typing import Dict, Any, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("TransformerExtraction")

def parse_list_column(value):
    """Parse a column value that contains a list representation."""
    if pd.isna(value):
        return []
    
    if isinstance(value, list):
        return value
    
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (SyntaxError, ValueError):
        return []

class SectionBasedTransformerExtractor:
    """Process specific text sections with the transformer model and store counts."""
    
    def __init__(self, model_path: str, device: str = None, batch_size: int = 32):
        """
        Initialize with a transformer model.
        
        Args:
            model_path: Path to the transformer model checkpoint
            device: Device to run the model on (default: auto-detect)
            batch_size: Batch size for inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def clean_text(self, text: str) -> str:
        """Clean text for transformer input."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Clean the text first
        text = self.clean_text(text)
        
        # Split on common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Further clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Additional filtering for minimum sentence length and structure
        filtered_sentences = []
        for s in sentences:
            # Skip if less than 3 words
            if len(s.split()) < 3:
                continue
            
            # Skip administrative headers matching patterns like:
            # "No de dossier de la SAI: TB6 17977 Client ID No."
            if re.search(r'(?i)(dossier|file|client|id).*(no|number|\d+)', s):
                continue
            
            # Skip sentences that are mostly numbers and IDs
            if len(re.findall(r'\d+', s)) > len(s.split()) / 3:  # If more than 1/3 of words are numbers
                continue
            
            filtered_sentences.append(s)
        
        return filtered_sentences
    
    def process_batch(self, sentences: List[str]) -> List[int]:
        """Process a batch of sentences through the transformer."""
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        return predictions.cpu().numpy().tolist()
    
    def process_dataframe(self, df: pd.DataFrame, sections: Optional[List[str]] = None) -> tuple:
        """
        Process dataframe with section-specific extraction.
        
        Args:
            df: Input dataframe
            sections: List of text sections to process (default: standard sections)
            
        Returns:
            Tuple of (processed dataframe, statistics)
        """
        logger.info("Processing with SectionBasedTransformerExtractor...")
        
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
            'total_sentences': 0,
            'total_extractions': 0,
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
                'total_sentences': 0,
                'total_extractions': 0,
                'documents_with_extractions': 0
            }
            
            # Process section
            output_column = f"{section_key}_transformer_extraction"
            count_column = f"{output_column}_count"
            
            # Apply extractor to each row with progress bar
            section_results = []
            
            # Add progress bar for document processing
            pbar = tqdm(df.iterrows(), total=len(df), desc=f"Processing {section_key}")
            for _, row in pbar:
                text = row.get(section, '')
                if pd.isna(text) or not text:
                    section_results.append([])
                    continue
                
                # Clean and split text
                cleaned_text = self.clean_text(text)
                sentences = self.split_into_sentences(cleaned_text)
                
                # Update stats
                stats['total_sentences'] += len(sentences)
                stats['section_stats'][section_key]['total_sentences'] += len(sentences)
                
                if not sentences:
                    section_results.append([])
                    continue
                
                # Process sentences in batches with nested progress bar
                extractions = []
                for i in range(0, len(sentences), self.batch_size):
                    batch = sentences[i:i + self.batch_size]
                    predictions = self.process_batch(batch)
                    
                    # Extract sentences predicted as determinations
                    for sentence, pred in zip(batch, predictions):
                        if pred == 1:  # 1 indicates determination sentence
                            extractions.append(sentence)
                
                # Update progress bar description with current stats
                pbar.set_postfix({
                    'sentences': len(sentences),
                    'extractions': len(extractions)
                })
                
                # Update stats
                stats['total_extractions'] += len(extractions)
                stats['section_stats'][section_key]['total_extractions'] += len(extractions)
                
                if len(extractions) > 0:
                    stats['section_stats'][section_key]['documents_with_extractions'] += 1
                
                section_results.append(extractions)
            
            pbar.close()
            
            # Add results and counts to dataframe
            df[output_column] = section_results
            df[count_column] = df[output_column].apply(len)
            
            # Log section completion
            logger.info(f"Completed processing {section_key} section")
            if stats['section_stats'][section_key]['total_sentences'] > 0:
                extraction_rate = (stats['section_stats'][section_key]['total_extractions'] / 
                                 stats['section_stats'][section_key]['total_sentences'])
                logger.info(f"  Extraction rate: {extraction_rate:.1%}")
        
        # Log statistics
        logger.info("Extraction statistics:")
        logger.info(f"  Total sentences processed: {stats['total_sentences']}")
        logger.info(f"  Total extractions: {stats['total_extractions']}")
        
        for section_key, section_stats in stats['section_stats'].items():
            if section_stats['total_sentences'] > 0:
                extraction_rate = section_stats['total_extractions'] / section_stats['total_sentences']
                logger.info(f"  {section_key.capitalize()} section:")
                logger.info(f"    Total sentences: {section_stats['total_sentences']}")
                logger.info(f"    Total extractions: {section_stats['total_extractions']}")
                logger.info(f"    Extraction rate: {extraction_rate:.1%}")
                logger.info(f"    Documents with extractions: {section_stats['documents_with_extractions']}")
        
        return df, stats

class TransformerExtractionRunner:
    """Runner for the transformer-based extraction pipeline stage."""
    
    def __init__(self, model_path: str):
        """
        Initialize the runner.
        
        Args:
            model_path: Path to the transformer model checkpoint
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
    
    def run(self, input_file: Union[str, Path], train_data: Union[str, Path], test_data: Union[str, Path],
            timestamp: str = None, config: Dict = None, sections: List[str] = None,
            batch_size: int = 32) -> Optional[Path]:
        """
        Run transformer extraction pipeline.
        
        Args:
            input_file: Path to input file 
            train_data: Path to training data
            test_data: Path to test data
            timestamp: Timestamp for this run (default: generate new)
            config: Configuration dictionary
            sections: List of sections to process
            batch_size: Batch size for inference
            
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
        batch_size = config.get('batch_size', batch_size)
        
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
        run_dir = pipeline_stages_dir / f"transformer_extraction_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Set output file path in the new directory
        output_path = run_dir / "validation_with_transformer_extraction.csv"
        
        # Add log file to the run directory
        file_handler = logging.FileHandler(run_dir / "transformer_extraction.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Using training data: {train_data}")
        logger.info(f"Using test data: {test_data}")
        logger.info(f"Results will be saved to: {output_path}")
        logger.info(f"Processing sections: {sections}")
        logger.info(f"Using batch size: {batch_size}")
        
        # Load data with better error handling and logging
        logger.info(f"Loading data from {input_file}")
        try:
            # First attempt: try reading with more robust settings
            df = pd.read_csv(
                input_file,
                encoding='utf-8',
                engine='python',  # More forgiving engine
                on_bad_lines='warn',
                quoting=csv.QUOTE_ALL,  # Handle all quotes
                escapechar='\\'
            )
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            logger.info(f"Columns: {', '.join(df.columns)}")
            
            # Apply limit if specified
            if 'limit' in config:
                limit = int(config['limit'])
                logger.info(f"Limiting dataset to first {limit} records")
                df = df.head(limit)
                logger.info(f"Dataset limited to {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            logger.info("Attempting alternative reading method...")
            try:
                # Second attempt: chunk reading
                chunks = []
                for chunk in pd.read_csv(
                    input_file,
                    encoding='utf-8',
                    engine='python',
                    on_bad_lines='skip',
                    chunksize=1000
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns using chunk method")
                logger.info(f"Columns: {', '.join(df.columns)}")
                
                # Apply limit if specified
                if 'limit' in config:
                    limit = int(config['limit'])
                    logger.info(f"Limiting dataset to first {limit} records")
                    df = df.head(limit)
                    logger.info(f"Dataset limited to {len(df)} records")
                
            except Exception as e:
                logger.error(f"Failed to read CSV file: {str(e)}")
                return None

        # Add CSV file validation
        logger.info(f"Validating dataset structure...")
        required_columns = ['decisionID']  # Add other required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None

        # Create transformer extractor
        extractor = SectionBasedTransformerExtractor(
            model_path=str(self.model_path),
            batch_size=batch_size
        )
        
        # Process data with configured sections
        logger.info("Processing data with transformer extractor...")
        results_df, stats = extractor.process_dataframe(df, sections=sections)
        
        # Save results with better error handling
        logger.info(f"Saving results to {output_path}")
        try:
            # Log dataframe info before saving
            logger.info(f"Saving DataFrame with {len(results_df)} rows and {len(results_df.columns)} columns")
            logger.info(f"Output columns: {', '.join(results_df.columns)}")
            
            # Save with explicit encoding and quoting - using lineterminator instead of line_terminator
            results_df.to_csv(
                output_path,
                index=False,
                encoding='utf-8',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                lineterminator='\n'  # Changed from line_terminator to lineterminator
            )
            logger.info("Successfully saved results")
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            return None
        
        # Save a config file with information about this run
        run_config = {
            "run_timestamp": timestamp,
            "model_path": str(self.model_path),
            "batch_size": batch_size,
            "sections_processed": sections,
            "extraction_stats": stats,
            "input_file": str(input_file),
            "output_file": str(output_path),
            "train_data": str(train_data),
            "test_data": str(test_data),
            "config": config
        }
        
        with open(run_dir / "run_config.json", 'w') as f:
            json.dump(run_config, f, indent=2)
        
        # Remove the file handler to avoid duplicate logs in future pipeline stages
        logger.removeHandler(file_handler)
        
        logger.info(f"Completed processing. Results saved to {output_path}")
        logger.info(f"Run information saved to {run_dir}")
        
        return output_path

def main():
    """Run transformer extraction pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run transformer extraction pipeline")
    parser.add_argument("--model", "-m", required=True, help="Path to transformer model checkpoint")
    parser.add_argument("--input", "-i", help="Input file path")
    parser.add_argument("--train", "-t", help="Training data path")
    parser.add_argument("--test", "-e", help="Test data path")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--sections", "-s", help="Comma-separated list of sections to process")
    parser.add_argument("--limit", "-l", type=int, help="Limit processing to first N records")
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Set default paths if not provided
    base_dir = Path(__file__).parent.parent.parent.parent
    input_file = args.input or base_dir / "determination" / "pipeline" / "data" / "preprocessed_determination_extraction_set.csv"
    train_data = args.train or base_dir / "data" / "merged" / "train_enriched.csv"
    test_data = args.test or base_dir / "data" / "merged" / "test_enriched.csv"
    
    # Update config with command line arguments
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.limit is not None:
        config['limit'] = args.limit
        logger.info(f"Setting processing limit to {args.limit} records")

    # Parse sections if provided
    sections = None
    if args.sections:
        sections = args.sections.split(",")

    # Run pipeline
    runner = TransformerExtractionRunner(model_path=args.model)
    output_path = runner.run(
        input_file=input_file,
        train_data=train_data,
        test_data=test_data,
        config=config,
        sections=sections,
        batch_size=args.batch_size
    )
    
    if output_path:
        print(f"Output saved to: {output_path}")
    else:
        print("Pipeline execution failed")

if __name__ == "__main__":
    main() 