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

from utils.text_cleaning import clean_text

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
    
    def __init__(self, model_path: str, device: str = None, batch_size: int = 32, threshold: float = 0.5):
        """
        Initialize with a transformer model.
        
        Args:
            model_path: Path to the transformer model checkpoint
            device: Device to run the model on (default: auto-detect)
            batch_size: Batch size for inference
            threshold: Probability threshold for positive classification (default: 0.5)
        """
        import torch as torch_lib
        self.device = device or ('cuda' if torch_lib.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.threshold = threshold
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using classification threshold: {self.threshold}")
        
        # First try to load model config to identify model type
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            model_type = getattr(config, "model_type", "unknown")
            logger.info(f"Detected model type: {model_type}")
            
            # Load model based on detected type
            if model_type == "roberta":
                # Try direct loading first
                try:
                    from transformers import RobertaTokenizer, RobertaForSequenceClassification
                    self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
                    self.model = RobertaForSequenceClassification.from_pretrained(model_path)
                    logger.info("Successfully loaded RoBERTa model directly")
                except Exception as e1:
                    logger.warning(f"Direct loading failed: {e1}")
                    # Try with base model first, then load vocab files
                    from transformers import RobertaTokenizer, RobertaForSequenceClassification
                    logger.info("Falling back to base RoBERTa tokenizer")
                    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                    # Still load the actual model
                    self.model = RobertaForSequenceClassification.from_pretrained(model_path)
                    logger.info("Used base tokenizer with fine-tuned model")
            elif model_type == "bert":
                from transformers import BertTokenizer, BertForSequenceClassification
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                self.model = BertForSequenceClassification.from_pretrained(model_path)
                logger.info("Successfully loaded BERT model")
            else:
                # Fallback to Auto classes
                logger.info(f"Using AutoTokenizer/AutoModel for {model_type} model")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                # Try slow tokenizer if fast fails
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                except:
                    logger.warning("Fast tokenizer failed, trying slow implementation")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            # Final desperate fallback - try with minimum version requirements
            try:
                logger.info("Attempting final fallback loading method...")
                from transformers import AutoModelForSequenceClassification
                import torch
                
                # Try to load model directly with PyTorch
                if Path(model_path).joinpath("model.safetensors").exists():
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    # Create a minimal tokenizer
                    self.tokenizer = MinimalTokenizer(model_path)
                    logger.info("Loaded model with minimal tokenizer")
                else:
                    raise RuntimeError("No usable model format found")
            except Exception as e2:
                logger.error(f"All loading attempts failed: {str(e2)}")
                raise RuntimeError(f"Failed to load model after multiple attempts: {str(e)}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Clean the text first using shared cleaning function
        text = clean_text(text)
        
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
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            # Get probability of positive class (index 1)
            positive_probs = probs[:, 1]
            # Apply threshold to get predictions
            predictions = (positive_probs >= self.threshold).int()
        
        return predictions.cpu().numpy().tolist()
    
    def process_dataframe(self, df: pd.DataFrame, sections: Optional[List[str]] = None, column_suffix_id: str = "transformer") -> tuple:
        """
        Process dataframe with section-specific extraction.
        
        Args:
            df: Input dataframe
            sections: List of text sections to process (default: standard sections)
            column_suffix_id: The identifier used for unique column naming (from config 'name' or stage key)
            
        Returns:
            Tuple of (processed dataframe, statistics)
        """
        logger.info(f"Processing with SectionBasedTransformerExtractor (Identifier: {column_suffix_id})...")
        
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
                logger.warning(f"Identifier '{column_suffix_id}': Section {section} not found in dataframe, skipping")
                continue
                
            logger.info(f"Identifier '{column_suffix_id}': Processing section: {section}")
            section_key = section.replace('_headers_text', '').replace('_text','') # Clean up section key
            
            # Initialize section stats
            stats['section_stats'][section_key] = {
                'total_sentences': 0,
                'total_extractions': 0,
                'documents_with_extractions': 0
            }
            
            # Use column_suffix_id to ensure unique columns
            output_column = f"{section_key}_{column_suffix_id}_extraction" 
            count_column = f"{output_column}_count"
            
            # Apply extractor to each row with progress bar
            section_results = []
            
            # Add progress bar for document processing
            pbar = tqdm(df.iterrows(), total=len(df), desc=f"Identifier '{column_suffix_id}': Processing {section_key}")
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
            
            # Add results and counts to dataframe (using the new unique names)
            df[output_column] = section_results
            df[count_column] = df[output_column].apply(len)
            
            logger.info(f"Identifier '{column_suffix_id}': Completed processing {section_key} section")
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
            batch_size: int = 32, threshold: float = 0.5, column_suffix_id: str = "transformer") -> Optional[Path]:
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
            threshold: Probability threshold for positive classification
            column_suffix_id: Identifier for column naming (from config 'name' or stage key).
            
        Returns:
            Path to output file
        """
        # Get threshold from config if provided
        threshold = config.get('threshold', threshold)
        logger.info(f"Using classification threshold: {threshold}")
        
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
        
        logger.info(f"Runner invoked with column identifier: {column_suffix_id}") # Log the identifier
        logger.info(f"Using input file: {input_file}")
        logger.info(f"Using training data: {train_data}")
        logger.info(f"Using test data: {test_data}")
        logger.info(f"Results will be saved to: {output_path}")
        logger.info(f"Processing sections: {sections}")
        logger.info(f"Using batch size: {batch_size}")
        
        # Load data with better error handling and logging
        logger.info(f"Loading data from {input_file}")
        try:
            # --- Attempt 1: Use default C engine (like the working sparse run likely does) ---
            logger.info("Attempting CSV read with default C engine...")
            df = pd.read_csv(
                input_file,
                encoding='utf-8',
                # engine='python', # Removed for first attempt
                on_bad_lines='warn', # Keep warn for now
                quoting=csv.QUOTE_ALL, 
                escapechar='\\' 
            )
            logger.info(f"DataFrame shape after initial read (C engine): {df.shape}")
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns (C engine)")
            
            # Apply limit if specified
            if 'limit' in config:
                limit = int(config['limit'])
                logger.info(f"Limiting dataset to first {limit} records")
                df = df.head(limit)
                logger.info(f"DataFrame shape after applying limit (if any): {df.shape}")

        except Exception as e_c_engine:
            logger.warning(f"Read with C engine failed or produced warning: {e_c_engine}")
            logger.info("Falling back to Python engine...")
            try:
                # --- Attempt 2: Fallback to Python engine (original method) ---
                 df = pd.read_csv(
                    input_file,
                    encoding='utf-8',
                    engine='python', # Fallback to python engine
                    on_bad_lines='warn',
                    quoting=csv.QUOTE_ALL, 
                    escapechar='\\'
                )
                 logger.info(f"DataFrame shape after initial read (Python engine): {df.shape}")
                 logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns (Python engine)")

                 # Apply limit if specified
                 if 'limit' in config:
                     limit = int(config['limit'])
                     logger.info(f"Limiting dataset to first {limit} records")
                     df = df.head(limit)
                     logger.info(f"DataFrame shape after applying limit (if any) to chunked data: {df.shape}")

            except Exception as e_py_engine:
                 logger.error(f"Error reading CSV with Python engine fallback: {str(e_py_engine)}")
                 # Consider the chunking fallback ONLY if Python engine also fails badly
                 # logger.info("Attempting alternative chunking method...") 
                 # ... (rest of original chunking code) ... # This part might not be needed if C engine works
                 return None # Or handle error appropriately

        # Add CSV file validation
        logger.info(f"Validating dataset structure...")
        required_columns = ['decisionID']  # Add other required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None

        # Create transformer extractor with threshold
        extractor = SectionBasedTransformerExtractor(
            model_path=str(self.model_path),
            batch_size=batch_size,
            threshold=threshold
        )
        
        # Process data with configured sections, passing the column identifier
        logger.info(f"Processing data with transformer extractor (Identifier: '{column_suffix_id}')...")
        results_df, stats = extractor.process_dataframe(df, sections=sections, column_suffix_id=column_suffix_id)
        
        # Log dataframe info before saving
        logger.info(f"DataFrame shape before saving results: {results_df.shape}")

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
                lineterminator='\n'
            )
            logger.info("Successfully saved results")
        except Exception as e:
            logger.error(f"Error saving CSV: {str(e)}")
            return None
        
        # Save a config file with information about this run
        run_config = {
            "run_timestamp": timestamp,
            "column_suffix_id": column_suffix_id, # Add identifier to run config
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

# Define a minimal tokenizer for emergency use
class MinimalTokenizer:
    """Minimal tokenizer implementation when regular loading fails"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.max_length = 512
        logger.warning("Using minimal tokenizer - limited functionality!")
        
        # Try to load vocab if possible
        vocab_path = Path(model_path) / "vocab.json"
        if vocab_path.exists():
            import json
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
    
    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors=None):
        """Minimal implementation that splits on whitespace and assigns tokens"""
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
            
        # Simple tokenize by whitespace
        all_tokens = []
        for text in texts:
            tokens = text.split()
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            all_tokens.append(tokens)
        
        # Find max length for padding
        if padding:
            max_len = max(len(tokens) for tokens in all_tokens)
        else:
            max_len = 0
            
        # Create input tensors
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            # Convert tokens to IDs (assign 1 for unknown)
            ids = [self.vocab.get(token, 1) for token in tokens]
            
            # Create attention mask (1 for real tokens)
            mask = [1] * len(ids)
            
            # Pad sequences
            if padding and len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
                mask = mask + [0] * (max_len - len(mask))
                
            input_ids.append(ids)
            attention_mask.append(mask)
            
        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

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
    parser.add_argument("--threshold", "-th", type=float, default=0.5, help="Probability threshold for positive classification")
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
    if args.threshold is not None:
        config['threshold'] = args.threshold

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
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    
    if output_path:
        print(f"Output saved to: {output_path}")
    else:
        print("Pipeline execution failed")

if __name__ == "__main__":
    main() 