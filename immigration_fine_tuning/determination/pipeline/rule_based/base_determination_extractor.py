import pandas as pd
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from functools import lru_cache

class BaseDeterminationExtractor:
    """
    Base class for determination extraction models with common functionality.
    """
    
    def __init__(self):
        # Performance tracking
        self.processed_cases = 0
        self.processing_time = 0.0
        
        # Lazy loading flag
        self._data_loaded = False
    
    @lru_cache(maxsize=10000)
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced preprocessing and caching."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Fix run-together words (common OCR errors) - all in one pass
        for original, replacement in [
            ('isallowed', 'is allowed'),
            ('isdismissed', 'is dismissed'),
            ('ishereby', 'is hereby'),
            ('declaredabandoned', 'declared abandoned'),
            ('tosubsection', 'to subsection'),
            ('wasallowed', 'was allowed'),
            ('bedismissed', 'be dismissed')
        ]:
            text = text.replace(original, replacement)
        
        # Replace apostrophes with a temporary character
        text = re.sub(r'(\w)\'(\w)', r'\1§\2', text)
        
        # Remove punctuation except for the temporary character in one operation
        text = re.sub(r'[^\w\s§]', ' ', text)
        
        # Restore apostrophes
        text = text.replace('§', "'")
        
        # Normalize whitespace in one operation
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _parse_determination_value(self, val: str) -> List[str]:
        """Parse determination values with enhanced handling of complex formats."""
        if not isinstance(val, str) or not val.strip():
            return []
            
        examples = []
        
        # Handle complex nested formats
        if '[' in val and ']' in val:
            # Pattern to extract content within nested quotes and brackets
            nested_items = re.findall(r'\[\'([^\']+)\'\]', val)
            for item in nested_items:
                if item.strip():
                    # Fix common OCR issues
                    item = re.sub(r'isallowed', 'is allowed', item)
                    item = re.sub(r'declaredabandoned', 'declared abandoned', item)
                    examples.append(item.strip())
            
            # If we found nested items, process them and return
            if nested_items:
                return [ex for ex in examples if ex.strip()]
        
        # Try to handle JSON-like formats
        if val.startswith('[') or val.startswith('{'):
            try:
                # Replace both single and double quotes with standard double quotes
                normalized_val = val.replace("'", '"')
                parsed = json.loads(normalized_val)
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            examples.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            examples.append(item['text'])
                        else:
                            examples.append(str(item))
                elif isinstance(parsed, dict) and 'text' in parsed:
                    examples.append(parsed['text'])
                
                # If we parsed JSON successfully, return the results
                if examples:
                    return [ex for ex in examples if ex.strip()]
            except:
                # If JSON parsing failed, continue with other methods
                pass
        
        # Look for comma-separated parts that could be multiple determinations
        if ',' in val and not examples:
            parts = val.split(',')
            for part in parts:
                clean_part = part.strip().strip("'").strip('"')
                if len(clean_part.split()) >= 3:  # Minimum length for a determination
                    examples.append(clean_part)
        
        # If we still haven't found anything, add the entire string
        if not examples and val.strip():
            examples.append(val.strip())
        
        return [ex for ex in examples if ex.strip()]
    
    def _extract_examples_from_file(self, file_path, use_chunking):
        """Extract determination examples from a data file."""
        examples = []
        
        # Define chunk processor function
        def process_chunk(chunk):
            chunk_examples = []
            
            if 'extracted_sentences_determination' not in chunk.columns:
                return chunk_examples
            
            for val in chunk['extracted_sentences_determination'].dropna():
                if not isinstance(val, str):
                    continue
                
                # Parse determination value
                parsed_examples = self._parse_determination_value(val)
                chunk_examples.extend(parsed_examples)
            
            return chunk_examples
        
        # Load and process data
        if use_chunking:
            print(f"Loading data from {file_path} using chunking...")
            chunk_size = 5000
            try:
                for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
                    examples.extend(process_chunk(chunk))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Loading data from {file_path} in one go...")
            try:
                df = pd.read_csv(file_path, low_memory=False)
                examples.extend(process_chunk(df))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return examples
    
    @lru_cache(maxsize=10000)
    def _split_into_sentences(self, text: str) -> Tuple[str, ...]:
        """
        Split text into sentences.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of sentences (for cache compatibility)
        """
        # Handle newlines
        text = re.sub(r'\n+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter empty sentences
        sentences = tuple(s.strip() for s in sentences if s.strip())
        
        return sentences
    
    def load_training_examples(self, train_path: str, test_path: str, use_chunking: bool = True) -> None:
        """
        Load training examples from both train and test files.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_training_examples")
    
    def extract_potential_determinations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential determination sentences from the text.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement extract_potential_determinations")
    
    def process_case(self, text: str) -> Dict[str, Any]:
        """
        Process a case to extract determinations.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_case")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model."""
        if self.processed_cases == 0:
            return {'processed_cases': 0, 'avg_time_per_case': 0}
            
        return {
            'processed_cases': self.processed_cases,
            'total_processing_time': f"{self.processing_time:.2f} seconds",
            'avg_time_per_case': f"{(self.processing_time / self.processed_cases) * 1000:.2f} ms"
        }
    
    def ensure_data_loaded(self):
        """Ensure training data is loaded (lazy loading)."""
        if not self._data_loaded:
            raise ValueError("Training data not loaded. Call load_training_examples() first.")