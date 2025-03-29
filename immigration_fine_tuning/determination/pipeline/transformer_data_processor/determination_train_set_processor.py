import re
import json
import os
import pandas as pd
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

class DeterminationTrainSetProcessor:
    def __init__(self):
        # Keywords for low precision validation
        self.determination_keywords = {
            "is allowed", "is dismissed", "finds that", "determine", 
            "concluded that", "decides that", "rules that"
        }
        
        # Maximum length for cleaned text
        self.max_length = 200
        
        # Columns requiring high precision (direct acceptance)
        self.high_precision_columns = {
            'sparse_explicit_extraction',
            'decision_basic_extraction',
            'conclusion_basic_extraction',
            'suspected_last_case_paragraph_basic_extraction'
        }
        
        # Columns requiring keyword validation
        self.low_precision_columns = {
            'reasons_basic_extraction',
            'analysis_basic_extraction'
        }

        # Enhanced stats tracking
        self.stats = {
            'accepted_by_column': defaultdict(int),
            'rejected_by_length': defaultdict(int),
            'rejected_by_keywords': defaultdict(int),
            'duplicates_removed': defaultdict(list)  # Will store {text: [columns where it appeared]}
        }

        # Enhanced patterns for document identifiers
        self.doc_id_patterns = [
            r'No ID client:?\s*[\w\-]+\s*\d*',  # Matches "No ID client: 4076-7411 1"
            r'\[REDACTED\]',  # Matches "[REDACTED]"
            r'\(TA\d+-\d+\)',  # Matches "(TA9-03406)"
            r'NOTICE OF\s*DEC\s*ISION',  # Matches "NOTICE OF DECISION" with possible spaces
            r'File No\.?\s*:?\s*[\w\-]+',  # File numbers
            r'Docket:\s*[\w\-]+',  # Docket numbers
            r'Citation:\s*[\w\-]+',  # Citations
            r'\d+\s*-\s*\d+(?:\s*\d+)?',  # Matches patterns like "1187 3", "VB1-03660", "5259-3734"
            r'\(CA IRB\)',  # Matches "(CA IRB)"
            r'(?:\d{4}\s*)?CanLII\s*\d+',  # Enhanced CanLII pattern
            r'[A-Z]{2,3}\d{1,2}\s*-\s*\d+',  # Matches case numbers like "VB1-03660"
        ]

    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation to create consistent format."""
        # Remove space before period
        text = re.sub(r'\s+\.', '.', text)
        # Ensure single space after period
        text = re.sub(r'\.\s*', '. ', text)
        # Remove trailing period and space
        text = text.rstrip('. ')
        # Add single period at end if not present
        if text and not text.endswith('.'):
            text = text + '.'
        return text

    def remove_doc_identifiers(self, text: str) -> str:
        """Remove document identifiers and metadata."""
        for pattern in self.doc_id_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements."""
        if not text:
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Remove escaped newlines first
        text = text.replace('\\n', ' ')
        text = text.replace('\\r', ' ')
        
        # Remove document identifiers
        text = self.remove_doc_identifiers(text)
        
        # Remove newlines and extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove dates (simple pattern, can be expanded)
        text = re.sub(r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', '', text)
        
        # Remove any remaining number patterns
        text = re.sub(r'(?<![a-zA-Z])\d+(?:\s*-\s*\d+)*(?![a-zA-Z])', '', text)  # Numbers and number ranges not part of words
        
        # Remove any remaining bracketed content
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove any remaining special characters at start/end
        text = re.sub(r'^[-\\/\s]+', '', text)
        text = re.sub(r'[-\\/\s]+$', '', text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Final cleanup and strip
        text = text.strip()
        
        # Remove empty or too short results
        if len(text) < 5:  # Arbitrary minimum length
            return ''
        
        # Remove any lines that are just numbers or special characters
        if re.match(r'^[\d\s\-\\/\.]+$', text):
            return ''
        
        return text

    def contains_determination_keywords(self, text: str) -> bool:
        """Check if text contains any determination keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.determination_keywords)

    def parse_extraction_field(self, field_value: str) -> List[Dict[str, Any]]:
        """Parse JSON-like string into list of dictionaries."""
        try:
            if isinstance(field_value, str):
                return json.loads(field_value.replace("'", '"'))
            return field_value if isinstance(field_value, list) else []
        except json.JSONDecodeError:
            return []

    def process_extraction(self, extraction: Dict[str, Any], is_high_precision: bool, column: str) -> str:
        """Process single extraction and return cleaned text if valid."""
        text = extraction.get('text', '')
        cleaned_text = self.clean_text(text)
        
        # Length validation
        if len(cleaned_text) > self.max_length:
            self.stats['rejected_by_length'][column] += 1
            return ''
            
        # For low precision extractions, validate keywords
        if not is_high_precision and not self.contains_determination_keywords(cleaned_text):
            self.stats['rejected_by_keywords'][column] += 1
            return ''
        
        self.stats['accepted_by_column'][column] += 1    
        return cleaned_text

    def process_row(self, row: Dict[str, Any]) -> List[str]:
        """Process a single row of data and return list of valid determination sentences."""
        valid_determinations = []

        # Process all columns
        for column in self.high_precision_columns | self.low_precision_columns:
            if column not in row:
                continue
                
            extractions = self.parse_extraction_field(row[column])
            is_high_precision = column in self.high_precision_columns
            
            for extraction in extractions:
                cleaned_text = self.process_extraction(extraction, is_high_precision, column)
                if cleaned_text:
                    valid_determinations.append(cleaned_text)

        return valid_determinations

    def create_training_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        """Create final training dataset with binary labels."""
        training_data = []
        seen_texts = {}  # Track unique texts and their sources: {text: [columns]}
        
        for row in data:
            valid_determinations = self.process_row(row)
            for determination in valid_determinations:
                # Normalize for comparison but keep original for storage
                normalized_text = determination.lower().strip()
                
                if normalized_text in seen_texts:
                    # Track duplicate for stats
                    current_column = next(col for col in self.high_precision_columns | self.low_precision_columns 
                                       if col in row)
                    self.stats['duplicates_removed'][normalized_text].append(current_column)
                    continue
                
                training_data.append({
                    'text': determination,
                    'label': 1
                })
                
                # Track where we first saw this text
                current_column = next(col for col in self.high_precision_columns | self.low_precision_columns 
                                   if col in row)
                seen_texts[normalized_text] = current_column
                
        # Update duplicates stats to include original source
        for text, column in seen_texts.items():
            if text in self.stats['duplicates_removed']:
                self.stats['duplicates_removed'][text].insert(0, column)
            
        return training_data

    def print_stats(self):
        """Print processing statistics."""
        print("\n=== Processing Statistics ===")
        
        print("\nAccepted sentences by column:")
        for column, count in self.stats['accepted_by_column'].items():
            print(f"{column}: {count}")
        
        print("\nRejected by length limit by column:")
        for column, count in self.stats['rejected_by_length'].items():
            print(f"{column}: {count}")
        
        print("\nRejected by keyword matching by column:")
        for column, count in self.stats['rejected_by_keywords'].items():
            print(f"{column}: {count}")
        
        print("\nDuplicate removals:")
        duplicate_count = len(self.stats['duplicates_removed'])
        print(f"Total unique texts with duplicates: {duplicate_count}")
        
        # Count total duplicate instances
        total_duplicate_instances = sum(len(columns) - 1 for columns in self.stats['duplicates_removed'].values())
        print(f"Total duplicate instances removed: {total_duplicate_instances}")
        
        # Show some example duplicates (limit to 5)
        if duplicate_count > 0:
            print("\nExample duplicates (up to 5):")
            for text, columns in list(self.stats['duplicates_removed'].items())[:5]:
                print(f"\nText: '{text}'")
                print(f"Found in columns: {' -> '.join(columns)}")
        
        total_accepted = sum(self.stats['accepted_by_column'].values())
        total_rejected = sum(self.stats['rejected_by_length'].values()) + sum(self.stats['rejected_by_keywords'].values())
        print(f"\nTotal accepted (before deduplication): {total_accepted}")
        print(f"Total rejected: {total_rejected}")
        print(f"Total after deduplication: {len(self.stats['duplicates_removed']) + (total_accepted - total_duplicate_instances - duplicate_count)}")
        print(f"Acceptance rate (after deduplication): {(total_accepted - total_duplicate_instances)/(total_accepted + total_rejected)*100:.2f}%")

def main():
    # Define paths
    base_dir = r"C:\Users\shil6369\cultural-law-interpretations\immigration_fine_tuning\determination\pipeline\data\determination_transformer"
    input_file = os.path.join(base_dir, "silver_labels_train.csv")
    output_file = os.path.join(base_dir, "processed_training_data_1_half.json")
    
    # Initialize processor
    processor = DeterminationTrainSetProcessor()
    
    # Read data
    print("Reading input data...")
    df = pd.read_csv(input_file)
    
    # Process data
    print("Processing data...")
    training_data = processor.create_training_data(df.to_dict('records'))
    
    # Save results
    print(f"Saving {len(training_data)} processed examples...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    processor.print_stats()
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
