import json
import pandas as pd
import os
from pathlib import Path

def process_determination_data(input_file):
    """
    Process JSONL file containing legal text annotations into a simplified format
    for determination classification.
    
    Args:
        input_file (str): Path to input JSONL file
        
    Returns:
        list: List of dictionaries with 'text' and 'label' keys
        Labels:
            0: No determination mention or label
            1: Has at least one DETERMINATION label
            2: Has 'determination' word but no DETERMINATION label
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            # Extract text
            text = entry['text'].lower()  # Convert to lowercase for word matching
            
            # Get all spans and their labels
            spans = entry.get('spans', [])
            
            # Check for any DETERMINATION labels in all spans
            determination_labels = [
                span.get('label') == 'DETERMINATION' 
                for span in spans
            ]
            has_determination_label = any(determination_labels)
            
            # For debugging
            if has_determination_label:
                determination_count = sum(determination_labels)
                if determination_count > 1:
                    print(f"Multiple determinations found ({determination_count}) in text: {entry['text']}")
            
            # Check if the word 'determination' appears in the text
            has_determination_word = 'determination' in text
            
            # Determine label
            if has_determination_label:
                label = 1
            elif has_determination_word:
                label = 2
            else:
                label = 0
            
            # Create processed entry
            processed_entry = {
                'text': entry['text'],  # Keep original text with original case
                'label': label
            }
            
            processed_data.append(processed_entry)
    
    return processed_data

def save_processed_data(processed_data, output_file):
    """
    Save processed data to a new JSONL file.
    
    Args:
        processed_data (list): List of processed entries
        output_file (str): Path to output file
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def main():
    # Use OS-agnostic paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'asylex_data', 
                             'manual_annotations_main_text_annotations_new_labels.jsonl')
    output_file = os.path.join(base_dir, 'determination', 'pipeline', 'data',
                              'determination_transformer', 'determination_transformer_data.jsonl')
    
    # Process the data
    processed_data = process_determination_data(input_file)
    
    # Save processed data
    save_processed_data(processed_data, output_file)
    
    # Print some statistics
    total = len(processed_data)
    determinations = sum(1 for entry in processed_data if entry['label'] == 1)
    determination_word = sum(1 for entry in processed_data if entry['label'] == 2)
    print(f"Total entries processed: {total}")
    print(f"Determination labeled entries: {determinations}")
    print(f"Entries with 'determination' word (no label): {determination_word}")
    print(f"Non-determination entries: {total - determinations - determination_word}")

if __name__ == "__main__":
    main() 