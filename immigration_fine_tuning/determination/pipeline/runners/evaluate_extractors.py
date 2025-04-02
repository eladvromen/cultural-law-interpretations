import os
import json
import time
import sys
import pandas as pd
from typing import List, Dict, Any, Type
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Add parent directories to sys.path to allow imports ---
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# Get the parent directory (pipeline)
pipeline_dir = os.path.dirname(current_dir)
# Get the grandparent directory (determination)
determination_dir = os.path.dirname(pipeline_dir)
# Get the great-grandparent directory (immigration_fine_tuning)
fine_tuning_dir = os.path.dirname(determination_dir)
# Get the great-great-grandparent directory (project root)
project_root_dir = os.path.dirname(fine_tuning_dir) 

# Add relevant directories to sys.path
sys.path.insert(0, pipeline_dir)
sys.path.insert(0, determination_dir) 
sys.path.insert(0, fine_tuning_dir)
sys.path.insert(0, project_root_dir) 
# --- End sys.path modification ---

# Import necessary components AFTER modifying sys.path
try:
    from extractors.base.base_determination_extractor import BaseDeterminationExtractor
    from extractors.basic.basic_determination_extraction import BasicDeterminationExtractor
    # Import other extractors as needed (uncomment when ready)
    from extractors.ngram.ngram_determination_extraction import NgramDeterminationExtractor 
    from extractors.sparse.sparse_explicit_extraction import SparseExplicitExtractor 
except ImportError as e:
    print(f"Error importing extractors: {e}")
    print("Please ensure the script is run from a location where Python can find the 'extractors' module,")
    print("or that the necessary paths are added to sys.path correctly.")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = os.path.join(pipeline_dir, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train_dataset.json')
TEST_PATH = os.path.join(DATA_DIR, 'test_dataset.json')

# --- Helper Function to Load Data ---
def load_data(data_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON file."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} items from {data_path}")
        # Basic validation: check if it's a list and items are dicts with 'text' and 'label'
        if not isinstance(data, list):
             raise ValueError("Data is not a list.")
        if data and not (isinstance(data[0], dict) and 'text' in data[0] and 'label' in data[0]):
             raise ValueError("Data items do not have 'text' and 'label' keys.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        sys.exit(1)
    except ValueError as ve:
         print(f"Error: Invalid data format in {data_path}: {ve}")
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {data_path}: {e}")
        sys.exit(1)

# --- Evaluation Function ---
def evaluate_extractor(extractor_class: Type[BaseDeterminationExtractor], 
                       test_data: List[Dict[str, Any]], 
                       train_path: str, 
                       test_path_for_loading: str) -> Dict[str, float]:
    """
    Initializes, (optionally) trains, and evaluates a single extractor.

    Args:
        extractor_class: The class of the extractor to evaluate.
        test_data: The loaded test data (list of dicts with 'text' and 'label').
        train_path: Path to the training data for extractor initialization.
        test_path_for_loading: Path to the test data for extractor initialization.

    Returns:
        A dictionary containing evaluation metrics (accuracy, precision, recall, f1).
    """
    print(f"\n--- Evaluating Extractor: {extractor_class.__name__} ---")
    
    # 1. Instantiate the extractor
    start_time = time.time()
    try:
        extractor = extractor_class()
    except Exception as e:
        print(f"Error instantiating {extractor_class.__name__}: {e}")
        return {} # Return empty dict on failure
        
    instantiation_time = time.time() - start_time
    print(f"Instantiation time: {instantiation_time:.2f} seconds")

    # 2. Load training examples if the method exists
    if hasattr(extractor, 'load_training_examples') and callable(getattr(extractor, 'load_training_examples')):
        print("Calling load_training_examples...")
        load_start_time = time.time()
        try:
            # Assuming use_chunking is a valid parameter based on BasicExtractor
            # Adjust if other extractors have different signatures
            extractor.load_training_examples(train_path, test_path_for_loading, use_chunking=True) 
        except NotImplementedError:
            print(f"Note: load_training_examples not implemented for {extractor_class.__name__}. Skipping.")
        except TypeError as te:
             # Handle cases where load_training_examples has a different signature
             if "unexpected keyword argument 'use_chunking'" in str(te):
                 print("Attempting load_training_examples without use_chunking...")
                 try:
                    extractor.load_training_examples(train_path, test_path_for_loading)
                 except Exception as e:
                     print(f"Error calling load_training_examples (fallback): {e}")
                     # Decide how to handle this - maybe skip evaluation for this extractor?
                     return {} 
             else:
                 print(f"TypeError calling load_training_examples: {te}")
                 return {}
        except Exception as e:
            print(f"Error during load_training_examples for {extractor_class.__name__}: {e}")
            # Optionally decide if evaluation can proceed without successful loading
            # For now, let's return empty metrics if loading fails critically
            return {} 
        load_time = time.time() - load_start_time
        print(f"Data loading/processing time: {load_time:.2f} seconds")
    else:
        print(f"No load_training_examples method found for {extractor_class.__name__}.")

    # 3. Run extraction on test data
    true_labels = []
    predictions = []
    extraction_times = []

    print(f"Running extraction on {len(test_data)} test items...")
    for item in test_data:
        text = item.get('text', '')
        true_label = item.get('label', None)

        # Ensure we have valid data for this item
        if text is None or true_label is None:
             print(f"Warning: Skipping item due to missing text or label: {item}")
             continue
             
        true_labels.append(true_label)

        extract_start_time = time.time()
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                print(f"Warning: Text is not a string for item label {true_label}. Treating as empty.")
                potential_determinations = []
            else:
                potential_determinations = extractor.extract_potential_determinations(text)
                
            # Prediction is 1 if any determination was extracted, 0 otherwise
            predictions.append(1 if potential_determinations else 0)
            
        except Exception as e:
            print(f"Error during extract_potential_determinations for text snippet: '{text[:50]}...'")
            print(f"Error: {e}")
            # Handle error: maybe append a default value (e.g., 0) or skip?
            # Appending 0 assumes failure to extract means no determination.
            predictions.append(0) 
            
        extraction_times.append(time.time() - extract_start_time)

    total_extraction_time = sum(extraction_times)
    avg_extraction_time = total_extraction_time / len(test_data) * 1000 if test_data else 0 # in ms
    print(f"Total extraction time: {total_extraction_time:.2f} seconds")
    print(f"Average time per item: {avg_extraction_time:.2f} ms")

    # 4. Calculate metrics
    if not true_labels or not predictions or len(true_labels) != len(predictions):
         print("Error: Mismatch in labels and predictions or no data to evaluate.")
         return {}
         
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        average='binary', 
        zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print("--- Metrics ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    return metrics

# --- Main Execution ---
def main():
    print("Starting Extractor Evaluation Script...")
    
    # --- Check if data files exist ---
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training data not found at {TRAIN_PATH}")
        # Decide if we can proceed without train data for some extractors
        # For now, exit if train data needed for initialization is missing
        # sys.exit(1) 
        print("Warning: Training data not found. Some extractors might not initialize correctly.")
    if not os.path.exists(TEST_PATH):
        print(f"Error: Test data not found at {TEST_PATH}")
        sys.exit(1)
        
    # --- Load Test Data ---
    test_data = load_data(TEST_PATH)
    if not test_data: # Exit if loading failed
         sys.exit(1)

    # --- Define Extractors to Evaluate ---
    # Add other extractors here as they become ready
    extractor_classes_to_evaluate: List[Type[BaseDeterminationExtractor]] = [
        BasicDeterminationExtractor,
        NgramDeterminationExtractor, 
        SparseExplicitExtractor,
    ]

    # --- Run Evaluation for Each Extractor ---
    all_results = {}
    for extractor_cls in extractor_classes_to_evaluate:
        results = evaluate_extractor(extractor_cls, test_data, TRAIN_PATH, TEST_PATH)
        if results: # Only store if evaluation was successful
             all_results[extractor_cls.__name__] = results
        else:
             print(f"Evaluation failed or was skipped for {extractor_cls.__name__}.")

    # --- Print Summary ---
    print("\n--- Evaluation Summary ---")
    if not all_results:
         print("No evaluation results to display.")
         return
         
    summary_df = pd.DataFrame.from_dict(all_results, orient='index')
    summary_df.index.name = 'Extractor'
    print(summary_df.round(4))

    # --- Optional: Save Summary ---
    # summary_output_path = os.path.join(current_dir, 'extractor_evaluation_summary.csv')
    # try:
    #     summary_df.to_csv(summary_output_path)
    #     print(f"\nEvaluation summary saved to {summary_output_path}")
    # except Exception as e:
    #     print(f"Error saving summary CSV: {e}")

if __name__ == "__main__":
    main() 