import json
import random
from pathlib import Path
import pandas as pd
import os

# Define base directory
BASE_DIR = r"C:\Users\shil6369\cultural-law-interpretations\immigration_fine_tuning\determination\pipeline\data\determination_transformer"

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data, output_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_data():
    # Define input and output paths
    det_data_path = os.path.join(BASE_DIR, "determination_transformer_data.jsonl")
    processed_data_path = os.path.join(BASE_DIR, "processed_training_data.json")
    train_output_path = os.path.join(BASE_DIR, "train_dataset.json")
    test_output_path = os.path.join(BASE_DIR, "test_dataset.json")

    # Load both datasets
    print("Loading datasets...")
    det_data = load_jsonl(det_data_path)
    processed_data = load_json(processed_data_path)

    # Initialize train and test sets
    train_data = []
    test_data = []

    print("Processing determination transformer data...")
    # Process determination_transformer_data.jsonl
    label_1_records = [x for x in det_data if x['label'] == 1]
    label_2_records = [{'text': x['text'], 'label': 0} for x in det_data if x['label'] == 2]
    label_0_records = [x for x in det_data if x['label'] == 0]

    # First, split label 2 records (now converted to 0)
    print("Processing label 2 records...")
    random.seed(42)  # for reproducibility
    random.shuffle(label_2_records)
    split_idx = int(len(label_2_records) * 0.9)
    label_2_train = label_2_records[:split_idx]
    label_2_test = label_2_records[split_idx:]

    # All label 1 records go to test
    print(f"Found {len(label_1_records)} records with label 1")
    test_data.extend(label_1_records)
    num_label_1 = len(label_1_records)

    # Calculate how many label 0 records we need for test
    # We need the same number as label 1s, but need to account for label 2s that went to test
    num_label_0_needed = num_label_1 - len(label_2_test)
    print(f"Selecting {num_label_0_needed} label 0 records for test set...")
    
    # Randomly select the required number of label 0 records for test
    test_label_0 = random.sample(label_0_records, num_label_0_needed)
    test_data.extend(test_label_0)
    test_data.extend(label_2_test)  # Add the test portion of label 2
    
    # Remaining label 0 records go to train
    train_label_0 = [x for x in label_0_records if x not in test_label_0]
    train_data.extend(train_label_0)
    train_data.extend(label_2_train)  # Add the train portion of label 2

    # Add all processed_training_data to train
    print("Adding processed training data to train set...")
    train_data.extend(processed_data)

    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Save datasets
    print("Saving datasets...")
    save_dataset(train_data, train_output_path)
    save_dataset(test_data, test_output_path)

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    print("\nLabel distribution in train set:")
    train_labels = pd.Series([x['label'] for x in train_data]).value_counts()
    print(train_labels)
    print("\nLabel distribution in test set:")
    test_labels = pd.Series([x['label'] for x in test_data]).value_counts()
    print(test_labels)

if __name__ == "__main__":
    try:
        print("Starting data split process...")
        split_data()
        print("\nData split completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
