import os
import pandas as pd
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path
import tarfile

# Create directory for the dataset in the specified path
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                        "data", "asylex_data")
os.makedirs(data_dir, exist_ok=True)

# Print the absolute path to make it clearer
print(f"Downloading AsyLex data to: {os.path.abspath(data_dir)}")

# List of specific files to download
files_to_download = [
    "cases_anonymized_txt_raw.tar.gz",   
    "all_sentences_anonymized.tar.xz",                   # Full case texts
    "determination_label_extracted_sentences.csv",          # Determination sentences
    "outcome_train_test/test_dataset_gold.csv",             # Gold standard test labels
    "outcome_train_test/train_dataset_silver.csv",          # Training data
    "case_cover/case_cover_entities_and_decision_outcome.csv", # Demographic data
    "main_and_case_cover_all_entities_inferred.csv",         # Added new dataset
    "manual_annotations/case_cover_ner_annotations.jsonl",   # Manual NER annotations for case cover
    "manual_annotations/main_text_annotations__pretrained_labels.jsonl", # Manual annotations with pretrained labels
    "manual_annotations/main_text_annotations_new_labels.jsonl"  # Manual annotations with new labels
]

# Download each file
for filename in files_to_download:
    try:
        print(f"\nDownloading {filename}...")
        
        # Download the file from Hugging Face
        file_path = hf_hub_download(
            repo_id="clairebarale/AsyLex",
            filename=filename,
            repo_type="dataset"
        )
        
        print(f"Downloaded to: {file_path}")
        
        # Create a clean filename for the destination
        clean_name = filename.replace("/", "_")
        dest_path = os.path.join(data_dir, clean_name)
        
        # Copy the file to our dataset directory
        shutil.copy(file_path, dest_path)
        print(f"Copied file to {dest_path}")
        
        # If it's a CSV file, try to read and save a sample
        if filename.endswith('.csv'):
            try:
                # Read with pandas
                df = pd.read_csv(file_path, nrows=5)
                sample_path = os.path.join(data_dir, f"{Path(clean_name).stem}_sample.csv")
                df.to_csv(sample_path, index=False)
                print(f"Saved sample to {sample_path}")
                
                # Print basic info about the dataset
                print(f"Dataset shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Could not parse CSV with pandas: {str(e)}")
        
        # If it's a tar.gz file, extract it
        if filename.endswith('.tar.gz'):
            try:
                extract_dir = os.path.join(data_dir, Path(clean_name).stem)
                os.makedirs(extract_dir, exist_ok=True)
                print(f"Extracting archive to {extract_dir}...")
                
                # Extract only a few files as samples (to avoid extracting everything)
                with tarfile.open(file_path, 'r:gz') as tar:
                    # Get list of all files
                    all_members = tar.getmembers()
                    # Extract only first 3 files as samples
                    sample_members = all_members[:3]
                    for member in sample_members:
                        tar.extract(member, path=extract_dir)
                
                print(f"Extracted {len(sample_members)} sample files from archive")
                print(f"Full archive contains {len(all_members)} files")
                
            except Exception as e:
                print(f"Could not extract archive: {str(e)}")
                
    except Exception as e:
        print(f"Failed to download {filename}: {str(e)}")

print("\nData download complete. Check the 'asylex_data' directory for the files.")