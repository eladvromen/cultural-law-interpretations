import os
import pandas as pd
import tarfile
import json
from pathlib import Path

# Path to your data directory
data_dir = r"C:\Users\shil6369\cultural-law-interpretations\immigration_fine_tuning\data\asylex_data"

# Output file for the structure information
output_file = os.path.join(data_dir, "dataset_structure.json")

# Dictionary to store dataset structures
dataset_structures = {}

print("=" * 80)
print("ANALYZING ASYLEX DATASET STRUCTURES")
print("=" * 80)

# Get all files in the directory
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Process each CSV file
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        print(f"\nAnalyzing {file}...")
        
        # Try different delimiters
        for delimiter in [',', ';', '\t']:
            try:
                # Read just a few rows to determine structure
                df = pd.read_csv(file_path, sep=delimiter, nrows=5)
                
                if len(df.columns) > 1:  # If we found a good delimiter
                    print(f"  Successfully read with delimiter: '{delimiter}'")
                    
                    # Get column information
                    columns = []
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        sample_values = df[col].head(3).tolist()
                        # Convert sample values to strings for JSON serialization
                        sample_values = [str(val) if not pd.isna(val) else "NaN" for val in sample_values]
                        
                        columns.append({
                            "name": col,
                            "dtype": dtype,
                            "sample_values": sample_values
                        })
                    
                    # Store structure information
                    dataset_structures[file] = {
                        "delimiter": delimiter,
                        "num_columns": len(df.columns),
                        "num_rows": "Unknown (only read 5 rows for analysis)",
                        "columns": columns
                    }
                    
                    # Try to get total row count without loading entire file
                    try:
                        # Count lines in file (subtract 1 for header)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            line_count = sum(1 for _ in f) - 1
                        dataset_structures[file]["num_rows"] = line_count
                        print(f"  Total rows: {line_count}")
                    except Exception as e:
                        print(f"  Could not count rows: {str(e)}")
                    
                    break  # Found a good delimiter, no need to try others
            except Exception as e:
                continue  # Try next delimiter
        
        if file not in dataset_structures:
            print(f"  Could not determine structure: {str(e)}")

# Process JSONL files
for file in files:
    if file.endswith('.jsonl'):
        file_path = os.path.join(data_dir, file)
        print(f"\nAnalyzing JSONL file {file}...")
        
        try:
            # Read the first few lines to analyze structure
            sample_records = []
            line_count = 0
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    if i < 5:  # Only read first 5 records for analysis
                        try:
                            record = json.loads(line)
                            sample_records.append(record)
                        except json.JSONDecodeError:
                            print(f"  Warning: Line {i+1} is not valid JSON")
            
            # Analyze structure based on first record
            if sample_records:
                first_record = sample_records[0]
                
                # Get keys and sample values
                keys = []
                for key, value in first_record.items():
                    # Get sample values from all records
                    sample_values = []
                    for record in sample_records[:3]:
                        if key in record:
                            # Convert to string for display
                            if isinstance(record[key], (dict, list)):
                                sample_values.append(str(type(record[key])))
                            else:
                                sample_values.append(str(record[key]))
                    
                    keys.append({
                        "name": key,
                        "type": type(value).__name__,
                        "sample_values": sample_values
                    })
                
                # Store structure information
                dataset_structures[file] = {
                    "type": "jsonl",
                    "num_records": line_count,
                    "keys": keys,
                    "sample_record": str(first_record)[:500] + "..." if len(str(first_record)) > 500 else str(first_record)
                }
                
                print(f"  Total records: {line_count}")
                print(f"  Keys found: {[k['name'] for k in keys]}")
            else:
                dataset_structures[file] = {
                    "type": "jsonl",
                    "num_records": line_count,
                    "error": "No valid JSON records found in sample"
                }
                print(f"  Total lines: {line_count}")
                print(f"  Warning: Could not find valid JSON records in sample")
                
        except Exception as e:
            print(f"  Error analyzing JSONL file: {str(e)}")
            dataset_structures[file] = {
                "type": "jsonl",
                "error": str(e)
            }

# Check for tar.gz file to analyze structure
tar_files = [
    ("cases_anonymized_txt_raw.tar.gz", "r:gz"),
    ("all_sentences_anonymized.tar.xz", "r:xz")
]

for tar_filename, mode in tar_files:
    tar_file = os.path.join(data_dir, tar_filename)
    if os.path.exists(tar_file):
        print(f"\nAnalyzing {os.path.basename(tar_file)}...")
        
        try:
            with tarfile.open(tar_file, mode) as tar:
                # Get all members
                all_members = tar.getmembers()
                
                # Print first few members to see what's inside
                print(f"  First few members in archive:")
                for i, member in enumerate(all_members[:5]):
                    print(f"    - {member.name} ({member.size} bytes, {'directory' if member.isdir() else 'file'})")
                
                # Filter for .txt files
                txt_members = [m for m in all_members if m.name.endswith('.txt')]
                
                # Also check for other common file extensions
                json_members = [m for m in all_members if m.name.endswith('.json')]
                csv_members = [m for m in all_members if m.name.endswith('.csv')]
                nested_archives = [m for m in all_members if any(m.name.endswith(ext) for ext in 
                                  ['.tar', '.tar.gz', '.tar.xz', '.zip', '.rar'])]
                
                # Get sample of file names for all types
                sample_files = [m.name for m in all_members[:5]]
                
                # Extract one sample file to analyze structure (try any file, not just txt)
                sample_content = "No files found or could not extract content"
                if all_members and not all(m.isdir() for m in all_members):
                    # Find first file that's not a directory
                    for member in all_members:
                        if not member.isdir():
                            try:
                                extracted = tar.extractfile(member)
                                if extracted:
                                    # Try to read as text
                                    content = extracted.read()
                                    try:
                                        content_text = content.decode('utf-8', errors='ignore')
                                        # Get first few lines as sample
                                        sample_content = '\n'.join(content_text.split('\n')[:10])
                                    except:
                                        # If it's binary data, just note the size
                                        sample_content = f"Binary data, size: {len(content)} bytes"
                                    break
                            except Exception as e:
                                sample_content = f"Could not extract file: {str(e)}"
                
                # Store structure information
                dataset_structures[os.path.basename(tar_file)] = {
                    "type": "archive",
                    "total_files": len(all_members),
                    "text_files": len(txt_members),
                    "json_files": len(json_members),
                    "csv_files": len(csv_members),
                    "nested_archives": len(nested_archives),
                    "sample_files": sample_files,
                    "sample_content": sample_content
                }
                
                print(f"  Contains {len(all_members)} total files/directories")
                print(f"  Text files: {len(txt_members)}")
                print(f"  JSON files: {len(json_members)}")
                print(f"  CSV files: {len(csv_members)}")
                print(f"  Nested archives: {len(nested_archives)}")
        except Exception as e:
            print(f"  Error analyzing {tar_filename}: {str(e)}")
            dataset_structures[os.path.basename(tar_file)] = {
                "type": "archive",
                "error": str(e)
            }

# Save structure information to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_structures, f, indent=2)

print("\n" + "=" * 80)
print(f"Analysis complete. Structure information saved to {output_file}")
print("=" * 80)

# Print summary of each dataset
print("\nDATASET STRUCTURE SUMMARY:")
for dataset, structure in dataset_structures.items():
    print(f"\n{dataset}:")
    
    if "type" in structure and structure["type"] == "jsonl":
        print(f"  Type: JSONL")
        print(f"  Records: {structure['num_records']}")
        if "keys" in structure:
            print(f"  Keys ({len(structure['keys'])}):")
            for key in structure["keys"]:
                print(f"    - {key['name']} ({key['type']})")
                if "sample_values" in key and key["sample_values"]:
                    print(f"      Sample values: {', '.join(str(v) for v in key['sample_values'][:2])}...")
    elif "columns" in structure:
        print(f"  Delimiter: '{structure['delimiter']}'")
        print(f"  Rows: {structure['num_rows']}")
        print(f"  Columns ({structure['num_columns']}):")
        
        for col in structure["columns"]:
            print(f"    - {col['name']} ({col['dtype']})")
            print(f"      Sample values: {', '.join(str(v) for v in col['sample_values'][:2])}...")
    else:
        print(f"  Type: {structure['type']}")
        print(f"  Total files: {structure['total_files']}")
        print(f"  Text files: {structure['text_files']}")
        print(f"  JSON files: {structure['json_files']}")
        print(f"  CSV files: {structure['csv_files']}")
        print(f"  Nested archives: {structure['nested_archives']}") 