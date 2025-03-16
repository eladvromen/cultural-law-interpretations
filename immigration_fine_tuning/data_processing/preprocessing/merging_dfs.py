import pandas as pd
import os
import sys

def get_ordered_sentences_by_decision_id(all_sentences_df):
    """
    Process all sentences and create a dictionary mapping each decisionID to its ordered sentences.
    
    Args:
        all_sentences_df (pd.DataFrame): DataFrame containing all sentences with decisionID and sentence_order
        
    Returns:
        dict: Dictionary mapping decisionID to joined sentence text
    """
    print("Creating ordered sentences dictionary...")
    
    # Ensure decisionID is string type for consistent mapping
    all_sentences_df['decisionID'] = all_sentences_df['decisionID'].astype(str)
    
    # Ensure sentence_order is numeric
    all_sentences_df['sentence_order'] = pd.to_numeric(all_sentences_df['sentence_order'], errors='coerce')
    
    # Sort by decisionID and sentence_order
    sorted_sentences = all_sentences_df.sort_values(by=['decisionID', 'sentence_order'])
    
    # Group by decisionID and join sentences
    sentence_dict = {}
    for decision_id, group in sorted_sentences.groupby('decisionID'):
        # Join sentences with spaces - using 'Text' column which is the actual column name
        joined_text = ' '.join(group['Text'].fillna('').astype(str))
        sentence_dict[decision_id] = joined_text
    
    # Print some sample keys to debug
    sample_keys = list(sentence_dict.keys())[:5]
    print(f"Sample dictionary keys: {sample_keys}")
    print(f"Created sentence dictionary for {len(sentence_dict)} unique decision IDs")
    
    return sentence_dict

def process_determination_dataset(determination_df):
    """
    Process determination dataset to group by decisionID and stack determinations in a list.
    
    Args:
        determination_df (pd.DataFrame): DataFrame containing determination data
        
    Returns:
        pd.DataFrame: Processed determination DataFrame with stacked determinations
    """
    print("Processing determination dataset...")
    
    # Ensure decisionID is string type for consistent mapping
    determination_df['decisionID'] = determination_df['decisionID'].astype(str)
    
    # Group by decisionID and aggregate determinations into lists
    grouped_determinations = determination_df.groupby('decisionID').agg({
        'extracted_sentences_determination': lambda x: list(x)
    }).reset_index()
    
    # Add count column
    grouped_determinations['determination_count'] = grouped_determinations['extracted_sentences_determination'].apply(len)
    
    print(f"Processed determination dataset: {len(grouped_determinations)} unique decision IDs")
    print(f"Sample determination counts: {grouped_determinations['determination_count'].value_counts().head()}")
    
    # Print a sample of the processed data
    print("\nSample processed determination data:")
    sample_data = grouped_determinations.head(2)
    for _, row in sample_data.iterrows():
        print(f"Decision ID: {row['decisionID']}")
        print(f"Determination count: {row['determination_count']}")
        print(f"First few determinations: {row['extracted_sentences_determination'][:2]}")
        print()
    
    return grouped_determinations

def merge_dataset(dataset_name):
    """
    Merge a dataset with other datasets to enrich it with metadata.
    The function performs left merges to ensure the number of records remains unchanged.
    Uses 'decisionID' as the primary key for merging.
    
    Args:
        dataset_name (str): Name of the dataset to enrich ('test' or 'train')
    
    Returns:
        pd.DataFrame: The enriched dataset
    """
    print(f"Starting merging pipeline for {dataset_name} dataset...")
    
    # Define absolute paths to data directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(base_dir, "immigration_fine_tuning", "data", "clean_data")
    processed_dir = os.path.join(base_dir, "immigration_fine_tuning", "data", "processed")
    
    print(f"Using data directory: {data_dir}")
    print(f"Using processed directory: {processed_dir}")
    
    # Load datasets
    print("Loading datasets...")
    
    # Base dataset (test or train)
    base_df = pd.read_csv(os.path.join(data_dir, f"{dataset_name}_clean.csv"))
    print(f"Initial {dataset_name} dataset shape: {base_df.shape}")
    print(f"{dataset_name} dataset columns: {base_df.columns.tolist()}")
    
    # Case texts dataset
    case_texts_df = pd.read_csv(os.path.join(data_dir, "case_texts_clean.csv"))
    print(f"Case texts dataset shape: {case_texts_df.shape}")
    print(f"Case texts columns: {case_texts_df.columns.tolist()}")
    
    # Determination dataset
    determination_df = pd.read_csv(os.path.join(data_dir, "determination_clean.csv"))
    print(f"Determination dataset shape: {determination_df.shape}")
    print(f"Determination columns: {determination_df.columns.tolist()}")
    
    # Process determination dataset to stack determinations by decisionID
    processed_determination_df = process_determination_dataset(determination_df)
    
    # All entities inferred dataset
    all_entities_infered = pd.read_pickle(os.path.join(processed_dir, "main_and_case_cover_all_entities_inferred.pkl"))
    print(f"All entities inferred dataset shape: {all_entities_infered.shape}")
    print(f"All entities columns: {all_entities_infered.columns.tolist()}")
    
    # Load all sentences dataset
    all_sentences_df = pd.read_pickle(os.path.join(data_dir, "all_sentences_clean.pkl"))
    print(f"All sentences dataset shape: {all_sentences_df.shape}")
    print(f"All sentences columns: {all_sentences_df.columns.tolist()}")
    
    # Create ordered sentences dictionary for efficient lookup
    sentence_dict = get_ordered_sentences_by_decision_id(all_sentences_df)
    
    # Use 'decisionID' as the primary key for merging
    id_col = 'decisionID'
    
    # Check if the ID column exists in each dataset
    for df_name, df in [(f"{dataset_name}_df", base_df), ("case_texts_df", case_texts_df), 
                        ("processed_determination_df", processed_determination_df),
                        ("all_entities_infered", all_entities_infered)]:
        if id_col in df.columns:
            print(f"'{id_col}' found in {df_name}")
            # Check for duplicates in each dataset
            dups = df[df.duplicated(subset=[id_col], keep=False)]
            if not dups.empty:
                print(f"  - Found {len(dups)} rows with duplicate '{id_col}' values in {df_name}")
                print(f"  - Number of unique '{id_col}' values: {df[id_col].nunique()} out of {len(df)} rows")
        else:
            print(f"WARNING: '{id_col}' NOT found in {df_name}")
    
    # Start merging process
    print(f"\nStarting merge operations for {dataset_name}...")
    merged_df = base_df.copy()
    
    # Check for 'decision_outcome' column in base_df
    has_decision_outcome = 'decision_outcome' in merged_df.columns
    if has_decision_outcome:
        print(f"Found 'decision_outcome' column in {dataset_name} dataset")
    
    # Merge with case_texts
    if id_col in base_df.columns and id_col in case_texts_df.columns:
        # Remove duplicates from case_texts before merging
        case_texts_unique = case_texts_df.drop_duplicates(subset=[id_col], keep='first')
        print(f"Case texts after removing duplicates: {case_texts_unique.shape}")
        
        # Check for 'decision_outcome' column in case_texts_unique
        if 'decision_outcome' in case_texts_unique.columns:
            print("Found 'decision_outcome' column in case_texts dataset, renaming to 'decision_outcome_estimated'")
            case_texts_unique = case_texts_unique.rename(columns={'decision_outcome': 'decision_outcome_estimated'})
        
        merged_df = pd.merge(merged_df, case_texts_unique, on=id_col, how='left')
        print(f"After merging with case_texts: {merged_df.shape}")
    else:
        print(f"Warning: Could not merge with case_texts due to missing '{id_col}' column")
    
    # Merge with processed determination dataset
    if id_col in merged_df.columns and id_col in processed_determination_df.columns:
        print(f"Merging with processed determination dataset (shape: {processed_determination_df.shape})...")
        
        # Convert IDs to string for consistent merging
        merged_df[id_col] = merged_df[id_col].astype(str)
        processed_determination_df[id_col] = processed_determination_df[id_col].astype(str)
        
        # Perform the merge
        merged_df = pd.merge(merged_df, processed_determination_df, on=id_col, how='left')
        print(f"After merging with processed determination: {merged_df.shape}")
        
        # Fill NaN values in determination_count with 0
        if 'determination_count' in merged_df.columns:
            merged_df['determination_count'] = merged_df['determination_count'].fillna(0).astype(int)
            print(f"Determination count distribution: {merged_df['determination_count'].value_counts().head()}")
        
        # Handle the extracted_sentences_determination column
        if 'extracted_sentences_determination' in merged_df.columns:
            print("Fixing extracted_sentences_determination column...")
            
            # Create a new list to store the fixed values
            fixed_determinations = []
            
            # Process each row individually
            for i, val in enumerate(merged_df['extracted_sentences_determination']):
                try:
                    # Check if the value is a scalar NaN
                    if isinstance(val, float) and pd.isna(val):
                        fixed_determinations.append([])
                    # Check if it's already a list
                    elif isinstance(val, list):
                        fixed_determinations.append(val)
                    # Handle string representation of lists (from previous processing)
                    elif isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                        try:
                            # Try to evaluate the string as a list
                            import ast
                            fixed_val = ast.literal_eval(val)
                            if isinstance(fixed_val, list):
                                fixed_determinations.append(fixed_val)
                            else:
                                fixed_determinations.append([val])
                        except:
                            # If evaluation fails, treat as a single string
                            fixed_determinations.append([val])
                    # Handle any other case
                    else:
                        fixed_determinations.append([str(val)])
                except Exception as e:
                    print(f"Error processing row {i}, value: {type(val)}, error: {e}")
                    # Default to empty list on error
                    fixed_determinations.append([])
            
            # Replace the column with the fixed values
            merged_df['extracted_sentences_determination'] = fixed_determinations
            
            # Verify the fix worked
            null_count = sum(1 for x in merged_df['extracted_sentences_determination'] if len(x) == 0)
            print(f"Rows with empty determination lists: {null_count}")
            print(f"Rows with non-empty determination lists: {len(merged_df) - null_count}")
    else:
        print(f"Warning: Could not merge with processed determination due to missing '{id_col}' column")
    
    # Merge with all_entities_infered
    if id_col in merged_df.columns and id_col in all_entities_infered.columns:
        # Investigate all_entities_infered dataset
        print(f"\nInvestigating all_entities_infered dataset:")
        print(f"Number of rows: {len(all_entities_infered)}")
        print(f"Number of unique '{id_col}' values: {all_entities_infered[id_col].nunique()}")
        
        # Check how many base IDs are in all_entities_infered
        base_ids = set(base_df[id_col].unique())
        entities_ids = set(all_entities_infered[id_col].unique())
        common_ids = base_ids.intersection(entities_ids)
        print(f"Number of {dataset_name} IDs found in all_entities_infered: {len(common_ids)} out of {len(base_ids)}")
        
        # Remove duplicates from all_entities_infered before merging
        all_entities_unique = all_entities_infered.drop_duplicates(subset=[id_col], keep='first')
        print(f"All entities after removing duplicates: {all_entities_unique.shape}")
        
        # Check for 'decision_outcome' column in all_entities_unique
        if 'decision_outcome' in all_entities_unique.columns:
            print("Found 'decision_outcome' column in all_entities dataset, renaming to 'decision_outcome_estimated'")
            all_entities_unique = all_entities_unique.rename(columns={'decision_outcome': 'decision_outcome_estimated'})
        
        # Ensure consistent types for the merge key
        print(f"Converting {id_col} to string type in all_entities_unique for consistent merging")
        all_entities_unique[id_col] = all_entities_unique[id_col].astype(str)
        merged_df[id_col] = merged_df[id_col].astype(str)
        
        # Print data types to verify
        print(f"merged_df['{id_col}'] dtype: {merged_df[id_col].dtype}")
        print(f"all_entities_unique['{id_col}'] dtype: {all_entities_unique[id_col].dtype}")
        
        # Perform the merge
        merged_df = pd.merge(merged_df, all_entities_unique, on=id_col, how='left')
        print(f"After merging with all_entities_infered: {merged_df.shape}")
        
        # Verify no duplicates in merged_df
        if merged_df.duplicated(subset=[id_col]).any():
            print(f"WARNING: Found duplicates in merged_df after all merges")
            print(f"Number of duplicate rows: {merged_df.duplicated(subset=[id_col]).sum()}")
            # Remove duplicates if they exist
            merged_df = merged_df.drop_duplicates(subset=[id_col], keep='first')
            print(f"Merged dataset after removing duplicates: {merged_df.shape}")
    else:
        print(f"Warning: Could not merge with all_entities_infered due to missing '{id_col}' column")
    
    # Add sentence_text column from the ordered sentences dictionary
    print("\nAdding sentence_text column from all_sentences dataset...")
    
    # Convert decisionID to string for consistent mapping
    merged_df[id_col] = merged_df[id_col].astype(str)
    
    # Print some sample IDs from both datasets to debug
    merged_sample_ids = merged_df[id_col].head(5).tolist()
    print(f"Sample merged dataset IDs: {merged_sample_ids}")
    
    # Check if these IDs exist in the sentence dictionary
    for sample_id in merged_sample_ids:
        print(f"ID {sample_id} exists in sentence_dict: {sample_id in sentence_dict}")
    
    # Add sentence_text column using the dictionary for efficient lookup
    merged_df['sentence_text'] = merged_df[id_col].map(sentence_dict)
    
    # Check how many records got sentence text
    sentence_text_count = merged_df['sentence_text'].notna().sum()
    print(f"Added sentence_text to {sentence_text_count} out of {len(merged_df)} records ({sentence_text_count/len(merged_df)*100:.2f}%)")
    
    # If no records got sentence text, try a different approach
    if sentence_text_count == 0:
        print("No records got sentence text. Trying a different approach...")
        
        # Try direct merge instead of dictionary mapping
        print("Creating temporary DataFrame from sentence dictionary...")
        sentence_df = pd.DataFrame({
            'decisionID': list(sentence_dict.keys()),
            'sentence_text': list(sentence_dict.values())
        })
        
        # Merge with the sentence DataFrame
        print(f"Merging with sentence DataFrame (shape: {sentence_df.shape})...")
        merged_df = pd.merge(merged_df, sentence_df, on='decisionID', how='left')
        
        # Check again how many records got sentence text
        sentence_text_count = merged_df['sentence_text'].notna().sum()
        print(f"After direct merge: Added sentence_text to {sentence_text_count} out of {len(merged_df)} records ({sentence_text_count/len(merged_df)*100:.2f}%)")
    
    # Clean up any remaining duplicate columns (e.g., decision_outcome_x, decision_outcome_y)
    # This handles cases where pandas automatically renamed columns during merge
    columns_to_check = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
    if columns_to_check:
        print(f"\nFound {len(columns_to_check)} columns with _x or _y suffixes, cleaning up...")
        
        for col in columns_to_check:
            base_col = col[:-2]  # Remove _x or _y suffix
            
            if col.endswith('_x'):
                # If it's from the base dataset (with _x suffix), keep it with the original name
                if f"{base_col}_y" in merged_df.columns:
                    print(f"Renaming {col} to {base_col} and {base_col}_y to {base_col}_estimated")
                    merged_df = merged_df.rename(columns={col: base_col, f"{base_col}_y": f"{base_col}_estimated"})
            elif col.endswith('_y') and f"{base_col}_x" not in merged_df.columns:
                # If only the _y version exists, rename it to estimated
                print(f"Renaming {col} to {base_col}_estimated")
                merged_df = merged_df.rename(columns={col: f"{base_col}_estimated"})
    
    # Specifically handle decision_outcome columns
    if 'decision_outcome_x' in merged_df.columns and 'decision_outcome_y' in merged_df.columns:
        print("Renaming decision_outcome_x to decision_outcome and decision_outcome_y to decision_outcome_estimated")
        merged_df = merged_df.rename(columns={
            'decision_outcome_x': 'decision_outcome',
            'decision_outcome_y': 'decision_outcome_estimated'
        })
    
    # Print column count to verify merges
    print(f"\nFinal merged dataset has {merged_df.shape[1]} columns")
    print(f"Column names: {merged_df.columns.tolist()}")
    
    # Verify that the number of records hasn't changed
    if merged_df.shape[0] == base_df.shape[0]:
        print(f"\nSuccess: The merged dataset has the same number of records as the original {dataset_name} dataset")
    else:
        print(f"\nWarning: The merged dataset has {merged_df.shape[0]} records, but the original {dataset_name} dataset had {base_df.shape[0]} records")
        
        # If we still have more records than the base dataset, force it to match
        if merged_df.shape[0] > base_df.shape[0]:
            print(f"Forcing merged dataset to have the same number of records as {dataset_name} dataset...")
            # Ensure we have all the base IDs
            base_ids = set(base_df[id_col])
            merged_ids = set(merged_df[id_col])
            missing_ids = base_ids - merged_ids
            if missing_ids:
                print(f"WARNING: {len(missing_ids)} {dataset_name} IDs are missing from the merged dataset")
            
            # Keep only rows with IDs from the base dataset
            merged_df = merged_df[merged_df[id_col].isin(base_df[id_col])]
            # If there are still duplicates, keep only the first occurrence
            if len(merged_df) > len(base_df):
                merged_df = merged_df.drop_duplicates(subset=[id_col], keep='first')
            
            print(f"Merged dataset after forcing: {merged_df.shape}")
    
    # Save the merged dataset
    output_dir = os.path.join(base_dir, "immigration_fine_tuning", "data", "merged")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle first (preserves all data types including lists)
    pickle_path = os.path.join(output_dir, f"{dataset_name}_enriched.pkl")
    merged_df.to_pickle(pickle_path)
    print(f"Merged dataset saved to {pickle_path}")
    
    # For CSV, convert list columns to strings to avoid serialization issues
    csv_df = merged_df.copy()
    
    # Convert list columns to strings for CSV export
    list_columns = [col for col in csv_df.columns if csv_df[col].apply(lambda x: isinstance(x, list)).any()]
    for col in list_columns:
        print(f"Converting list column '{col}' to string for CSV export")
        csv_df[col] = csv_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{dataset_name}_enriched.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Merged dataset (with converted list columns) saved to {csv_path}")
    
    return merged_df

def merge_datasets():
    """
    Merge both test and train datasets with other datasets to enrich them with metadata.
    """
    # Process test dataset
    test_enriched = merge_dataset("test")
    print("\n" + "="*80 + "\n")
    
    # Process train dataset
    train_enriched = merge_dataset("train")
    
    return test_enriched, train_enriched

if __name__ == "__main__":
    test_enriched, train_enriched = merge_datasets()
    print("\nMerging pipeline completed for both test and train datasets.")
    print(f"Test enriched shape: {test_enriched.shape}")
    print(f"Train enriched shape: {train_enriched.shape}")
