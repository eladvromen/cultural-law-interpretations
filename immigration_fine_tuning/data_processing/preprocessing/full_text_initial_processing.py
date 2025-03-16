import os
import re
import tarfile
import pandas as pd
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_metadata_from_filename(filename):
    """
    Extract year and decision ID from filename pattern [year]canlii[decisionID].txt
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        tuple: (year, decision_id) or (None, None) if pattern doesn't match
    """
    pattern = r'(\d{4})canlii(\d+)\.txt'
    match = re.match(pattern, os.path.basename(filename))
    
    if match:
        year = match.group(1)
        decision_id = match.group(2)
        return year, decision_id
    else:
        return None, None

def process_case_archive(archive_path, output_path=None, sample_size=None):
    """
    Process the case archive and convert to DataFrame with [year, decisionID, full_text]
    
    Args:
        archive_path (str): Path to the tar.gz archive
        output_path (str, optional): Path to save the output JSON
        sample_size (int, optional): Number of files to process (for testing)
        
    Returns:
        pd.DataFrame: DataFrame with columns [year, decision_id, full_text]
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive file not found: {archive_path}")
    
    logger.info(f"Processing archive: {archive_path}")
    
    data = []
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get all text files from the archive
        members = [m for m in tar.getmembers() if m.name.endswith('.txt')]
        
        # Limit to sample size if specified
        if sample_size:
            members = members[:sample_size]
            
        for member in tqdm(members, desc="Processing files"):
            try:
                # Extract filename
                filename = os.path.basename(member.name)
                
                # Extract year and decision ID from filename
                year, decision_id = extract_metadata_from_filename(filename)
                
                if year and decision_id:
                    # Extract file content
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='replace')
                        
                        # Add to data list
                        data.append({
                            'year': year,
                            'decisionID': decision_id,
                            'full_text': content
                        })
                else:
                    logger.warning(f"Couldn't parse metadata from filename: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing file {member.name}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    logger.info(f"Processed {len(df)} files successfully")
    
    # Save to JSON if output path is provided
    if output_path:
        logger.info(f"Saving to {output_path}")
        # Convert DataFrame to list of dictionaries for JSON serialization
        records = df.to_dict(orient='records')
        
        # Save as JSON with proper encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
    return df

def main():
    """
    Main function to run the preprocessing pipeline
    """
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    archive_path = os.path.join(base_dir, 'immigration_fine_tuning/data/asylex_data/cases_anonymized_txt_raw.tar.gz')
    output_dir = os.path.join(base_dir, 'immigration_fine_tuning/data/processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'case_texts_processed.json')
    
    # Process the archive
    df = process_case_archive(archive_path, output_path)
    
    logger.info(f"Preprocessing complete. DataFrame shape: {df.shape}")
    
    # Display sample
    if not df.empty:
        logger.info("\nSample data:")
        logger.info(df.head(3))

if __name__ == "__main__":
    main()
