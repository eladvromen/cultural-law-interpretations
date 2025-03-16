import os
import json
import pandas as pd
import numpy as np
import tarfile
from typing import Dict, Any, List, Optional
import logging
import csv  # Import the csv module directly

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, data_dir: str, structure_file: str = "dataset_structure.json"):
        """
        Initialize the dataset processor.
        
        Args:
            data_dir: Directory containing the datasets
            structure_file: JSON file with dataset structure information
        """
        self.data_dir = data_dir
        self.structure_file = os.path.join(data_dir, structure_file)
        self.dataset_structure = self._load_structure()
        self.dataframes = {}
    
    def _load_structure(self) -> Dict[str, Any]:
        """Load the dataset structure from JSON file."""
        try:
            with open(self.structure_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading structure file: {e}")
            return {}
    
    def _get_file_config(self, filename: str) -> Dict[str, Any]:
        """Get configuration for a specific file from the structure."""
        if filename in self.dataset_structure:
            return self.dataset_structure[filename]
        else:
            logger.warning(f"No structure information found for {filename}")
            return {}
    
    def read_csv_safely(self, filename: str, fallback_options: bool = True) -> Optional[pd.DataFrame]:
        """
        Read a CSV file with proper error handling and fallback options.
        
        Args:
            filename: Name of the CSV file to read
            fallback_options: Whether to try fallback options if initial read fails
            
        Returns:
            DataFrame or None if all reading attempts fail
        """
        filepath = filename if os.path.isabs(filename) else os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        file_config = self._get_file_config(os.path.basename(filepath))
        
        # Get delimiter from config or default to semicolon for extracted files
        if "all_sentences_anonymized.csv" in filepath:
            delimiter = ";"  # Force semicolon for this specific file
        else:
            delimiter = file_config.get('delimiter', ',')
        
        # First attempt: Use the configuration from the structure file
        try:
            logger.info(f"Attempting to read {filepath} with delimiter '{delimiter}'")
            df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False)
            logger.info(f"Successfully read {filepath} with {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Error reading {filepath} with standard options: {e}")
            
            if not fallback_options:
                logger.error(f"Failed to read {filepath} and fallback options disabled")
                return None
            
            # Fallback options
            fallback_attempts = [
                # Try with Python engine which is more flexible
                {"engine": "python"},
                # Try with different quoting options - fixed to use csv module directly
                {"quoting": csv.QUOTE_NONE, "escapechar": "\\"},
                # Try with different encoding
                {"encoding": "utf-8"},
                # Try with different encoding and error handling
                {"encoding": "latin1"},
                # Try to skip bad lines
                {"on_bad_lines": "skip"},
                # Try with a different delimiter if the original one isn't semicolon
                {"delimiter": ";" if delimiter != ";" else ","},
                # Try with tab delimiter
                {"delimiter": "\t"},
                # Try with automatic delimiter detection
                {"delimiter": None, "engine": "python"}
            ]
            
            # Try each fallback option
            for i, options in enumerate(fallback_attempts):
                try:
                    logger.info(f"Fallback attempt {i+1} for {filepath} with options: {options}")
                    df = pd.read_csv(filepath, low_memory=False, **options)
                    logger.info(f"Successfully read {filepath} with fallback option {i+1}")
                    return df
                except Exception as e:
                    logger.warning(f"Fallback attempt {i+1} failed: {e}")
            
            # If all else fails, try reading the file line by line
            logger.info(f"Attempting to read {filepath} line by line")
            return self._read_line_by_line(filepath, delimiter)
    
    def _read_line_by_line(self, filepath: str, delimiter: str) -> Optional[pd.DataFrame]:
        """
        Read a CSV file line by line, skipping problematic lines.
        
        Args:
            filepath: Path to the CSV file
            delimiter: Delimiter to use
            
        Returns:
            DataFrame or None if reading fails
        """
        try:
            # Read the file as text
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Find the header line
            header = lines[0].strip().split(delimiter)
            
            # Process each line
            data = []
            for i, line in enumerate(lines[1:], 1):
                try:
                    # Split the line by delimiter
                    values = line.strip().split(delimiter)
                    
                    # If the number of values doesn't match the header, adjust
                    if len(values) > len(header):
                        # Too many values, combine extras
                        adjusted_values = values[:len(header)-1]
                        adjusted_values.append(delimiter.join(values[len(header)-1:]))
                        values = adjusted_values
                    elif len(values) < len(header):
                        # Too few values, pad with NaN
                        values.extend([np.nan] * (len(header) - len(values)))
                    
                    data.append(values)
                except Exception as e:
                    logger.warning(f"Error processing line {i+1}: {e}")
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=header)
            logger.info(f"Successfully read {filepath} line by line with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to read {filepath} line by line: {e}")
            return None
    
    def extract_csv_from_archive(self, archive_filename: str) -> Optional[str]:
        """
        Extract a CSV file from a tar archive.
        
        Args:
            archive_filename: Name of the archive file
            
        Returns:
            Path to the extracted CSV file or None if extraction fails
        """
        archive_path = os.path.join(self.data_dir, archive_filename)
        if not os.path.exists(archive_path):
            logger.error(f"Archive file not found: {archive_path}")
            return None
        
        # Determine the mode based on file extension
        mode = 'r:gz' if archive_filename.endswith('.tar.gz') else 'r:xz'
        
        try:
            # Create extraction directory
            extract_dir = os.path.join(self.data_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            # Open the archive
            with tarfile.open(archive_path, mode) as tar:
                # Find CSV files in the archive
                csv_members = [m for m in tar.getmembers() if m.name.endswith('.csv')]
                
                if not csv_members:
                    logger.warning(f"No CSV files found in {archive_filename}")
                    return None
                
                # Extract the first CSV file
                csv_member = csv_members[0]
                logger.info(f"Extracting {csv_member.name} from {archive_filename}")
                
                # Extract to the extraction directory
                tar.extract(csv_member, path=extract_dir)
                
                # Return the path to the extracted file
                extracted_path = os.path.join(extract_dir, csv_member.name)
                logger.info(f"Extracted to {extracted_path}")
                return extracted_path
                
        except Exception as e:
            logger.error(f"Error extracting CSV from {archive_filename}: {e}")
            return None
    
    def _convert_decision_id_to_int(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Safely convert decisionID column to integer type.
        
        Args:
            df: DataFrame containing decisionID column
            filename: Name of the file (for logging)
            
        Returns:
            DataFrame with converted decisionID column
        """
        if 'decisionID' in df.columns:
            logger.info(f"Converting decisionID to integer in {filename}")
            try:
                # First check if there are non-numeric values
                non_numeric_mask = pd.to_numeric(df['decisionID'], errors='coerce').isna() & df['decisionID'].notna()
                
                if non_numeric_mask.any():
                    # Log the non-numeric values
                    non_numeric_values = df.loc[non_numeric_mask, 'decisionID'].unique()
                    logger.warning(f"Found non-numeric decisionID values in {filename}: {non_numeric_values}")
                    
                    # Replace non-numeric values with a numeric placeholder (-999)
                    df.loc[non_numeric_mask, 'decisionID'] = -999
                    logger.info(f"Replaced non-numeric decisionID values with -999 in {filename}")
                
                # Now convert to integer
                df['decisionID'] = pd.to_numeric(df['decisionID'], errors='coerce').fillna(-1).astype(int)
                logger.info(f"Successfully converted decisionID to integer in {filename}")
            except Exception as e:
                logger.error(f"Error converting decisionID to integer in {filename}: {e}")
        
        return df
    
    def process_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Process all datasets in the data directory.
        
        Returns:
            Dictionary of dataframes
        """
        # Get all CSV files in the data directory
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        # Process each file
        for filename in csv_files:
            logger.info(f"Processing {filename}")
            df = self.read_csv_safely(filename)
            if df is not None:
                # Convert decisionID to integer
                df = self._convert_decision_id_to_int(df, filename)
                
                # Store the dataframe
                self.dataframes[filename] = df
                
                # Print basic info
                logger.info(f"{filename}: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"Columns: {', '.join(df.columns)}")
        
        # Process archives containing CSV files
        archive_files = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith(('.tar.gz', '.tar.xz'))
        ]
        
        for archive_filename in archive_files:
            logger.info(f"Processing archive: {archive_filename}")
            
            # Check if this archive contains CSV files
            archive_info = self._get_file_config(archive_filename)
            if archive_info.get('csv_files', 0) > 0:
                # Extract CSV from archive
                extracted_path = self.extract_csv_from_archive(archive_filename)
                
                if extracted_path:
                    # Get just the filename part
                    extracted_filename = os.path.basename(extracted_path)
                    
                    # Read the extracted CSV
                    df = self.read_csv_safely(extracted_path)
                    
                    if df is not None:
                        # Convert decisionID to integer
                        df = self._convert_decision_id_to_int(df, extracted_filename)
                        
                        # Store with a name that indicates it came from an archive
                        key = f"{os.path.splitext(archive_filename)[0]}_extracted"
                        self.dataframes[key] = df
                        
                        # Print basic info
                        logger.info(f"{key}: {len(df)} rows, {len(df.columns)} columns")
                        logger.info(f"Columns: {', '.join(df.columns)}")
        
        return self.dataframes
    
    def save_processed_dataframes(self, output_dir: str = None) -> None:
        """
        Save processed dataframes to pickle files for faster loading.
        
        Args:
            output_dir: Directory to save the pickle files (defaults to data_dir)
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        for filename, df in self.dataframes.items():
            # Create pickle filename
            pickle_filename = os.path.splitext(filename)[0] + '.pkl'
            pickle_path = os.path.join(output_dir, pickle_filename)
            
            # Save dataframe
            try:
                df.to_pickle(pickle_path)
                logger.info(f"Saved {pickle_filename}")
            except Exception as e:
                logger.error(f"Error saving {pickle_filename}: {e}")
    
    def load_specific_datasets(self, filenames: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load specific datasets by name.
        
        Args:
            filenames: List of CSV filenames to load
            
        Returns:
            Dictionary of dataframes
        """
        result = {}
        for filename in filenames:
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            logger.info(f"Loading {filename}")
            df = self.read_csv_safely(filename)
            if df is not None:
                # Convert decisionID to integer
                df = self._convert_decision_id_to_int(df, filename)
                
                result[filename] = df
                logger.info(f"{filename}: {len(df)} rows, {len(df.columns)} columns")
        
        return result


# Example usage
if __name__ == "__main__":
    # Set the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           "data", "asylex_data")
    
    # Create output directory for processed files
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the processor
    processor = DatasetProcessor(data_dir)
    
    # Load specific datasets
    specific_files = [
        "outcome_train_test_train_dataset_silver",
        "outcome_train_test_test_dataset_gold",
        "determination_label_extracted_sentences",
        "case_cover_case_cover_entities_and_decision_outcome",
        "main_and_case_cover_all_entities_inferred" 
    ]
    
    # Process the dataframes
    dataframes = processor.load_specific_datasets(specific_files)
    
    # Process the all_sentences_anonymized.tar.xz archive
    archive_filename = "all_sentences_anonymized.tar.xz"
    logger.info(f"Processing archive: {archive_filename}")
    extracted_path = processor.extract_csv_from_archive(archive_filename)
    
    if extracted_path:
        # Read the extracted CSV
        df = processor.read_csv_safely(extracted_path)
        
        if df is not None:
            # Convert decisionID to integer
            df = processor._convert_decision_id_to_int(df, "all_sentences_anonymized")
            
            # Store with a name that indicates it came from an archive
            key = "all_sentences_anonymized_extracted"
            dataframes[key] = df
            logger.info(f"{key}: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {', '.join(df.columns)}")
            
            # Save to pickle
            pickle_path = os.path.join(output_dir, f"{key}.pkl")
            try:
                df.to_pickle(pickle_path)
                logger.info(f"Saved {pickle_path}")
            except Exception as e:
                logger.error(f"Error saving {pickle_path}: {e}")
    
    # Save the processed dataframes
    for filename, df in dataframes.items():
        pickle_filename = os.path.splitext(filename)[0] + '.pkl'
        pickle_path = os.path.join(output_dir, pickle_filename)
        try:
            df.to_pickle(pickle_path)
            logger.info(f"Saved {pickle_path}")
        except Exception as e:
            logger.error(f"Error saving {pickle_path}: {e}")