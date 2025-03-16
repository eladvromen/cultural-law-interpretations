import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import os
import pickle
from pathlib import Path
import re
from difflib import SequenceMatcher


class DeterminationRemovalProcessor:
    """
    Processor for removing determination statements from refugee legal documents.
    
    This class implements a position-based heuristic to decide whether to remove
    only the determination sentence or all subsequent sentences based on where
    the determination appears in the document.
    """
    
    def __init__(self, 
                 sentences_path: str, 
                 determinations_path: str,
                 position_threshold: float = 0.7):
        """
        Initialize the processor with data paths and threshold.
        
        Args:
            sentences_path: Path to the sentences dataset pickle file
            determinations_path: Path to the determinations dataset pickle file
            position_threshold: Threshold for the position-based heuristic (default: 0.7)
        """
        self.sentences_path = sentences_path
        self.determinations_path = determinations_path
        self.position_threshold = position_threshold
        
        # Data containers
        self.sentences_df = None
        self.determinations_df = None
        self.document_sentence_counts = {}
        self.determination_dict = {}
        self.document_sentences = {}
        
        # Results
        self.matched_determinations = []
        self.removed_sentences = set()
        self.processed_sentences = None
        
        # Load and prepare data
        self._load_data()
        self._prepare_data()
    
    def _load_data(self):
        """Load the sentences and determinations datasets."""
        try:
            self.sentences_df = pd.read_pickle(self.sentences_path)
            self.determinations_df = pd.read_pickle(self.determinations_path)
            
            print(f"Loaded {len(self.sentences_df)} sentences and {len(self.determinations_df)} determinations.")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    
    def _prepare_data(self):
        """Prepare data structures for efficient processing."""
        # Compute total sentences per document
        self.document_sentence_counts = self.sentences_df.groupby('decisionID').size().to_dict()
        
        # Create determination lookup dictionary
        self.determination_dict = {}
        for _, row in self.determinations_df.iterrows():
            decision_id = row['decisionID']
            determination_text = row['extracted_sentences_determination']
            
            # Handle list format - could be a string representation of a list or an actual list
            if isinstance(determination_text, list):
                # Join all elements if it's a list
                determination_text = ' '.join(determination_text)
            elif isinstance(determination_text, str):
                # Clean up string that looks like a list
                determination_text = determination_text.strip("[]'\"")
                # Replace list separators if present
                determination_text = determination_text.replace("', '", " ")
            
            if decision_id not in self.determination_dict:
                self.determination_dict[decision_id] = []
            
            self.determination_dict[decision_id].append(determination_text)
        
        # Group sentences by document for faster access
        self.document_sentences = {
            decision_id: group.sort_values('sentence_order').reset_index(drop=True)
            for decision_id, group in self.sentences_df.groupby('decisionID')
        }
        
        print(f"Prepared data for {len(self.document_sentences)} documents.")
    
    def _find_matching_sentences(self, decision_id: str, determination_text: str) -> List[Tuple[int, float]]:
        """
        Find sentences that match the determination text using multiple strategies.
        
        Args:
            decision_id: The document ID
            determination_text: The determination text to match
            
        Returns:
            List of tuples (sentence_order, relative_position)
        """
        if decision_id not in self.document_sentences:
            return []
        
        document_df = self.document_sentences[decision_id]
        total_sentences = self.document_sentence_counts[decision_id]
        matches = []
        matched_sentences = set()  # Track which sentences we've already matched
        
        # Clean the determination text
        clean_det_text = determination_text.lower().strip()
        
        # Extract key determination phrases
        key_phrases = self._extract_key_phrases(clean_det_text)
        
        # Strategy 1: Exact substring match
        for _, row in document_df.iterrows():
            sentence_order = row['sentence_order']
            if sentence_order in matched_sentences:
                continue
            
            sentence_text = row['Text'].lower().strip()
            
            # Check for exact substring match
            if clean_det_text in sentence_text:
                relative_position = sentence_order / total_sentences
                matches.append((sentence_order, relative_position))
                matched_sentences.add(sentence_order)
        
        # Strategy 2: Key phrase matching
        if key_phrases:
            for _, row in document_df.iterrows():
                sentence_order = row['sentence_order']
                if sentence_order in matched_sentences:
                    continue
                
                sentence_text = row['Text'].lower().strip()
                
                # Check if any key phrase is in the sentence
                for phrase in key_phrases:
                    if phrase in sentence_text:
                        relative_position = sentence_order / total_sentences
                        matches.append((sentence_order, relative_position))
                        matched_sentences.add(sentence_order)
                        break
        
        # Strategy 3: Word-based similarity
        if not matches:
            # Split into words and create a set for faster matching
            det_words = set(re.findall(r'\b\w+\b', clean_det_text))
            significant_words = {word for word in det_words if len(word) > 3 and word not in 
                                {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they'}}
            
            for _, row in document_df.iterrows():
                sentence_order = row['sentence_order']
                if sentence_order in matched_sentences:
                    continue
                
                sentence_text = row['Text'].lower().strip()
                sentence_words = set(re.findall(r'\b\w+\b', sentence_text))
                
                # Calculate word overlap
                common_words = significant_words.intersection(sentence_words)
                if len(significant_words) > 0 and len(common_words) / len(significant_words) > 0.6:
                    relative_position = sentence_order / total_sentences
                    matches.append((sentence_order, relative_position))
                    matched_sentences.add(sentence_order)
        
        # Strategy 4: Sequence similarity for short determinations
        if not matches and len(clean_det_text) < 100:
            for _, row in document_df.iterrows():
                sentence_order = row['sentence_order']
                if sentence_order in matched_sentences:
                    continue
                
                sentence_text = row['Text'].lower().strip()
                
                # Use sequence matcher to find similarity ratio
                similarity = SequenceMatcher(None, clean_det_text, sentence_text).ratio()
                if similarity > 0.7:  # High similarity threshold
                    relative_position = sentence_order / total_sentences
                    matches.append((sentence_order, relative_position))
                    matched_sentences.add(sentence_order)
        
        return matches
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases that indicate a determination.
        
        Args:
            text: The determination text
            
        Returns:
            List of key phrases
        """
        key_phrases = []
        
        # Common determination phrases
        determination_indicators = [
            "appeal is allowed", 
            "appeal is dismissed",
            "appeal was allowed",
            "appeal was dismissed",
            "claim is rejected",
            "claim was rejected",
            "application is refused",
            "application is granted",
            "set aside the determination",
            "sets aside the determination",
            "refugee protection is conferred",
            "refugee protection is not conferred",
            "convention refugee",
            "person in need of protection"
        ]
        
        # Check for each indicator
        for indicator in determination_indicators:
            if indicator in text:
                key_phrases.append(indicator)
        
        # Look for specific patterns
        if "appeal" in text and "allow" in text:
            key_phrases.append("appeal is allowed")
        
        if "appeal" in text and "dismiss" in text:
            key_phrases.append("appeal is dismissed")
        
        if "set aside" in text or "sets aside" in text:
            key_phrases.append("set aside")
        
        if "determination" in text and "rpd" in text:
            key_phrases.append("determination of the rpd")
        
        return key_phrases
    
    def process(self) -> pd.DataFrame:
        """
        Process the documents to remove determination sentences based on position threshold.
        
        Returns:
            DataFrame of processed sentences with determinations removed
        """
        self.matched_determinations = []
        self.removed_sentences = set()
        
        # Add debugging info
        print(f"Number of documents with determinations: {len(self.determination_dict)}")
        print(f"Number of documents with sentences: {len(self.document_sentences)}")
        
        # Check overlap between determination documents and sentence documents
        determination_docs = set(self.determination_dict.keys())
        sentence_docs = set(self.document_sentences.keys())
        common_docs = determination_docs.intersection(sentence_docs)
        print(f"Documents with both determinations and sentences: {len(common_docs)}")
        
        # Print some examples to debug matching issues
        print("\nDEBUG: Sample data examples")
        sample_docs = list(common_docs)[:3]  # Take first 3 common documents
        
        for doc_id in sample_docs:
            print(f"\nDocument: {doc_id}")
            
            # Print sample determinations
            print("Sample determinations:")
            for i, det_text in enumerate(self.determination_dict[doc_id][:2]):  # First 2 determinations
                print(f"  {i+1}. {det_text[:100]}...")  # First 100 chars
            
            # Print sample sentences
            print("Sample sentences:")
            sentences_df = self.document_sentences[doc_id]
            for i, (_, row) in enumerate(sentences_df.iloc[:5].iterrows()):  # First 5 sentences
                print(f"  {i+1}. {row['Text'][:100]}...")  # First 100 chars
        
        # Process each document with determinations
        for decision_id, determination_texts in self.determination_dict.items():
            if decision_id not in self.document_sentences:
                continue
            
            # Add debugging for each document
            print(f"Processing document: {decision_id}")
            print(f"Number of determinations: {len(determination_texts)}")
            print(f"Number of sentences: {len(self.document_sentences[decision_id])}")
            
            # Find all matching sentences for each determination
            document_matches = []
            for determination_text in determination_texts:
                matches = self._find_matching_sentences(decision_id, determination_text)
                for sentence_order, position in matches:
                    document_matches.append((sentence_order, position, determination_text))
            
            print(f"Found {len(document_matches)} matches for document {decision_id}")
            
            # Sort by position (descending) to process from end to beginning
            document_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Apply removal logic based on position threshold
            for sentence_order, position, determination_text in document_matches:
                # Create a record of the matched determination
                self.matched_determinations.append({
                    'decisionID': decision_id,
                    'sentence_order': sentence_order,
                    'position': position,
                    'determination_text': determination_text,
                    'threshold_applied': self.position_threshold,
                    'removal_type': 'full_tail' if position > self.position_threshold else 'single'
                })
                
                if position > self.position_threshold:
                    # If in last 30% of document, remove this and all subsequent sentences
                    total_sentences = self.document_sentence_counts[decision_id]
                    for i in range(sentence_order, total_sentences + 1):
                        self.removed_sentences.add((decision_id, i))
                else:
                    # Otherwise, remove only the matching sentence
                    self.removed_sentences.add((decision_id, sentence_order))
        
        # Filter out removed sentences
        filtered_rows = []
        for _, row in self.sentences_df.iterrows():
            if (row['decisionID'], row['sentence_order']) not in self.removed_sentences:
                filtered_rows.append(row)
        
        self.processed_sentences = pd.DataFrame(filtered_rows)
        
        print(f"Processing complete. Removed {len(self.removed_sentences)} sentences.")
        print(f"Matched {len(self.matched_determinations)} determination statements.")
        
        return self.processed_sentences
    
    def get_removal_statistics(self) -> Dict:
        """
        Get statistics about the removal process.
        
        Returns:
            Dictionary with removal statistics
        """
        if not self.matched_determinations:
            return {"error": "No processing has been performed yet."}
        
        matched_df = pd.DataFrame(self.matched_determinations)
        
        stats = {
            "total_documents_processed": len(matched_df['decisionID'].unique()),
            "total_determinations_matched": len(matched_df),
            "total_sentences_removed": len(self.removed_sentences),
            "sentences_before": len(self.sentences_df),
            "sentences_after": len(self.processed_sentences),
            "removal_percentage": (len(self.removed_sentences) / len(self.sentences_df)) * 100,
            "threshold_used": self.position_threshold,
            "removal_types": {
                "full_tail": len(matched_df[matched_df['removal_type'] == 'full_tail']),
                "single": len(matched_df[matched_df['removal_type'] == 'single'])
            }
        }
        
        return stats
    
    def save_processed_data(self, output_path: str):
        """
        Save the processed sentences to a pickle file.
        
        Args:
            output_path: Path to save the processed sentences
        """
        if self.processed_sentences is None:
            raise ValueError("No processed data available. Run process() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed sentences
        self.processed_sentences.to_pickle(output_path)
        print(f"Saved processed sentences to {output_path}")
    
    def save_removal_metadata(self, output_path: str):
        """
        Save metadata about the removal process.
        
        Args:
            output_path: Path to save the metadata
        """
        # Instead of raising an error, create empty metadata if no matches
        if not self.matched_determinations:
            print("Warning: No determinations were matched during processing.")
            metadata = {
                "matched_determinations": [],
                "removal_statistics": {
                    "total_documents_processed": 0,
                    "total_determinations_matched": 0,
                    "total_sentences_removed": 0,
                    "sentences_before": len(self.sentences_df),
                    "sentences_after": len(self.processed_sentences),
                    "removal_percentage": 0,
                    "threshold_used": self.position_threshold,
                    "removal_types": {"full_tail": 0, "single": 0}
                }
            }
        else:
            metadata = {
                "matched_determinations": self.matched_determinations,
                "removal_statistics": self.get_removal_statistics()
            }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the metadata
        with open(output_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved removal metadata to {output_path}")


def run_threshold_validation(
    sentences_path: str,
    determinations_path: str,
    output_dir: str,
    thresholds: List[float] = [0.6, 0.65, 0.7, 0.75, 0.8],
    sample_ratio: float = 0.1,
    specific_decision_ids: List[str] = None
):
    """
    Run validation on multiple thresholds using a sample of the data.
    
    Args:
        sentences_path: Path to the sentences dataset
        determinations_path: Path to the determinations dataset
        output_dir: Directory to save outputs
        thresholds: List of thresholds to test
        sample_ratio: Ratio of documents to sample for validation (ignored if specific_decision_ids is provided)
        specific_decision_ids: List of specific decision IDs to use (overrides sample_ratio)
    
    Returns:
        Dictionary with results for each threshold
    """
    # Load the data
    sentences_df = pd.read_pickle(sentences_path)
    
    # Sample documents - either specific IDs or random sample
    if specific_decision_ids is not None:
        sampled_decision_ids = specific_decision_ids
        print(f"Using {len(sampled_decision_ids)} specific documents for validation")
    else:
        all_decision_ids = sentences_df['decisionID'].unique()
        sample_size = max(1, int(len(all_decision_ids) * sample_ratio))
        sampled_decision_ids = np.random.choice(all_decision_ids, size=sample_size, replace=False)
        print(f"Randomly sampled {len(sampled_decision_ids)} documents for validation")
    
    # Filter sentences to only include sampled documents
    sampled_sentences = sentences_df[sentences_df['decisionID'].isin(sampled_decision_ids)]
    
    # Create a temporary file for the sampled sentences
    sample_path = os.path.join(output_dir, "sampled_sentences.pkl")
    os.makedirs(output_dir, exist_ok=True)
    sampled_sentences.to_pickle(sample_path)
    
    # Save the list of sampled document IDs for reference
    with open(os.path.join(output_dir, "sampled_document_ids.txt"), 'w') as f:
        for doc_id in sampled_decision_ids:
            f.write(f"{doc_id}\n")
    
    # Process with each threshold
    results = {}
    for threshold in thresholds:
        print(f"\nProcessing with threshold: {threshold}")
        
        # Create processor with current threshold
        processor = DeterminationRemovalProcessor(
            sentences_path=sample_path,
            determinations_path=determinations_path,
            position_threshold=threshold
        )
        
        # Process the data
        processed_df = processor.process()
        
        # Save results
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        processor.save_processed_data(os.path.join(threshold_dir, "processed_sentences.pkl"))
        processor.save_removal_metadata(os.path.join(threshold_dir, "removal_metadata.pkl"))
        
        # Also save as CSV for easier manual inspection
        processed_df.to_csv(os.path.join(threshold_dir, "processed_sentences.csv"), index=False)
        
        # Create a human-readable summary of what was removed
        removal_summary = pd.DataFrame(processor.matched_determinations)
        if not removal_summary.empty:
            removal_summary.to_csv(os.path.join(threshold_dir, "removal_summary.csv"), index=False)
        
        # Store statistics
        results[threshold] = processor.get_removal_statistics()
    
    # Clean up temporary file
    if os.path.exists(sample_path):
        os.remove(sample_path)
    
    # Save comparative results
    comparative_results_path = os.path.join(output_dir, "threshold_comparison.pkl")
    with open(comparative_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as a readable format
    with open(os.path.join(output_dir, "threshold_comparison.txt"), 'w') as f:
        f.write("Threshold Comparison Results\n")
        f.write("===========================\n\n")
        for threshold, stats in results.items():
            f.write(f"Threshold: {threshold}\n")
            f.write("-" * 30 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\n")
    
    print(f"\nThreshold validation complete. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    # Fix the path construction to include immigration_fine_tuning
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Correct paths with immigration_fine_tuning included
    sentences_path = os.path.join(base_dir, "immigration_fine_tuning", "data", "clean_data", "all_sentences_clean.pkl")
    determinations_path = os.path.join(base_dir, "immigration_fine_tuning", "data", "processed", "determination_label_extracted_sentences.pkl")
    
    output_dir = os.path.join(base_dir, "immigration_fine_tuning", "data", "processed", "determination_removal")
    
    # Run threshold validation with exactly 15 documents
    validation_output_dir = os.path.join(output_dir, "threshold_validation")
    
    # Load sentences to get document IDs
    try:
        sentences_df = pd.read_pickle(sentences_path)
        all_decision_ids = sentences_df['decisionID'].unique()
        
        # Sample exactly 15 documents
        sample_size = min(15, len(all_decision_ids))
        sampled_decision_ids = np.random.choice(all_decision_ids, size=sample_size, replace=False)
        
        # Run validation with these specific documents
        results = run_threshold_validation(
            sentences_path=sentences_path,
            determinations_path=determinations_path,
            output_dir=validation_output_dir,
            specific_decision_ids=sampled_decision_ids  # Use our 15 sampled documents
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Checking if file exists: {os.path.exists(sentences_path)}")
        print(f"Checking base directory: {base_dir}")
        print("Please verify the correct path to your data files.")
