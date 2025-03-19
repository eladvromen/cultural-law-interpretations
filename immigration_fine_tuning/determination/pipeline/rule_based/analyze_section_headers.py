import pandas as pd
import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import numpy as np
import json
from typing import List, Dict, Tuple, Set

class SectionHeaderAnalyzer:
    """
    Analyzes legal documents to identify and count section headers.
    """
    
    def __init__(self, min_words=1, max_words=6, min_count=5):
        """
        Initialize the analyzer with parameters for header identification.
        
        Args:
            min_words: Minimum number of words in a header
            max_words: Maximum number of words in a header
            min_count: Minimum count to include in results
        """
        self.min_words = min_words
        self.max_words = max_words
        self.min_count = min_count
        
        # Pattern to identify potential section headers (all caps with optional numbers)
        self.header_pattern = re.compile(r'^\s*([A-Z][A-Z\s\d\.\-\:]{2,})\s*$')
        
        # Common headers we're particularly interested in
        self.key_headers = {
            'determination': ['DETERMINATION', 'DECISION', 'DISPOSITION', 'FINDINGS'],
            'conclusion': ['CONCLUSION', 'CONCLUSIONS'],
            'analysis': ['ANALYSIS', 'ASSESSMENT', 'REASONS', 'REASONS FOR DECISION']
        }
        
        # Patterns to identify and exclude non-header elements
        self.exclude_patterns = [
            re.compile(r'^[A-Z]{1,2}\d+\s*-\s*\d+'),  # Case numbers like TA5-12331
            re.compile(r'^X+$'),                      # Redacted text like XXXXX
            re.compile(r'^X+\s+X+'),                  # Multiple redacted words
            re.compile(r'^\d+$'),                     # Just numbers
            re.compile(r'^[A-Z]\.$'),                 # Single letter with period
            re.compile(r'^[IVX]+\.$'),                # Roman numerals with period
            re.compile(r'^[A-Z]{1,2}\d+$')            # Short codes
        ]
        
        # Results storage
        self.header_counts = Counter()
        self.documents_processed = 0
        self.documents_with_headers = 0
        self.headers_by_document = {}
    
    def extract_potential_headers(self, text: str) -> List[str]:
        """
        Extract potential section headers from text.
        
        Args:
            text: Document text to analyze
            
        Returns:
            List of potential headers
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        lines = text.split('\n')
        potential_headers = []
        
        for line in lines:
            match = self.header_pattern.match(line)
            if match:
                header = match.group(1).strip()
                
                # Skip if it matches any exclude pattern
                if any(pattern.match(header) for pattern in self.exclude_patterns):
                    continue
                
                # Skip if it's mostly X's (redacted text)
                if header.count('X') > len(header) * 0.5:
                    continue
                
                # Check word count
                word_count = len(header.split())
                if self.min_words <= word_count <= self.max_words:
                    potential_headers.append(header)
        
        return potential_headers
    
    def process_document(self, doc_id: str, text: str) -> List[str]:
        """
        Process a single document to extract headers.
        
        Args:
            doc_id: Document identifier
            text: Document text
            
        Returns:
            List of headers found in the document
        """
        headers = self.extract_potential_headers(text)
        
        # Update counts
        for header in headers:
            self.header_counts[header] += 1
        
        # Store headers by document
        if headers:
            self.headers_by_document[doc_id] = headers
            self.documents_with_headers += 1
        
        self.documents_processed += 1
        
        return headers
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'full_text', id_column: str = 'decisionID'):
        """
        Process all documents in a dataframe.
        
        Args:
            df: Dataframe containing documents
            text_column: Column containing document text
            id_column: Column containing document identifiers
        """
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            if pd.isna(row[text_column]) or not isinstance(row[text_column], str):
                continue
                
            doc_id = row[id_column] if id_column in row and not pd.isna(row[id_column]) else f"doc_{idx}"
            self.process_document(doc_id, row[text_column])
    
    def get_top_headers(self, n: int = 50) -> List[Tuple[str, int]]:
        """
        Get the top N most common headers.
        
        Args:
            n: Number of top headers to return
            
        Returns:
            List of (header, count) tuples
        """
        return self.header_counts.most_common(n)
    
    def get_filtered_headers(self) -> Dict[str, int]:
        """
        Get headers that meet the minimum count threshold.
        
        Returns:
            Dictionary of headers and their counts
        """
        return {header: count for header, count in self.header_counts.items() if count >= self.min_count}
    
    def get_key_header_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for key headers we're interested in.
        
        Returns:
            Dictionary with stats for each key header category
        """
        stats = {}
        
        for category, headers in self.key_headers.items():
            category_stats = {}
            for header in headers:
                category_stats[header] = self.header_counts[header]
            stats[category] = category_stats
        
        return stats
    
    def plot_top_headers(self, n: int = 30, figsize: Tuple[int, int] = (12, 10), 
                         output_path: str = None):
        """
        Plot the top N headers by frequency.
        
        Args:
            n: Number of headers to include
            figsize: Figure size (width, height)
            output_path: Path to save the figure (if None, displays instead)
        """
        top_headers = self.get_top_headers(n)
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame(top_headers, columns=['Header', 'Count'])
        
        # Sort by count
        plot_df = plot_df.sort_values('Count', ascending=True)
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='Count', y='Header', data=plot_df, palette='viridis')
        
        # Add count labels
        for i, count in enumerate(plot_df['Count']):
            ax.text(count + 0.5, i, str(count), va='center')
        
        plt.title(f'Top {n} Section Headers by Frequency')
        plt.xlabel('Count')
        plt.ylabel('Header')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved plot to {output_path}")
        else:
            plt.show()
    
    def plot_key_header_categories(self, figsize: Tuple[int, int] = (15, 8), 
                                  output_path: str = None):
        """
        Plot the frequency of headers in our key categories.
        
        Args:
            figsize: Figure size (width, height)
            output_path: Path to save the figure (if None, displays instead)
        """
        # Prepare data for plotting
        categories = []
        headers = []
        counts = []
        
        for category, header_dict in self.get_key_header_stats().items():
            for header, count in header_dict.items():
                categories.append(category)
                headers.append(header)
                counts.append(count)
        
        # Create dataframe
        plot_df = pd.DataFrame({
            'Category': categories,
            'Header': headers,
            'Count': counts
        })
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='Header', y='Count', hue='Category', data=plot_df, palette='Set2')
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(str(int(p.get_height())), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', 
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.title('Frequency of Key Section Headers')
        plt.xlabel('Header')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved plot to {output_path}")
        else:
            plt.show()
    
    def save_results(self, output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        results = {
            'documents_processed': self.documents_processed,
            'documents_with_headers': self.documents_with_headers,
            'header_counts': dict(self.header_counts),
            'top_headers': dict(self.get_top_headers(100)),
            'key_header_stats': self.get_key_header_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved analysis results to {output_path}")
    
    def save_headers_by_document(self, output_path: str):
        """
        Save headers found in each document to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        with open(output_path, 'w') as f:
            json.dump(self.headers_by_document, f, indent=2)
        
        print(f"Saved headers by document to {output_path}")
    
    def get_true_section_headers(self, min_count: int = 5) -> Dict[str, int]:
        """
        Get headers that are likely true section headers (not case numbers or redacted info).
        
        Args:
            min_count: Minimum count to include
            
        Returns:
            Dictionary of headers and their counts
        """
        # Additional filtering for post-processing
        exclude_words = ['NIL', 'CORAM', 'AMENDED', 'PRIVATE', 'HUIS', 'CLOS']
        
        # Get all headers that meet the minimum count
        all_headers = self.get_filtered_headers()
        
        # Filter out headers that are likely not true section headers
        true_headers = {}
        for header, count in all_headers.items():
            # Skip if it contains any exclude words
            if any(word in header for word in exclude_words):
                continue
            
            # Skip if it's mostly numbers or special characters
            alpha_count = sum(1 for c in header if c.isalpha())
            if alpha_count < len(header) * 0.5:
                continue
            
            # Skip if it's very long (likely not a header)
            if len(header) > 30:
                continue
            
            true_headers[header] = count
        
        return true_headers

def main():
    # Get project root directory (3 levels up from this script)
    script_dir = Path(__file__).resolve()
    project_root = script_dir.parent.parent.parent.parent
    
    # Create output directory
    output_dir = script_dir.parent / "data_of_interest"
    output_dir.mkdir(exist_ok=True)
    
    # Load validation set
    validation_path = script_dir.parent / "validation_set.csv"
    
    if not validation_path.exists():
        print(f"Error: Validation file not found at {validation_path}")
        return
    
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded validation set with {len(validation_df)} cases")
    
    # Load test set from merged data using relative path
    test_path = project_root / "immigration_fine_tuning" / "data" / "merged" / "test_enriched.csv"
    
    if not test_path.exists():
        print(f"Error: Test data file not found at {test_path}")
        print("Checking if environment variable is set...")
        
        # Try using environment variable as fallback
        data_dir = os.environ.get("IMMIGRATION_DATA_DIR")
        if data_dir:
            test_path = Path(data_dir) / "merged" / "test_enriched.csv"
            if not test_path.exists():
                print(f"Error: Test data file not found at {test_path}")
                return
        else:
            print("Environment variable IMMIGRATION_DATA_DIR not set.")
            print("Please set this variable to the path of your data directory.")
            return
    
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"Loaded test data with {len(test_df)} cases")
    
    # Initialize analyzer
    analyzer = SectionHeaderAnalyzer(min_words=1, max_words=6, min_count=5)
    
    # Process validation set
    print("\nProcessing validation set...")
    analyzer.process_dataframe(validation_df, text_column='full_text', id_column='decisionID')
    
    # Process test set
    print("\nProcessing test set...")
    analyzer.process_dataframe(test_df, text_column='full_text', id_column='decisionID')
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Total documents processed: {analyzer.documents_processed}")
    print(f"Documents with headers: {analyzer.documents_with_headers} ({analyzer.documents_with_headers/analyzer.documents_processed:.1%})")
    print(f"Unique headers found: {len(analyzer.header_counts)}")
    print(f"Headers with count >= {analyzer.min_count}: {len(analyzer.get_filtered_headers())}")
    
    # Print top headers
    print("\nTop 20 Headers:")
    for header, count in analyzer.get_top_headers(20):
        print(f"  {header}: {count}")
    
    # Print key header stats
    print("\nKey Header Categories:")
    for category, headers in analyzer.get_key_header_stats().items():
        print(f"  {category.upper()}:")
        for header, count in headers.items():
            print(f"    {header}: {count}")
    
    # Create plots
    print("\nGenerating plots...")
    analyzer.plot_top_headers(n=30, output_path=output_dir / "top_headers.png")
    analyzer.plot_key_header_categories(output_path=output_dir / "key_headers.png")
    
    # Save results
    analyzer.save_results(output_dir / "header_analysis.json")
    analyzer.save_headers_by_document(output_dir / "headers_by_document.json")
    
    # Create a CSV with all headers and their counts
    headers_df = pd.DataFrame(analyzer.get_top_headers(1000), columns=['Header', 'Count'])
    headers_df.to_csv(output_dir / "all_headers.csv", index=False)
    print(f"Saved all headers to {output_dir / 'all_headers.csv'}")
    
    # Create separate analyses for validation and test sets
    print("\nGenerating separate analyses for validation and test sets...")
    
    # Validation set analysis
    validation_analyzer = SectionHeaderAnalyzer(min_words=1, max_words=6, min_count=3)
    validation_analyzer.process_dataframe(validation_df, text_column='full_text', id_column='decisionID')
    validation_analyzer.save_results(output_dir / "validation_header_analysis.json")
    validation_headers_df = pd.DataFrame(validation_analyzer.get_top_headers(1000), columns=['Header', 'Count'])
    validation_headers_df.to_csv(output_dir / "validation_headers.csv", index=False)
    
    # Test set analysis
    test_analyzer = SectionHeaderAnalyzer(min_words=1, max_words=6, min_count=3)
    test_analyzer.process_dataframe(test_df, text_column='full_text', id_column='decisionID')
    test_analyzer.save_results(output_dir / "test_header_analysis.json")
    test_headers_df = pd.DataFrame(test_analyzer.get_top_headers(1000), columns=['Header', 'Count'])
    test_headers_df.to_csv(output_dir / "test_headers.csv", index=False)
    
    # Compare headers between validation and test sets
    validation_headers = set(validation_analyzer.get_filtered_headers().keys())
    test_headers = set(test_analyzer.get_filtered_headers().keys())
    
    common_headers = validation_headers.intersection(test_headers)
    validation_only = validation_headers - test_headers
    test_only = test_headers - validation_headers
    
    print(f"\nHeader comparison between validation and test sets:")
    print(f"  Common headers: {len(common_headers)}")
    print(f"  Validation-only headers: {len(validation_only)}")
    print(f"  Test-only headers: {len(test_only)}")
    
    # Save comparison results
    comparison = {
        'common_headers': list(common_headers),
        'validation_only': list(validation_only),
        'test_only': list(test_only)
    }
    
    with open(output_dir / "header_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\nAnalysis complete. Results saved to:", output_dir)

    # Get true section headers
    true_headers = analyzer.get_true_section_headers(min_count=10)
    print(f"\nTrue Section Headers (count >= 10):")
    for header, count in sorted(true_headers.items(), key=lambda x: x[1], reverse=True):
        print(f"  {header}: {count}")

    # Create a CSV with true section headers
    true_headers_df = pd.DataFrame(sorted(true_headers.items(), key=lambda x: x[1], reverse=True), 
                                  columns=['Header', 'Count'])
    true_headers_df.to_csv(output_dir / "true_section_headers.csv", index=False)
    print(f"Saved true section headers to {output_dir / 'true_section_headers.csv'}")

    # Create a plot of true section headers
    plt.figure(figsize=(12, 10))
    top_true_headers = dict(sorted(true_headers.items(), key=lambda x: x[1], reverse=True)[:30])
    plt.barh(list(top_true_headers.keys()), list(top_true_headers.values()), color='teal')
    plt.xlabel('Count')
    plt.ylabel('Header')
    plt.title('Top 30 True Section Headers')
    plt.tight_layout()
    plt.savefig(output_dir / "true_section_headers.png")
    print(f"Saved true section headers plot to {output_dir / 'true_section_headers.png'}")

if __name__ == "__main__":
    main() 