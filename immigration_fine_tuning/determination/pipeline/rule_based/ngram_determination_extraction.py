import pandas as pd
import numpy as np
import re
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm
from collections import Counter, defaultdict
from functools import lru_cache
import random

from .base_determination_extractor import BaseDeterminationExtractor

class NgramDeterminationExtractor(BaseDeterminationExtractor):
    """
    A model that uses n-gram matching with balanced precision and recall.
    """
    
    def __init__(self, min_ngram_size=2, max_ngram_size=4, match_threshold=0.75):
        """
        Initialize the N-gram determination extractor with optimized data structures.
        
        Args:
            min_ngram_size: Minimum size of n-grams to use
            max_ngram_size: Maximum size of n-grams to use
            match_threshold: Threshold for considering a match (0.0-1.0)
        """
        super().__init__()
        
        # N-gram parameters
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size
        self.match_threshold = match_threshold
        
        # Storage for known determination sentences and their n-grams
        self.determination_sentences = set()
        self.ngram_index = defaultdict(set)
        self.determination_list = []
        self.determination_map = {}  # For O(1) lookups
        
        # Define structural markers for determination sentences - expanded for better recall
        self.determination_markers = [
            # Appeal outcomes
            "appeal is", "appeals are", "appeal was", "appeals were", 
            "dismissing the appeal", "dismiss the appeal", "dismissed the appeal",
            "appeal be dismissed", "appeal should be dismissed",
            "appeal has been", "appeal of", "appeal for",
            
            # Orders
            "removal order is", "order is set aside", "order is vacated",
            
            # Applications
            "application is", "application to", "application for",
            
            # Claims
            "claim is", "claims are", "reject the claim", "reject his claim",
            
            # Status determinations
            "appellant has not lost", "appellant has lost", "permanent resident status",
            
            # Other common phrases
            "sufficient humanitarian", "humanitarian and compassionate",
            "claimant is not", "refugee protection is", 
            "convention refugee", "person in need of protection",
            "panel finds", "panel determines", "panel concludes"
        ]
        
        # Direct match patterns - more inclusive patterns
        self.direct_match_patterns = [
            # Appeal outcomes
            re.compile(r"\bappeal\s+is\s+(?:allowed|dismissed|refused|rejected|not\s+allowed)"),
            re.compile(r"\bappeals?\s+(?:are|were|was)\s+(?:allowed|dismissed|refused|rejected)"),
            re.compile(r"\bdismiss(?:ing|ed)?\s+the\s+(?:appellant'?s?\s+)?appeal"),
            re.compile(r"\bappeal\s+(?:be|should\s+be|has\s+been)\s+dismissed"),
            re.compile(r"\bappeal\s+(?:of|for)\s+\w+\s+is\s+(?:allowed|dismissed)"),
            
            # Removal order outcomes
            re.compile(r"\bremoval\s+order\s+is\s+set\s+aside"),
            
            # Status determinations
            re.compile(r"\bappellant\s+has\s+not\s+lost\s+(?:his|her|their)\s+permanent\s+resident\s+status"),
            
            # H&C considerations
            re.compile(r"\bsufficient\s+humanitarian\s+and\s+compassionate\s+considerations"),
            
            # Explicit determinations
            re.compile(r"\bthe\s+panel\s+(?:finds|determines|concludes)\s+that\s+(?:the\s+)?(?:appellant|claimant|appeal)"),
            re.compile(r"\b(?:I|we)\s+(?:find|determine|conclude)\s+that"),
            re.compile(r"\bclaimant\s+is\s+(?:not\s+)?a\s+(?:convention\s+refugee|person\s+in\s+need\s+of\s+protection)"),
            re.compile(r"\breject\s+(?:his|her|their)\s+claims?")
        ]
        
        # Expanded common phrases list for better recall
        self.common_determination_phrases = [
            # Appeal outcomes - standard
            "appeal is allowed", "appeal is dismissed", "appeal is refused",
            "appeals are dismissed", "appeals are allowed", "appeal was dismissed",
            
            # Appeal outcomes - variations
            "appeal is hereby dismissed", "dismissing the appeal", "dismissed the appeal",
            "dismiss the appellant's appeal", "appeal is terminated",
            "appeal is declared abandoned", "appeal be dismissed", "appeal should be dismissed",
            
            # Application outcomes
            "application to reopen the appeal is granted",
            "application is granted", "application is refused",
            
            # Removal order outcomes
            "removal order is set aside", "order is vacated",
            
            # Status determinations
            "appellant has not lost his permanent resident status",
            "appellant has not lost her permanent resident status",
            
            # H&C considerations
            "sufficient humanitarian and compassionate considerations",
            "humanitarian and compassionate grounds",
            
            # Refugee determinations
            "claimant is not a convention refugee",
            "claimant is a convention refugee",
            "claimant is not a person in need of protection",
            "neither a convention refugee nor a person in need of protection",
            "confirm the decision", "substitute my determination",
            "reject his claim", "reject her claim", "reject their claims"
        ]
        
        # Create optimized data structures
        self.determination_set = set()  # For fast membership testing
        self.combined_direct_pattern = self._combine_patterns(self.direct_match_patterns)
        self.common_phrases_lower = {phrase.lower() for phrase in self.common_determination_phrases}
        self.determination_markers_lower = {marker.lower() for marker in self.determination_markers}
    
    def _combine_patterns(self, patterns):
        """Combine multiple regex patterns into a single pattern for efficiency."""
        pattern_strings = []
        for pattern in patterns:
            # Extract the pattern string without the leading/trailing markers
            pattern_str = pattern.pattern
            pattern_strings.append(f"(?:{pattern_str})")
        
        # Combine with OR operator and compile
        combined = re.compile("|".join(pattern_strings))
        return combined
    
    @lru_cache(maxsize=100000)  # Increased cache size
    def clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced preprocessing and caching."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Fix run-together words (common OCR errors) - all in one pass
        for original, replacement in [
            ('isallowed', 'is allowed'),
            ('isdismissed', 'is dismissed'),
            ('ishereby', 'is hereby'),
            ('declaredabandoned', 'declared abandoned'),
            ('tosubsection', 'to subsection'),
            ('wasallowed', 'was allowed'),
            ('bedismissed', 'be dismissed')
        ]:
            text = text.replace(original, replacement)
        
        # Replace apostrophes with a temporary character
        text = re.sub(r'(\w)\'(\w)', r'\1§\2', text)
        
        # Remove punctuation except for the temporary character in one operation
        text = re.sub(r'[^\w\s§]', ' ', text)
        
        # Restore apostrophes
        text = text.replace('§', "'")
        
        # Normalize whitespace in one operation
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_ngrams(self, text: str) -> Dict[int, List[str]]:
        """
        Generate n-grams from text more efficiently.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping n-gram sizes to lists of n-grams
        """
        # Clean the text once
        cleaned_text = self.clean_text(text)
        
        # Split into words
        words = cleaned_text.split()
        
        # Skip if too short
        if len(words) < self.min_ngram_size:
            return {}
        
        # Generate n-grams of different sizes more efficiently
        ngrams = {}
        for n in range(self.min_ngram_size, min(self.max_ngram_size + 1, len(words) + 1)):
            ngrams[n] = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        return ngrams
    
    def index_determination(self, determination: str) -> int:
        """
        Add a determination sentence to the index with optimized lookups.
        
        Args:
            determination: Determination sentence
            
        Returns:
            Index of the determination
        """
        # Clean and normalize
        cleaned = self.clean_text(determination)
        
        # Skip very short or empty texts
        if len(cleaned.split()) < 2:
            return -1
        
        # Check if already indexed using set for O(1) lookup
        if cleaned in self.determination_set:
            return self.determination_map.get(cleaned, -1)
        
        # Add to our collections
        self.determination_set.add(cleaned)
        self.determination_list.append(cleaned)
        sentence_idx = len(self.determination_list) - 1
        self.determination_map[cleaned] = sentence_idx
        
        # Generate and index n-grams
        ngrams = self.generate_ngrams(cleaned)
        for n, grams in ngrams.items():
            for gram in grams:
                self.ngram_index[gram].add(sentence_idx)
        
        return sentence_idx
    
    def load_training_examples(self, train_path: str, test_path: str, use_chunking: bool = True) -> None:
        """
        Load training examples from both train and test files.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            use_chunking: Whether to use chunking for large files
        """
        print("Loading training examples for ngram model...")
        start_time = time.time()
        
        examples = []
        
        # Load examples from training data
        examples.extend(self._extract_examples_from_file(train_path, use_chunking))
        
        # Load examples from test data
        examples.extend(self._extract_examples_from_file(test_path, use_chunking))
        
        # Add fallback common phrases
        examples.extend(self.common_determination_phrases)
        
        # Remove duplicates and very short examples
        examples = [ex for ex in list(set(examples)) if isinstance(ex, str) and len(ex.strip()) > 3]
        
        print(f"Extracted {len(examples)} unique example sentences")
        
        # Print sample of the extracted examples for verification
        if examples:
            print("\nSample of extracted determinations:")
            for i, example in enumerate(examples[:10]):
                print(f"  {i+1}. {example}")
        
        # Index all valid examples
        indexed_count = 0
        for example in tqdm(examples, desc="Indexing determinations"):
            if self.index_determination(example) >= 0:
                indexed_count += 1
        
        print(f"Indexed {indexed_count} unique determination sentences")
        print(f"Created index with {len(self.ngram_index)} unique n-grams")
        
        elapsed = time.time() - start_time
        print(f"Finished loading and processing examples in {elapsed:.2f} seconds")
        
        # Mark as loaded
        self._data_loaded = True
    
    def _balanced_precision_recall_match(self, cleaned_sentence: str) -> Tuple[float, Optional[str], Optional[int]]:
        """
        More balanced matching function that improves recall while maintaining precision.
        
        Args:
            cleaned_sentence: Already cleaned sentence text
            
        Returns:
            Tuple of (score, matched_text, matched_index)
        """
        # Skip very short sentences
        if len(cleaned_sentence.split()) < 3:
            return 0.0, None, None
        
        # Check for exact match first
        if cleaned_sentence in self.determination_set:
            idx = self.determination_map.get(cleaned_sentence)
            return 1.0, self.determination_list[idx], idx
        
        # Check for strong substring matches
        max_substring_score = 0.0
        max_substring_idx = -1
        
        # Check against sample of determinations
        sample_size = min(500, len(self.determination_list))
        sample_indices = random.sample(range(len(self.determination_list)), sample_size) if len(self.determination_list) > sample_size else range(len(self.determination_list))
        
        cleaned_len = len(cleaned_sentence)
        
        # More lenient length ratio constraints
        for idx in sample_indices:
            determination = self.determination_list[idx]
            det_len = len(determination)
            
            # More permissive length checks - 0.6 instead of 0.7
            if det_len < cleaned_len * 0.6 or det_len > cleaned_len * 1.4:
                continue
            
            # Substring checks
            if determination in cleaned_sentence:
                ratio = det_len / cleaned_len
                if ratio > max_substring_score:
                    max_substring_score = ratio
                    max_substring_idx = idx
            elif cleaned_sentence in determination:
                ratio = cleaned_len / det_len
                if ratio > max_substring_score:
                    max_substring_score = ratio
                    max_substring_idx = idx
        
        # Return if we found a good substring match
        if max_substring_score >= self.match_threshold:
            return max_substring_score, self.determination_list[max_substring_idx], max_substring_idx
        
        # Generate n-grams
        ngrams = self.generate_ngrams(cleaned_sentence)
        
        # Find candidate matches
        candidate_counts = Counter()
        total_ngrams = 0
        
        # Process larger n-grams first (more specific)
        for n in range(self.max_ngram_size, self.min_ngram_size - 1, -1):
            if n not in ngrams:
                continue
                
            grams = ngrams[n]
            total_ngrams += len(grams)
            
            for gram in grams:
                # Less strict filtering of common n-grams
                matching_sentences = self.ngram_index.get(gram, set())
                if len(matching_sentences) > 80:  # Increased from 50 to 80
                    continue
                    
                for sentence_idx in matching_sentences:
                    candidate_counts[sentence_idx] += 1
        
        # Find best match
        if candidate_counts and total_ngrams > 0:
            best_idx, best_count = candidate_counts.most_common(1)[0]
            best_score = best_count / total_ngrams
            
            # More lenient threshold
            if best_score >= self.match_threshold:
                return best_score, self.determination_list[best_idx], best_idx
        
        return 0.0, None, None
    
    def extract_potential_determinations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential determination sentences with balanced precision and recall.
        
        Args:
            text: Document text
            
        Returns:
            List of potential determination sentences with metadata
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        matches = []
        
        # Find determination sections with a wider net
        determination_sections = []
        for i, sentence in enumerate(sentences):
            upper_sentence = sentence.strip().upper()
            # Look for explicit determination sections
            if (upper_sentence.startswith(("DETERMINATION", "DECISION", "NOTICE OF DECISION", "DISPOSITION")) or
                upper_sentence in ("DETERMINATION", "DECISION", "DISPOSITION")):
                # Include this and following sentences as a determination section
                determination_sections.append((i, min(i+8, len(sentences))))
            
            # Also look for conclusion sections
            if upper_sentence.startswith(("CONCLUSION", "ANALYSIS AND CONCLUSION", "FINDINGS AND CONCLUSION")):
                determination_sections.append((i, min(i+5, len(sentences))))
        
        # Process sentences with more balanced matching criteria
        for i, sentence in enumerate(sentences):
            position = i / max(1, len(sentences) - 1)
            sentence_lower = sentence.lower()
            
            # Check if in a determination section
            in_determination_section = any(start <= i < end for start, end in determination_sections)
            
            # Check for determination markers - more lenient approach
            has_marker = False
            for marker in self.determination_markers_lower:
                if marker in sentence_lower:
                    has_marker = True
                    break
            
            # 1. Direct pattern matching (most precise)
            if self.combined_direct_pattern.search(sentence_lower):
                matches.append({
                    'text': sentence,
                    'matched_text': sentence,
                    'score': 20.0,
                    'position': position,
                    'length': len(sentence),
                    'match_type': 'direct_pattern'
                })
                continue
            
            # 2. Common phrase matching
            matched_phrase = False
            for phrase in self.common_phrases_lower:
                if phrase in sentence_lower:
                    matches.append({
                        'text': sentence,
                        'matched_text': phrase,
                        'score': 18.0,
                        'position': position,
                        'length': len(sentence),
                        'match_type': 'phrase_match'
                    })
                    matched_phrase = True
                    break
            
            if matched_phrase:
                continue
            
            # 3. Exact match with indexed determinations
            cleaned = self.clean_text(sentence)
            if cleaned in self.determination_set:
                idx = self.determination_map.get(cleaned)
                matches.append({
                    'text': sentence,
                    'matched_text': self.determination_list[idx],
                    'score': 20.0,
                    'position': position,
                    'length': len(sentence),
                    'match_type': 'exact'
                })
                continue
            
            # 4. N-gram matching for sentences that have markers or are in determination sections
            # More lenient approach that increases recall
            if has_marker or in_determination_section:
                score, matched_text, matched_idx = self._balanced_precision_recall_match(cleaned)
                
                if score > 0:
                    matches.append({
                        'text': sentence,
                        'matched_text': matched_text,
                        'score': score * 20.0,
                        'position': position,
                        'length': len(sentence),
                        'match_type': 'ngram'
                    })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches
    
    def process_case(self, text: str) -> Dict[str, Any]:
        """
        Process a case to extract determinations with balanced precision and recall.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with extracted determinations
        """
        # Make sure data is loaded
        self.ensure_data_loaded()
        
        start_time = time.time()
        
        # Extract determinations
        potential_determinations = self.extract_potential_determinations(text)
        
        # Filter more inclusively for better recall
        determinations = []
        seen_texts = set()
        
        for det in sorted(potential_determinations, key=lambda x: x['score'], reverse=True):
            # Basic deduplication
            if det['text'].lower() in seen_texts:
                continue
            
            # Threshold for inclusion
            if det['match_type'] in ['direct_pattern', 'exact', 'phrase_match'] or det['score'] >= 15.0:
                determinations.append(det)
                seen_texts.add(det['text'].lower())
        
        # Less aggressive adaptive thresholding
        direct_matches_count = sum(1 for d in determinations if d['match_type'] in ['direct_pattern', 'exact', 'phrase_match'])
        
        # If we found many direct matches, be more selective with n-gram matches
        if direct_matches_count >= 3:
            n_gram_matches = [d for d in determinations if d['match_type'] == 'ngram' and d['score'] >= 17.0]
            direct_matches = [d for d in determinations if d['match_type'] != 'ngram']
            determinations = direct_matches + n_gram_matches
        
        # Optionally reduce the max number of results
        if len(determinations) > 6:
            determinations = determinations[:6]
        
        # Track performance
        self.processed_cases += 1
        self.processing_time += time.time() - start_time
        
        return {'extracted_determinations': determinations}


def check_training_examples(train_path):
    """Check a few examples from training data."""
    print("\nChecking sample determinations from training data:")
    
    try:
        # Read just the first chunk
        df_sample = next(pd.read_csv(train_path, chunksize=100))
        
        if 'extracted_sentences_determination' in df_sample.columns:
            count = 0
            for val in df_sample['extracted_sentences_determination'].dropna()[:5]:
                if isinstance(val, str):
                    print(f"Example {count+1}: {val}")
                    count += 1
    except Exception as e:
        print(f"Error checking examples: {e}")


def main():
    """
    Process the validation set using the NgramDeterminationExtractor.
    """
    start_time = time.time()
    
    # Create the extractor model with lower threshold
    model = NgramDeterminationExtractor(min_ngram_size=2, max_ngram_size=4, match_threshold=0.5)  # Reduced from 0.65
    
    # Get paths using relative references
    base_dir = Path(__file__).parent.parent.parent.parent
    train_path = base_dir / "data" / "merged" / "train_enriched.csv"
    test_path = base_dir / "data" / "merged" / "test_enriched.csv"
    validation_path = Path(__file__).parent / "preprocessed_validation_set.csv"
    
    # Check if paths exist
    if not train_path.exists():
        print(f"Train data not found at {train_path}")
        print("Using fallback paths...")
        # Try windows-specific path (adjust as needed)
        train_path = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/train_enriched.csv")
        test_path = Path("C:/Users/shil6369/cultural-law-interpretations/immigration_fine_tuning/data/merged/test_enriched.csv")
    
    # Call this at the beginning of main
    check_training_examples(train_path)
    
    # Load training examples 
    model.load_training_examples(str(train_path), str(test_path), use_chunking=True)
    
    # Test on a sample case (optional)
    if validation_path.exists():
        print("\nTesting model on validation set sample...")
        validation_df = pd.read_csv(validation_path)
        
        if len(validation_df) > 0:
            sample = validation_df.iloc[0]
            if 'cleaned_text' in sample:
                result = model.process_case(sample['cleaned_text'])
                print(f"Sample case ID: {sample.get('decisionID', 'unknown')}")
                print(f"Found {len(result['extracted_determinations'])} determination sentences")
                for i, det in enumerate(result['extracted_determinations']):
                    print(f"{i+1}. Score: {det['score']:.1f}, Match: {det['match_type']}")
                    print(f"   Text: {det['text'][:100]}..." if len(det['text']) > 100 else f"   Text: {det['text']}")
                    print(f"   Matched: {det['matched_text']}")
    
    # Import evaluation module and run evaluation
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from evaluation import run_evaluation
        
        print("\nRunning evaluation on validation set...")
        results, metrics, analysis_paths = run_evaluation(
            model, 
            validation_file=validation_path,
            output_dir=Path(__file__).parent,
            analyze_failures=True
        )
        
        # Get performance stats
        perf_stats = model.get_performance_stats()
        
        # Save results
        output_path = Path(__file__).parent / "ngram_determination_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'performance': perf_stats,
                'total_runtime': f"{time.time() - start_time:.2f} seconds",
                'case_results': [
                    {
                        'case_id': r['case_id'],
                        'metrics': r['metrics'],
                        'extracted_count': r['metrics']['extracted_count'],
                        'expected_count': r['metrics']['expected_count']
                    } for r in results
                ],
                'analysis_files': analysis_paths
            }, f, indent=2)
        
        print(f"Saved results to {output_path}")
        print("\nPerformance Stats:")
        for k, v in perf_stats.items():
            print(f"  {k}: {v}")
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 