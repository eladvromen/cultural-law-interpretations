import pandas as pd
import re
import json
from pathlib import Path
import os
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm
from functools import lru_cache
import time
from collections import Counter

from base_determination_extractor import BaseDeterminationExtractor

class BasicDeterminationExtractor(BaseDeterminationExtractor):
    """
    A rule-based model for extracting determination sentences from immigration decisions.
    The model uses patterns learned from training data and common determination structures.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define pattern categories once with appropriate metadata
        self.pattern_categories = {
            # Category name: (priority, patterns)
            'exact_match': (20, [
                # List of exact phrases for direct string matching
                "appeal is allowed", "appeal is dismissed", "appeal is refused",
                "appeal was dismissed", "appeals are dismissed", "appeals were dismissed",
                "appeal be dismissed", "appeal should be dismissed",
                "removal order is set aside", "appeal has been abandoned",
                "dismiss this appeal", "dismissing the appeal", "appeal is not allowed",
                "appellant has not lost his permanent resident status",
                "appellant has not lost her permanent resident status",
                "sufficient humanitarian and compassionate considerations"
            ]),
            
            'standard_phrases': (15, [
                # High-confidence patterns
                r"(?:the\s+)?appeal\s+is\s+(?:allowed|dismissed)",
                r"(?:the\s+)?appeals?\s+(?:are|were)\s+dismissed",
                r"(?:the\s+)?removal\s+order\s+is\s+set\s+aside",
                r"appellant\s+has\s+not\s+lost\s+(?:his|her|their)\s+permanent\s+resident\s+status",
                r"sufficient\s+humanitarian\s+and\s+compassionate\s+(?:considerations|reasons)"
            ]),
            
            'appeal_outcomes': (10, [
                r"(?:the\s+)?appeal\s+is\s+(?:allowed|dismissed|refused|rejected|not\s+allowed)",
                r"(?:the\s+)?appeals?\s+(?:are|were|was)\s+(?:allowed|dismissed|refused|rejected|not\s+allowed)",
                r"(?:the\s+)?appeal\s+(?:be|to\s+be|should\s+be)\s+dismissed",
                r"(?:the\s+)?appeals?\s+(?:were\s+to\s+be)\s+dismissed",
                r"(?:the\s+)?appeal\s+(?:has\s+been|is)\s+(?:abandoned|determined)",
                r"(?:dismiss|dismissing)\s+(?:the|this)\s+appeal",
                r"appeal\s+(?:of|for)\s+[a-z\s\.]+\s+is\s+allowed",
                r"appeal\s+is\s*,\s*therefore\s*,\s*allowed",
                r"appeal\s+is\s+allowed\s*\.\s*the",
                r"appeal\s+is\s+not\s+allowed",
                r"appeal\s+was\s+not\s+allowed"
            ]),
            
            'removal_orders': (10, [
                r"(?:the\s+)?removal\s+order\s+is\s+set\s+aside"
            ]),
            
            'applications': (10, [
                r"(?:the\s+)?application\s+(?:for|to)\s+withdraw(?:al)?\s+is\s+(?:dismissed|allowed|granted|refused)",
                r"(?:the\s+)?application\s+to\s+withdraw\s+the\s+appeal\s+is\s+an\s+abuse\s+of\s+process",
                r"dismiss\s+the\s+appellant's\s+application"
            ]),
            
            'status_determinations': (10, [
                r"appellant\s+has\s+not\s+lost\s+(?:his|her|their)\s+permanent\s+resident\s+status"
            ]),
            
            'hc_considerations': (10, [
                r"sufficient\s+humanitarian\s+and\s+compassionate\s+(?:considerations|reasons)"
            ]),
            
            'general_determinations': (5, [
                r"(?:I|we|the panel|the member|the board)\s+(?:therefore\s+)?(?:find|conclude|determine)",
                r"(?:I|we|the panel|the member|the board)\s+(?:am|are)\s+(?:not\s+)?satisfied",
                r"(?:refugee\s+protection|asylum)\s+is\s+(?:granted|conferred|not granted|refused)",
                r"(?:the\s+)?decision\s+(?:of|by)\s+the\s+(?:RPD|RAD|IRB|Board)\s+is\s+(?:set aside|affirmed|upheld|vacated)"
            ])
        }
        
        # Keywords that indicate determination sentences
        self.determination_keywords = [
            "therefore", "accordingly", "consequently", "as a result", 
            "for these reasons", "for the foregoing reasons", "thus",
            "in conclusion", "to conclude", "in the result"
        ]
        
        # Learned patterns from examples will be stored here
        self.learned_patterns = []
        
        # Direct matching set (for O(1) lookups)
        self.exact_determinations = set(self.pattern_categories['exact_match'][1])
        
        # Compile patterns once
        self._compile_optimized_patterns()
    
    def _compile_optimized_patterns(self):
        """Compile all patterns into optimized regex objects."""
        # Store compiled patterns by category
        self.compiled_patterns = {}
        
        # Convert exact match patterns to lowercase for case-insensitive matching
        self.exact_match_lowercase = [pattern.lower() for pattern in self.pattern_categories['exact_match'][1]]
        
        # Compile regex patterns by category
        for category, (priority, patterns) in self.pattern_categories.items():
            if category == 'exact_match':
                continue  # Skip - we use direct string matching for these
                
            # Combine patterns within each category
            combined_patterns = []
            for pattern in patterns:
                # Strip existing case-insensitivity flag if present
                if pattern.startswith('(?i)'):
                    pattern = pattern[4:]
                combined_patterns.append(f"(?:{pattern})")
            
            # Compile the combined pattern with case-insensitivity
            try:
                self.compiled_patterns[category] = re.compile(f"(?i){'|'.join(combined_patterns)}")
            except re.error as e:
                print(f"Warning: Could not compile patterns for {category}: {e}")
                # Fall back to individual patterns
                self.compiled_patterns[category] = []
                for i, pattern in enumerate(patterns):
                    try:
                        self.compiled_patterns[category].append(
                            re.compile(f"(?i){pattern}")
                        )
                    except re.error:
                        print(f"  - Skipping invalid pattern: {pattern}")
    
    def clean_example(self, example):
        """Clean up an example determination for learning."""
        if not isinstance(example, str):
            return None
            
        # Clean up the example
        example = example.strip()
        
        # Remove common wrapping patterns
        example = re.sub(r"^\[\'|\'\]$", "", example)  # Remove ['...'] wrapper
        example = re.sub(r"^\"|\"\s*$", "", example)   # Remove "..." wrapper
        
        # Normalize spacing
        example = re.sub(r"\s+", " ", example)
        
        # Fix common OCR errors
        example = re.sub(r"(?i)isallowed", "is allowed", example)
        example = re.sub(r"(?i)isset", "is set", example)
        example = re.sub(r"(?i)isallowedin", "is allowed in", example)
        example = re.sub(r"(?i)allowedforjin", "allowed for jin", example)
        example = re.sub(r"(?i)allowed\.the", "allowed. The", example)
        
        return example if example else None
    
    def load_training_examples(self, train_path, test_path, use_chunking=True):
        """Load and process training examples from data files."""
        print("Loading training examples for basic determination model...")
        start_time = time.time()
        
        # Extract examples from both files
        train_examples = self._extract_examples_from_file(train_path, use_chunking)
        test_examples = self._extract_examples_from_file(test_path, use_chunking)
        
        # Combine and deduplicate
        all_examples = list(set(train_examples + test_examples))
        
        # Filter out very short or empty examples
        valid_examples = [ex for ex in all_examples if isinstance(ex, str) and len(ex.strip()) > 10]
        
        print(f"Extracted {len(valid_examples)} unique example sentences")
        
        # Learn from examples
        self.learn_from_examples(valid_examples)
        
        elapsed = time.time() - start_time
        print(f"Finished loading and processing examples in {elapsed:.2f} seconds")
        
        # Mark as loaded
        self._data_loaded = True
        
        return len(valid_examples)

    def _extract_examples_from_file(self, file_path, use_chunking):
        """Extract determination examples from a data file."""
        examples = []
        
        # Define chunk processor function
        def process_chunk(chunk):
            chunk_examples = []
            
            if 'extracted_sentences_determination' not in chunk.columns:
                return chunk_examples
            
            for val in chunk['extracted_sentences_determination'].dropna():
                if not isinstance(val, str):
                    continue
                
                # Try parsing as JSON with error handling
                parsed_examples = self._parse_determination_value(val)
                chunk_examples.extend(parsed_examples)
            
            return chunk_examples
        
        # Load and process data
        if use_chunking:
            print(f"Loading data from {file_path} using chunking...")
            chunk_size = 5000
            try:
                for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
                    examples.extend(process_chunk(chunk))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Loading data from {file_path} in one go...")
            try:
                df = pd.read_csv(file_path, low_memory=False)
                examples.extend(process_chunk(df))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return examples

    def _parse_determination_value(self, val):
        """Parse a determination value from the dataset."""
        if not isinstance(val, str) or not val.strip():
            return []
        
        examples = []
        
        # Try to parse as JSON
        if val.startswith('[') or val.startswith('{'):
            try:
                # Handle single and double quotes
                parsed = json.loads(val.replace("'", '"'))
                
                if isinstance(parsed, list):
                    # Handle list of strings or dicts
                    for item in parsed:
                        if isinstance(item, str):
                            examples.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            examples.append(item['text'])
                        else:
                            examples.append(str(item))
                elif isinstance(parsed, dict) and 'text' in parsed:
                    examples.append(parsed['text'])
                else:
                    # Add the whole string as fallback
                    examples.append(val)
            except:
                # If parsing fails, treat as plain text
                examples.append(val)
        else:
            # Plain text
            examples.append(val)
        
        return examples
    
    def learn_from_examples(self, examples):
        """
        Learn patterns from example determination sentences.
        
        Args:
            examples: List of example determination sentences
        """
        # Process and clean all examples first
        clean_examples = []
        for example in examples:
            clean_example = self.clean_example(example)
            if clean_example and len(clean_example) > 5:  # Minimum length check
                clean_examples.append(clean_example)
                # Add to exact determinations for direct matching
                self.exact_determinations.add(clean_example)
        
        print(f"Added {len(self.exact_determinations)} exact determination sentences for direct matching")
        
        # Extract common patterns from examples
        appeal_outcome_patterns = set()
        status_determination_patterns = set()
        other_patterns = set()
        
        for example in clean_examples:
            lower_example = example.lower()
            
            # Group into categories for more efficient pattern matching
            if "appeal" in lower_example and any(outcome in lower_example for outcome in 
                                               ["allowed", "dismissed", "refused"]):
                appeal_outcome_patterns.add(example)
            elif "appellant" in lower_example and "status" in lower_example:
                status_determination_patterns.add(example)
            elif any(term in lower_example for term in 
                    ["humanitarian", "removal order", "refugee", "protection"]):
                other_patterns.add(example)
        
        # Create a smaller number of more efficient patterns
        self._add_grouped_patterns(appeal_outcome_patterns, "appeal outcomes")
        self._add_grouped_patterns(status_determination_patterns, "status determinations")
        self._add_grouped_patterns(other_patterns, "other determinations")
        
        print(f"Compiled {len(self.learned_patterns)} optimized pattern groups from examples")
    
    def _add_grouped_patterns(self, examples, category):
        """Create efficient grouped patterns from examples."""
        if not examples:
            return
            
        # For efficiency, we'll use a combined pattern approach
        patterns = []
        
        # First, extract some unique patterns
        templates = set()
        for example in examples:
            # Add the exact example as a pattern
            templates.add(example.lower())
            
            # Create a more generic version by replacing names with placeholders
            # Instead of using problematic look-behind with variable width, use split and join
            parts = example.lower()
            
            # Handle "appeal of NAME is allowed" pattern
            if "appeal of " in parts and " is " in parts:
                # Split and rejoin with placeholder
                before, after = parts.split("appeal of ", 1)
                if " is " in after:
                    name_part, end_part = after.split(" is ", 1)
                    # Include only if name part looks like a name (not too long)
                    if len(name_part.split()) <= 3:
                        generic = f"{before}appeal of [NAME] is {end_part}"
                        templates.add(generic)
            
            # Handle "appeal for NAME is allowed" pattern
            if "appeal for " in parts and " is " in parts:
                before, after = parts.split("appeal for ", 1)
                if " is " in after:
                    name_part, end_part = after.split(" is ", 1)
                    if len(name_part.split()) <= 3:
                        generic = f"{before}appeal for [NAME] is {end_part}"
                        templates.add(generic)
            
            # Handle specific file numbers
            if "file " in parts:
                before, after = parts.split("file ", 1)
                file_parts = after.split()
                if file_parts and re.match(r'^[a-z0-9\-]+$', file_parts[0]):
                    rest = " ".join(file_parts[1:])
                    generic = f"{before}file [FILE] {rest}"
                    templates.add(generic)
        
        # Create regex patterns from templates
        for template in templates:
            # Replace placeholders with regex patterns
            pattern = re.escape(template)
            pattern = pattern.replace('\\[NAME\\]', '[a-z0-9\\s\\.\\-]+')
            pattern = pattern.replace('\\[FILE\\]', '[a-z0-9\\-]+')
            patterns.append(f"(?:{pattern})")
        
        # Combine into a single pattern if we have a reasonable number
        if patterns:
            if len(patterns) <= 50:  # Limit for reasonable regex performance
                try:
                    combined_pattern = re.compile(f"(?i){'|'.join(patterns)}")
                    self.learned_patterns.append((category, combined_pattern))
                except re.error as e:
                    print(f"Warning: Couldn't compile combined pattern for {category}: {e}")
                    # Fallback: add individual patterns
                    for i, pattern in enumerate(patterns):
                        try:
                            self.learned_patterns.append((f"{category}_{i}", re.compile(f"(?i){pattern}")))
                        except re.error:
                            continue
            else:
                # If we have too many patterns, batch them
                for i in range(0, len(patterns), 50):
                    batch = patterns[i:i+50]
                    try:
                        combined_pattern = re.compile(f"(?i){'|'.join(batch)}")
                        self.learned_patterns.append((f"{category}_batch_{i//50}", combined_pattern))
                    except re.error as e:
                        print(f"Warning: Couldn't compile batch {i//50} for {category}: {e}")
                        # If batch fails, add individual patterns
                        for j, pattern in enumerate(batch):
                            try:
                                self.learned_patterns.append(
                                    (f"{category}_{i+j}", re.compile(f"(?i){pattern}"))
                                )
                            except re.error:
                                continue
    
    @lru_cache(maxsize=10000)
    def _split_into_sentences(self, text: str) -> Tuple[str, ...]:
        """
        Split text into sentences, considering legal formatting.
        Cached for performance with identical texts.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of sentences (tuple for hashability in cache)
        """
        # Replace newlines with spaces to handle split sentences
        text = re.sub(r'\n+', ' ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences
        sentences = tuple(s.strip() for s in sentences if s.strip())
        
        return sentences
    
    @lru_cache(maxsize=10000)
    def _score_sentence_optimized(self, sentence: str) -> float:
        """
        Optimized sentence scoring with tiered pattern matching.
        
        Args:
            sentence: Sentence to score
            
        Returns:
            Score (higher = more likely a determination)
        """
        # Convert to lowercase once
        lower_sentence = sentence.lower()
        
        # Tiered scoring based on pattern categories
        for category, (base_score, _) in self.pattern_categories.items():
            # Skip learned patterns category - we check it separately
            if category == 'learned_patterns':
                continue
            
            # Check exact matches (most efficient)
            if category == 'exact_match':
                for phrase in self.exact_match_lowercase:
                    if phrase in lower_sentence:
                        return base_score  # Early return with highest confidence
            
            # Check regex patterns
            elif category in self.compiled_patterns:
                # Handle both single pattern and list of patterns
                patterns = self.compiled_patterns[category]
                if isinstance(patterns, list):
                    # Multiple individual patterns
                    for pattern in patterns:
                        if pattern.search(sentence):
                            return base_score  # Early return on match
                else:
                    # Single combined pattern
                    if patterns.search(sentence):
                        return base_score  # Early return on match
        
        # If no strong matches, check for supporting evidence
        score = 0.0
        
        # Check learned patterns (more expensive)
        for category, pattern in self.learned_patterns:
            if pattern.search(sentence):
                score += 5.0
                break  # Only add once
        
        # Check for keywords
        for keyword in self.determination_keywords:
            if f" {keyword} " in f" {lower_sentence} ":
                score += 1.0
        
        # Consider sentence length
        length = len(lower_sentence.split())
        if 5 <= length <= 30:
            score += 0.5
        elif length > 30:
            score -= 0.5
        
        # Check for outcome terms
        outcome_terms = ["allowed", "dismissed", "granted", "refused", "rejected", "set aside", "affirmed", "abandoned"]
        for term in outcome_terms:
            if f" {term} " in f" {lower_sentence} ":
                score += 1.0
                break
        
        return score
    
    def extract_potential_determinations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential determination sentences from the text.
        
        Args:
            text: Document text
            
        Returns:
            List of potential determination sentences with metadata
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        potential_determinations = {}  # Use dict for deduplication with sentence as key
        
        # Split into sentences 
        sentences = self._split_into_sentences(text)
        
        # First pass: direct matching with exact sentences (very efficient)
        for i, sentence in enumerate(sentences):
            if sentence in self.exact_determinations:
                potential_determinations[sentence] = {
                    'text': sentence,
                    'score': 20.0,  # Highest score for exact matches
                    'position': i / max(1, len(sentences) - 1),
                    'length': len(sentence),
                    'match_type': 'exact'
                }
        
        # Second pass: optimized pattern matching for remaining sentences
        for i, sentence in enumerate(sentences):
            # Skip if already matched exactly
            if sentence in potential_determinations:
                continue
                
            score = self._score_sentence_optimized(sentence)
            
            if score > 0:
                potential_determinations[sentence] = {
                    'text': sentence,
                    'score': score,
                    'position': i / max(1, len(sentences) - 1),
                    'length': len(sentence),
                    'match_type': 'pattern'
                }
        
        # Convert dict to list
        result = list(potential_determinations.values())
        
        # Sort by score (highest first)
        result.sort(key=lambda x: x['score'], reverse=True)
        
        return result
    
    def process_case(self, text: str) -> Dict[str, Any]:
        """Process a case to extract determinations."""
        # Make sure data is loaded
        self.ensure_data_loaded()
        
        start_time = time.time()
        
        # Early return for empty or invalid text
        if not isinstance(text, str) or not text.strip():
            self.processed_cases += 1
            self.processing_time += time.time() - start_time
            return {'extracted_determinations': []}
        
        # Extract potential determinations
        determinations = self.extract_potential_determinations(text)
        
        # Filter and select final determinations
        final_determinations = self._select_determinations(determinations)
        
        # Track performance
        self.processed_cases += 1
        self.processing_time += time.time() - start_time
        
        return {'extracted_determinations': final_determinations}

    def _select_determinations(self, potential_determinations):
        """Select final determinations from candidates using adaptive thresholding."""
        if not potential_determinations:
            return []
        
        # Sort by score (highest first)
        sorted_determinations = sorted(potential_determinations, key=lambda x: x['score'], reverse=True)
        
        # Adaptive thresholding based on highest score
        if sorted_determinations[0]['score'] >= 10:
            threshold = max(5, sorted_determinations[0]['score'] * 0.7)
            selected = [d for d in sorted_determinations if d['score'] >= threshold]
            
            # Cap at reasonable number
            return selected[:5] if len(selected) > 5 else selected
        else:
            # Low confidence - just take top 3
            return sorted_determinations[:3]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model."""
        if self.processed_cases == 0:
            return {'processed_cases': 0, 'avg_time_per_case': 0}
            
        return {
            'processed_cases': self.processed_cases,
            'total_processing_time': f"{self.processing_time:.2f} seconds",
            'avg_time_per_case': f"{(self.processing_time / self.processed_cases) * 1000:.2f} ms"
        }

    def _extract_templates(self, examples):
        """Extract generalized templates from examples."""
        templates = set()
        
        for example in examples:
            # Always add the exact example
            if example:
                templates.add(example.lower())
            
            # Skip short examples for template generation
            if len(example.split()) < 3:
                continue
            
            text = example.lower()
            
            # Generate templates based on common patterns
            if "appeal of " in text or "appeal for " in text:
                template = self._create_name_template(text)
                if template:
                    templates.add(template)
            
            if "file " in text:
                template = self._create_file_template(text)
                if template:
                    templates.add(template)
        
        return templates

    def _create_name_template(self, text):
        """Create template for text containing names."""
        # Handle "appeal of NAME" pattern
        for prefix in ["appeal of ", "appeal for "]:
            if prefix in text and " is " in text:
                parts = text.split(prefix, 1)
                if len(parts) != 2:
                    continue
                    
                before, after = parts
                if " is " not in after:
                    continue
                    
                name_part, remainder = after.split(" is ", 1)
                # Only replace if name looks reasonable
                if 1 <= len(name_part.split()) <= 3:
                    return f"{before}{prefix}[NAME] is {remainder}"
        
        return None

    def _create_file_template(self, text):
        """Create template for text containing file numbers."""
        if "file " not in text:
            return None
            
        before, after = text.split("file ", 1)
        words = after.split()
        
        if not words:
            return None
            
        # Check if first word looks like a file number
        if re.match(r'^[a-z0-9\-]+$', words[0]):
            rest = " ".join(words[1:])
            return f"{before}file [FILE] {rest}"
        
        return None
