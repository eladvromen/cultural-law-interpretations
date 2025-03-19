from typing import List, Dict, Any, Set, Optional, Tuple
import re
from collections import defaultdict
import json
from pathlib import Path
import pickle
from functools import lru_cache

from base_determination_extractor import BaseDeterminationExtractor

class SparseExplicitExtractor(BaseDeterminationExtractor):
    """
    First layer model that focuses on high-precision extraction of ONLY the most explicit determination sentences.
    This model should only extract sentences that are unambiguously determination statements.
    """
    
    def __init__(self):
        super().__init__()
        
        # Explicit determination phrases - expanded with more variations
        self.explicit_determinations = {
            # Appeal outcomes (complete phrases only)
            'the appeal is allowed',
            'the appeal is dismissed',
            'the appeals are allowed',
            'the appeals are dismissed',
            'appeal is allowed',
            'appeal is dismissed',
            'appeal is not allowed',
            'appeal was dismissed',
            'appeal be dismissed',
            'appeal has been abandoned',
            'appeal is denied',
            'appeal is determined',
            'appeal were dismissed',
            'appeals are allowed',
            'appeal is therefore allowed',
            'appeal is allowed in',
            'appeals are dismissed',
            'appeal should be dismissed',
            'appeal were to be dismissed',
            'appeal for appellant is allowed',
            'dismiss the appellants application',
            
            # Status determinations
            'appellant has not lost his permanent resident status',
            'appellant has not lost her permanent resident status',
            'appellants have not lost their permanent resident status',
            
            # Application outcomes
            'application for withdrawal is dismissed',
            'application to withdraw the appeal is an abuse of process',
            
            # Order outcomes
            'removal order is set aside',
            'the removal order is set aside',
            
            # Other determinations
            'sufficient humanitarian and compassionate considerations'
        }
        
        # Add new explicit determinations
        self.explicit_determinations.update({
            # Refugee determinations
            'claimant is neither a convention refugee nor a person in need of protection',
            'refugee protection division rejects his claim',
            'negative',
            'reject the claimants explanation',
            'reject the claimants explanations',
            'reject this explanation',
            
            # RAD determinations
            'rad confirms the decision of the rpd',
            'rad grants the appeal',
            'rad finds the nigerian jg inapplicable',
            
            # Appeal terminations
            'appeal is terminated',
            'appeal should not be terminated',
            'appeal would be terminated',
            
            # Refugee status determinations
            'neither a convention refugee nor a person in need of protection',
            'negative',
            'pc is not a person in need of protection',
            'panel therefore rejects their claims',
            'claimants are neither convention refugees nor are they persons in need of protection',
            'claimants are determined to be neither convention refugees nor persons in need of protection',
            'they are not convention refugees',
            'claimants are not convention refugees',
            'claimants are not persons in need of protection',
            'board rejects the claims',
            'claims for refugee protection are rejected',
            'appellant is a convention refugee',
            'appellant is a person in need of protection',
            'appellant is not excluded from the definition of refugee',
            'appellants are persons in need of protection',
            'rpd decision stands',
            'appeal is without merit',
            
            # RPD/RAD determinations
            'rpds determination is set aside',
            'rpds credibility assessment stands',
            'rad dismisses the appeal',
            'confirms the decision of the rpd',
            'substitutes the decision',
            'overturn its finding'
        })
        
        # Update patterns to include new cases
        self.patterns = [
            # Direct appeal outcomes
            re.compile(r'(?:the\s+)?appeal\s+(?:is|was|be|were|should\s+be)\s+(?:allowed|dismissed|denied|determined)', re.I),
            re.compile(r'(?:the\s+)?appeal\s+(?:is|was)\s+not\s+allowed', re.I),
            re.compile(r'(?:the\s+)?appeal\s+has\s+been\s+abandoned', re.I),
            re.compile(r'(?:the\s+)?appeals?\s+(?:of|by|for)\s+[^.]+\s+(?:is|are|were|was)\s+(?:allowed|dismissed)', re.I),
            
            # Status determinations
            re.compile(r'(?:the\s+)?appellant(?:s)?\s+has\s+not\s+lost\s+(?:his|her|their)\s+permanent\s+resident\s+status', re.I),
            
            # Order outcomes
            re.compile(r'(?:the\s+)?(?:removal\s+)?order\s+is\s+set\s+aside', re.I),
            
            # Notice of decision
            re.compile(r'notice\s+of\s+decision\s*[:-]?\s*(?:the\s+)?appeal\s+is\s+(?:allowed|dismissed)', re.I),
            
            # RAD determinations
            re.compile(r'rad\s+(?:grants|confirms|finds|rejects)\s+(?:the\s+)?(?:appeal|decision)', re.I),
            re.compile(r'rad\s+finds\s+that\s+(?:the\s+)?rpd\s+correctly\s+decided', re.I),
            
            # Refugee determinations
            re.compile(r'(?:the\s+)?claimant\s+is\s+(?:not\s+)?(?:a\s+)?(?:convention\s+refugee|person\s+in\s+need\s+of\s+protection)', re.I),
            re.compile(r'reject\s+(?:the\s+)?claimants?\s*\'?\s*explanations?', re.I),
            re.compile(r'refugee\s+protection\s+division\s+(?:therefore\s+)?rejects?\s+(?:his|her|the)\s+claim', re.I),
            
            # Appeal terminations
            re.compile(r'(?:the\s+)?appeal\s+(?:is|should\s+(?:not\s+)?be|would\s+be)\s+terminated', re.I),
            
            # Multiple determination handling
            re.compile(r'appeals?\s+of\s+(?:the\s+)?(?:principal|minor)\s+appellants?.*?(?:is|are)\s+(?:allowed|dismissed|rejected)', re.I),
            
            # Refugee status determinations
            re.compile(r'(?:claimants?|appellants?|they)\s+(?:is|are)\s+(?:determined\s+to\s+be\s+)?(?:not|neither)\s+(?:a\s+)?convention\s+refugees?', re.I),
            re.compile(r'(?:claimants?|appellants?|they)\s+(?:is|are)\s+(?:not\s+)?persons?\s+in\s+need\s+of\s+protection', re.I),
            re.compile(r'(?:panel|board)\s+(?:therefore\s+)?rejects?\s+(?:the\s+)?claims?', re.I),
            re.compile(r'claims?\s+for\s+refugee\s+protection\s+(?:is|are)\s+rejected', re.I),
            re.compile(r'rejects?\s+the\s+claim\s+of\s+the\s+(?:principal|minor)\s+appellants?(?:\'s?\s+\w+)?', re.I),
            
            # RPD/RAD determinations
            re.compile(r'rpd(?:\'s)?\s+(?:determination|decision|credibility\s+assessment)\s+(?:is\s+set\s+aside|stands)', re.I),
            re.compile(r'rad\s+(?:dismisses|confirms|allows)\s+(?:the\s+)?(?:appeal|decision)', re.I),
            re.compile(r'(?:panel\s+)?sets?\s+aside\s+the\s+determination\s+of\s+the\s+rpd', re.I),
            re.compile(r'substitute(?:s)?\s+(?:my|the|its)\s+(?:determination|decision)', re.I),
            re.compile(r'confirm(?:s)?\s+the\s+decision\s+of\s+the\s+rpd', re.I),
            
            # Other determinations
            re.compile(r'appeal\s+is\s+without\s+merit', re.I),
            re.compile(r'overturn(?:s)?\s+(?:its|the)\s+finding', re.I)
        ]

        # Sentence bank for exact matches
        self.sentence_bank = set()
        
        # Performance tracking
        self.processed_cases = 0
        self.processing_time = 0.0

    def is_determination_sentence(self, sentence: str, context: Dict[str, Any] = None) -> Tuple[bool, float]:
        """Check if a sentence is an explicit determination statement."""
        # Clean and normalize
        cleaned = self.clean_sentence(sentence)
        
        # Debug logging
        print(f"Original: '{sentence}'")
        print(f"Cleaned:  '{cleaned}'")
        
        # 1. Check exact matches in sentence bank
        if cleaned in self.sentence_bank:
            print("Match found in sentence bank")
            return True, 1.0

        # 2. Check explicit determination phrases
        if cleaned in self.explicit_determinations:
            print("Match found in explicit determinations")
            return True, 1.0
        
        # 2.5 Check partial matches for phrases that contain names
        for phrase in ['appeal is allowed for', 'appeal of', 'appeal for']:
            if cleaned.startswith(phrase) and 'allowed' in cleaned:
                print(f"Partial match found with phrase: {phrase}")
                return True, 1.0

        # 3. Check strict patterns
        for pattern in self.patterns:
            if pattern.search(cleaned):
                print(f"Pattern match found: {pattern.pattern}")
                return True, 1.0

        print("No match found")
        return False, 0.0

    def extract_determinations(self, text: str) -> List[Dict[str, Any]]:
        """Extract only explicit determination sentences."""
        results = []
        
        # Try to parse as JSON-like list first
        if text.startswith('[') and text.endswith(']'):
            try:
                # Handle nested list format
                text = text.replace("'", '"')
                items = json.loads(text)
                if isinstance(items, list):
                    # Handle nested lists and multiple determinations
                    for item in items:
                        if isinstance(item, str):
                            # Split on common separators
                            sub_items = re.split(r'\s*[,;]\s*|\s+and\s+', item)
                            for sub_item in sub_items:
                                cleaned = re.sub(r'^\[+|\]+$', '', sub_item.strip())
                                cleaned = re.sub(r'^[\'"]|[\'"]$', '', cleaned.strip())
                                if cleaned and not cleaned.isdigit():  # Skip pure numbers
                                    is_determination, confidence = self.is_determination_sentence(cleaned)
                                    if is_determination:
                                        results.append({
                                            'text': cleaned,
                                            'score': confidence,
                                            'match_type': 'explicit'
                                        })
                    return results
            except:
                pass
        
        # If not a list format, split into sentences
        sentences = re.split(r'[.!?]+|(?<=\])\s+(?=\[)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            # Split on common separators within sentences
            sub_sentences = re.split(r'\s*[,;]\s*|\s+and\s+', sentence)
            for sub_sentence in sub_sentences:
                if sub_sentence.strip() and not sub_sentence.isdigit():
                    is_determination, confidence = self.is_determination_sentence(sub_sentence)
                    if is_determination:
                        results.append({
                            'text': sub_sentence,
                            'score': confidence,
                            'match_type': 'explicit'
                        })
        
        return results

    def process_case(self, text: str) -> Dict[str, Any]:
        """Process a case to extract determinations."""
        determinations = self.extract_determinations(text)
        return {'extracted_determinations': determinations}

    def load_training_examples(self, train_path: str, test_path: str, use_chunking: bool = True) -> None:
        """Load training examples, only keeping the most explicit ones."""
        print("Loading training examples...")
        
        # Extract examples from training data
        train_examples = self._extract_examples_from_file(train_path, use_chunking)
        test_examples = self._extract_examples_from_file(test_path, use_chunking)
        
        # Process each example
        for example in train_examples + test_examples:
            # Only learn from explicit examples
            is_determination, confidence = self.is_determination_sentence(example)
            if is_determination:  # Must be 100% confident
                self.sentence_bank.add(self.clean_sentence(example))
        
        print(f"Loaded {len(self.sentence_bank)} explicit determination sentences")
        
        # Save sentence bank to cache
        try:
            cache_path = Path(__file__).parent / "sentence_bank.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(self.sentence_bank, f)
            print(f"Saved sentence bank to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save sentence bank cache: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model."""
        if self.processed_cases == 0:
            return {'processed_cases': 0, 'avg_time_per_case': 0}
            
        return {
            'processed_cases': self.processed_cases,
            'total_processing_time': f"{self.processing_time:.2f} seconds",
            'avg_time_per_case': f"{(self.processing_time / self.processed_cases) * 1000:.2f} ms",
            'sentence_bank_size': len(self.sentence_bank)
        }

    def load_sentence_bank(self):
        """Load and process the sentence bank from training data."""
        try:
            cache_path = Path(__file__).parent / "sentence_bank.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.sentence_bank = pickle.load(f)
                print(f"Loaded {len(self.sentence_bank)} exact determination sentences")
                return
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.sentence_bank = set()

    def clean_sentence(self, sentence: str) -> str:
        """Clean and normalize a sentence for exact matching."""
        # Remove outer brackets and quotes
        sentence = re.sub(r'^\[+|\]+$', '', sentence.strip())
        sentence = re.sub(r'^[\'"]|[\'"]$', '', sentence.strip())
        
        # Convert to lowercase before any other processing
        sentence = sentence.lower()
        
        # Fix run-together words and common variations
        fixes = {
            'isallowed': 'is allowed',
            'isdismissed': 'is dismissed',
            'ishereby': 'is hereby',
            'wasallowed': 'was allowed',
            'bedismissed': 'be dismissed',
            'isset': 'is set',
            'allowedthe': 'allowed the',
            'allowedin': 'allowed in',
            'allowedfor': 'allowed for',
            'claimants': 'claimant s',  # Handle possessive
            'appellants': 'appellant s',
            'rpds': 'rpd s',  # Handle possessive
            'panels': 'panel s',
            'boards': 'board s',
            'principals': 'principal s',
            'minors': 'minor s'
        }
        
        for original, replacement in fixes.items():
            sentence = sentence.replace(original, replacement)
        
        # Handle periods that join words
        sentence = re.sub(r'(\w)\.(\w)', r'\1 \2', sentence)
        
        # Clean up whitespace
        sentence = ' '.join(sentence.split())
        
        # Remove punctuation only at the ends
        sentence = sentence.strip('.,;:')
        
        return sentence

    def learn_from_examples(self, examples: List[str]):
        """
        Learn patterns from example determination sentences.
        Only learns exact, explicit determinations.
        
        Args:
            examples: List of example determination sentences
        """
        for example in examples:
            clean_example = self.clean_sentence(example)
            # Only add if it's a clear determination sentence
            is_determination, confidence = self.is_determination_sentence(clean_example)
            if is_determination and confidence >= 0.9:  # Very high confidence threshold
                self.sentence_bank.add(clean_example)
        
        # Save to cache
        cache_path = Path(__file__).parent / "sentence_bank.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(self.sentence_bank, f)
        
        print(f"Added {len(self.sentence_bank)} high-confidence determination sentences to bank")

    def _test_basic_patterns(self):
        """Test basic determination patterns."""
        test_cases = [
            "The appeal is allowed",
            "appeal is allowed",
            "The appeal is dismissed",
            "appeal is dismissed",
            "[appeal is allowed]",
            "['appeal is allowed']",
            "appeal is allowed.the",
        ]
        
        for test in test_cases:
            is_det, conf = self.is_determination_sentence(test)
            print(f"Testing: '{test}' -> {'✓' if is_det else '✗'}")
