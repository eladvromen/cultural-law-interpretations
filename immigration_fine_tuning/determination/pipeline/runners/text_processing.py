import pandas as pd
import re
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional, Set, Pattern
import time
from tqdm import tqdm
from dataclasses import dataclass
from functools import lru_cache

# Module level constants
PAGE_POINTER_PATTERNS = [
            re.compile(r'(?:\d+\s+)?RPD File No\.\s*/\s*N°\s*de dossier de la SPR\s*:\s*[A-Z0-9\-\s]+'),
            re.compile(r'\d+\s+(?:IRB|CISR|RPD|SPR|RAD|SAR)\s+\d+'),
            re.compile(r'(?:Page|p\.)\s*\d+\s*(?:of|de)\s*\d+'),
            re.compile(r'^\s*\d+\s*$')  # Standalone page numbers
        ]
        
METADATA_PATTERNS = [
            re.compile(r'REFUGEE PROTECTION DIVISION\s*/\s*.*\s*/\s*.*\s*/\s*.*'),
            re.compile(r'(?:NEGATIVE|POSITIVE)\s*/\s*(?:[A-Z]+\s*/\s*)+[A-Z]+$'),
            re.compile(r'(?:MALE|FEMALE)\s*/\s*(?:NEGATIVE|POSITIVE)\s*/\s*[A-Z]+$')
        ]
        
FRENCH_PATTERNS = [
            re.compile(r'(?i)vous\s+procurer\s+les\s+présents\s+motifs'),
            re.compile(r'(?i)La\s+Direction\s+des\s+services'),
            re.compile(r'(?i)Vous\s+n\'avez\s+qu\'à\s+en\s+faire'),
            re.compile(r'(?i)étage,\s+Ottawa'),
            re.compile(r'(?i)demandeur\s+d\'asile'),
            re.compile(r'(?i)Date\s+d\'audience'),
            re.compile(r'(?i)Lieu\s+de\s+l\'audience')
        ]
        
FRENCH_INDICATORS = ['de', 'du', 'la', 'les', 'des', 'pour', 'avec', 'dans', 'par']

# Define header categories and their key terms
HEADER_CATEGORIES = {
            # Decision/Notice headers (final outcomes)
            'decision_headers': [
                'NOTICE OF DECISION', 'DECISION AND REASONS', 'DECISION AND ORDER',
                'NOTICE OF DECISION AND REASONS', 'DECISION', 'DISPOSITION',
                'NOTICE', 'ORDER', 'THE DECISION'
            ],
            
            # Determination headers (explicit determination statements)
            'determination_headers': [
                'DETERMINATION', 'DETERMINATIVE ISSUE', 'DETERMINATION OF THE APPEAL',
                'RAD DETERMINATION', 'RPD DETERMINATION', 'FINDINGS',
                'DETERMINATION OF THE APPLICATION', 'THE DETERMINATION'
            ],
            
            # Analysis headers (focused on analysis)
            'analysis_headers': [
                'ANALYSIS', 'ASSESSMENT', 'ANALYSIS AND FINDINGS',
                'ANALYSIS AND FINDINGS OF FACT', 'STANDARD OF REVIEW'
            ],
            
            # Reasons for decision headers (separate from analysis)
            'reasons_headers': [
                'REASONS FOR DECISION', 'REASONS AND DECISION', 'REASONS', 
                'ORAL REASONS FOR DECISION', 'ORAL REASONS', 'ORAL DECISION',
                'ORAL DECISION AND REASONS', 'REASONS FOR ORAL DECISION'
            ],
            
            # Conclusion headers
            'conclusion_headers': [
                'CONCLUSION', 'CONCLUSIONS', 'SUMMARY AND DETERMINATION', 'THE CONCLUSION'
            ]
        }
        
        # Headers important for determination extraction
DETERMINATION_RELATED_HEADERS = {
            'decision_headers',
            'determination_headers',
            'analysis_headers',
            'reasons_headers',
            'conclusion_headers'
        }
        
        # Outcome patterns for decision extraction
OUTCOME_PATTERNS = {
            'allowed': [
                re.compile(r'(?i)appeal\s+is\s+allowed'),
                re.compile(r'(?i)application\s+is\s+allowed'),
                re.compile(r'(?i)claim\s+is\s+allowed')
            ],
            'dismissed': [
                re.compile(r'(?i)appeal\s+is\s+dismissed'),
                re.compile(r'(?i)application\s+is\s+dismissed'),
                re.compile(r'(?i)claim\s+is\s+dismissed')
            ],
            'set_aside': [
                re.compile(r'(?i)order\s+is\s+set\s+aside'),
                re.compile(r'(?i)decision\s+is\s+set\s+aside'),
                re.compile(r'(?i)set\s+aside\s+the\s+decision')
            ],
            'granted': [
                re.compile(r'(?i)refugee\s+protection\s+is\s+granted'),
                re.compile(r'(?i)application\s+is\s+granted')
            ],
            'refused': [
                re.compile(r'(?i)refugee\s+protection\s+is\s+refused'),
                re.compile(r'(?i)application\s+is\s+refused'),
                re.compile(r'(?i)claim\s+is\s+rejected')
            ]
        }
        
OUTCOME_KEYWORDS = {
    'allowed': ['allow', 'grant', 'accept', 'positive'],
    'dismissed': ['dismiss', 'reject', 'refuse', 'negative', 'denied'],
    'set_aside': ['set aside', 'vacate', 'quash'],
}

# Pattern for numbered headers
NUMBERED_HEADER_PATTERN = re.compile(r'^\s*\d+\s+([A-Z][A-Za-z\s]+)\s*$')


@dataclass
class DecisionOutcome:
    allowed: bool = False
    dismissed: bool = False
    set_aside: bool = False
    granted: bool = False
    refused: bool = False
    confidence: float = 0.0
    outcome_text: Optional[str] = None


class TextCleaner:
    """
    Handles text cleaning operations like removing page numbers and French content.
    """
    
    @staticmethod
    def remove_page_pointers(text: str) -> str:
        """Remove page pointers and file numbers using multiple patterns."""
        for pattern in PAGE_POINTER_PATTERNS:
            text = pattern.sub('', text)
        return text
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def is_french_line(line: str) -> bool:
        """Check if a line is likely in French using pattern matching."""
        if not line.strip():
            return False
            
        for pattern in FRENCH_PATTERNS:
            if pattern.search(line):
                return True
        
        # Additional heuristics for French detection
        words = re.findall(r'\b\w+\b', line.lower())
        
        # If more than 30% of words are French indicators, likely French
        if words and len(words) > 3:
            french_count = sum(1 for word in words if word in FRENCH_INDICATORS)
            if french_count / len(words) > 0.3:
                return True
        
        return False
    
    @classmethod
    def remove_french_content(cls, text: str) -> str:
        """Remove French content while preserving document structure."""
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            if not cls.is_french_line(line):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace while preserving paragraph structure."""
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize newlines (replace multiple newlines with double newline)
        text = re.sub(r'\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def clean_redacted_info(text: str) -> str:
        """Standardize redacted information (XXXXX)."""
        # Replace varying lengths of X with [REDACTED]
        text = re.sub(r'X{3,}', '[REDACTED]', text)
        
        # Also handle other redaction patterns
        text = re.sub(r'\[REDACTED\]', '[REDACTED]', text)
        text = re.sub(r'\[REDACTION\]', '[REDACTED]', text)
        
        return text
    
    @staticmethod
    def extract_metadata(text: str) -> Tuple[str, Optional[str]]:
        """Extract metadata from the document using multiple patterns."""
        metadata = None
        
        # Try each metadata pattern
        for pattern in METADATA_PATTERNS:
            match = pattern.search(text)
            if match:
                metadata = match.group(0).strip()
                # Remove the metadata line from the text
                text = text[:match.start()] + text[match.end():]
                break
        
        return text.strip(), metadata


class HeaderDetector:
    """
    Handles detection and categorization of document headers.
    """
    
    def __init__(self):
        # Generate flexible regex patterns for each header term
        self.header_patterns = self._generate_header_patterns()
        
        # Create a quick lookup set of all header terms for fast checking
        self.header_terms_set = set()
        for category_terms in HEADER_CATEGORIES.values():
            for term in category_terms:
                self.header_terms_set.add(term.upper())
                # Also add without spaces for faster checking
                self.header_terms_set.add(term.upper().replace(' ', ''))
    
    def _generate_header_patterns(self) -> Dict[str, Dict[str, List[Pattern]]]:
        """
        Generate flexible regex patterns for each header category and term.
        
        Returns:
            Dictionary of patterns by category and term
        """
        patterns = {}
        
        for category, terms in HEADER_CATEGORIES.items():
            category_patterns = {}
            
            for term in terms:
                term_patterns = []
                
                # Create base pattern with flexible spacing
                base_pattern = r'^\s*'
                
                # Add each character with optional spaces
                words = term.split()
                for i, word in enumerate(words):
                    if i > 0:
                        base_pattern += r'\s+'  # Space between words
                    
                    # Add each character with optional spaces
                    for j, char in enumerate(word):
                        if j > 0:
                            base_pattern += r'\s*'  # Optional space between chars
                        base_pattern += re.escape(char)
                
                # Create variations - compile patterns only once for better performance
                term_patterns.append(re.compile(base_pattern + r'\s*$', re.IGNORECASE))  # Exact match
                term_patterns.append(re.compile(base_pattern + r'\s+.*$', re.IGNORECASE))  # With suffix
                
                # For multi-word terms, also match with prefix
                if len(words) > 1:
                    term_patterns.append(re.compile(r'^\s*.*\s+' + base_pattern + r'\s*$', re.IGNORECASE))
                
                # Add pattern for numbered headers (e.g., "2 Analysis", "8 CONCLUSION")
                term_patterns.append(re.compile(r'^\s*\d+\s+' + base_pattern + r'\s*$', re.IGNORECASE))
                
                # Add pattern for headers that appear at the end of sentences (e.g., "She also fears imprisonment if she returns to Tunisia. DECISION")
                term_patterns.append(re.compile(r'.*[\.!\?]\s+' + base_pattern + r'\s*$', re.IGNORECASE))
                
                # Add pattern for headers in square brackets with numbers (e.g., "DETERMINATION\\n\\n[6]")
                term_patterns.append(re.compile(base_pattern + r'\s*\n\s*\[\d+\]', re.IGNORECASE))
                
                # Add pattern for headers followed by text in square brackets (e.g., "CONCLUSION [18]")
                term_patterns.append(re.compile(base_pattern + r'\s*\[\d+\]', re.IGNORECASE))
                
                category_patterns[term] = term_patterns
            
            patterns[category] = category_patterns
        
        return patterns
    
    @staticmethod
    def normalize_header_text(text: str) -> str:
        """
        Normalize header text by removing extra spaces and standardizing format.
        
        Args:
            text: Header text to normalize
            
        Returns:
            Normalized header text
        """
        # Remove quotes and other special characters
        text = text.replace('"', '').replace("'", "")
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove trailing punctuation
        text = re.sub(r'[.:]$', '', text)
        
        return text
    
    @staticmethod
    @lru_cache(maxsize=1024)  # Add caching for significant performance improvement
    def preprocess_line(line: str) -> str:
        """
        Preprocess a line to handle special characters and formatting.
        
        Args:
            line: Line to preprocess
            
        Returns:
            Preprocessed line
        """
        # Remove quotes and other special characters
        line = line.replace('"', '').replace("'", "")
        
        # Remove excessive newlines
        line = re.sub(r'\n+', ' ', line)
        
        # Check for numbered headers (e.g., "2 Analysis", "8 CONCLUSION")
        match = NUMBERED_HEADER_PATTERN.match(line)
        if match:
            # Extract the header text without the number
            return match.group(1).strip()
        
        # Quick check to see if we need to do expensive operations
        line_upper = line.upper()
        key_terms = ["DECISION", "DETERMINATION", "CONCLUSION", "ANALYSIS", "REASONS"]
        if not any(term in line_upper for term in key_terms):
            return line.strip()
            
        # Check for headers at the end of sentences (e.g., "... returns to Tunisia. DECISION")
        for category in HEADER_CATEGORIES.values():
            for term in category:
                # Only check if term might be present in the line (performance optimization)
                if term in line_upper:
                    pattern = re.compile(r'(.*?[\.!\?]\s+)(' + re.escape(term) + r')(\s*$|\s*\[\d+\])', re.IGNORECASE)
                    match = pattern.match(line)
                    if match:
                        return match.group(2).strip()  # Return just the header term
        
        return line.strip()
    
    @lru_cache(maxsize=1024)  # Add caching for significant performance improvement
    def match_header(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Match a line of text to a header category and term.
        
        Args:
            text: Text to match
            
        Returns:
            Tuple of (category, term) if matched, else (None, None)
        """
        # Quick check if this might be a header line
        text_upper = text.upper()
        text_upper_no_spaces = text_upper.replace(' ', '')
        
        # Fast initial check - if none of the common header keywords are present, skip expensive checks
        key_terms = ["DECISION", "DETERMINATION", "CONCLUSION", "ANALYSIS", "REASONS"]
        if not any(term in text_upper for term in key_terms):
            return None, None
        
        # Special case for lines ending with known headers (very common pattern)
        for category, terms in HEADER_CATEGORIES.items():
            for term in terms:
                term_upper = term.upper()
                # Check for exact match at the end of the line
                if text_upper.endswith(term_upper):
                    return category, term
                # Check for header with brackets
                if term_upper in text_upper and re.search(rf'{re.escape(term_upper)}\s*\[\d+\]', text_upper):
                    return category, term
        
        # If the quick check passed but special cases didn't match, do full preprocessing
        preprocessed_text = self.preprocess_line(text)
        normalized_text = self.normalize_header_text(preprocessed_text)
        
        # Now go through all patterns, but in a more efficient order
        # Process single-word headers first (faster to check)
        for category, term_patterns in self.header_patterns.items():
            for term, patterns in term_patterns.items():
                # Skip patterns if the term is definitely not in the text
                # This avoids unnecessary regex operations
                term_upper = term.upper()
                term_upper_no_spaces = term_upper.replace(' ', '')
                
                if term_upper in text_upper or term_upper_no_spaces in text_upper_no_spaces:
                    for pattern in patterns:
                        if pattern.search(normalized_text):
                            return category, term
        
        return None, None
    
    def identify_section_headers(self, text: str) -> Dict[str, Any]:
        """
        Identify section headers in the text and categorize them by semantic group.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with identified headers by category
        """
        lines = text.split('\n')
        
        # Initialize result structure
        headers = {
            'by_category': {},
            'by_line': [],
            'determination_related': set()
        }
        
        # Initialize categories
        for category in HEADER_CATEGORIES:
            headers['by_category'][category] = []
        
        # Process each line to identify headers
        i = 0
        while i < len(lines):
            line_text = lines[i].strip()
            if not line_text:
                i += 1
                continue
            
            # Quick check for potential header content
            line_upper = line_text.upper()
            if not any(term in line_upper for term in ["DECISION", "DETERMINATION", "CONCLUSION", "ANALYSIS", "REASONS"]):
                i += 1
                continue
            
            # Check for multi-line header patterns only if the line might be a header
            multi_line_header = False
            if i < len(lines) - 2 and any(header_term in line_upper for header_term in ["DETERMINATION", "CONCLUSION", "DECISION"]):
                next_line = lines[i+1].strip()
                
                # Only check multi-line patterns if the next line contains brackets or is very short
                if next_line.startswith('[') or len(next_line) < 20:
                    two_line_text = line_text + '\n' + next_line
                    
                    # Only check 3 lines if needed
                    three_line_text = None
                    if i < len(lines) - 3:
                        next_next_line = lines[i+2].strip()
                        if next_next_line.startswith('[') or len(next_next_line) < 20:
                            three_line_text = two_line_text + '\n' + next_next_line
                    
                    # Try matching with the combined text
                    for combined_text in [two_line_text, three_line_text] if three_line_text else [two_line_text]:
                        if combined_text is None:
                            continue
                            
                        category, term = self.match_header(combined_text)
                        if category:
                            header_info = {
                                'line_number': i,
                                'text': combined_text,
                                'normalized_text': self.normalize_header_text(combined_text),
                                'category': category,
                                'term': term
                            }
                            
                            headers['by_line'].append(header_info)
                            headers['by_category'][category].append(header_info)
                            
                            if category in DETERMINATION_RELATED_HEADERS:
                                headers['determination_related'].add(i)
                            
                            multi_line_header = True
                            i += 2 if combined_text == two_line_text else 3
                            break
                    
                    if multi_line_header:
                        continue
            
            # Single line matching
            category, term = self.match_header(line_text)
            if category:
                header_info = {
                    'line_number': i,
                    'text': line_text,
                    'normalized_text': self.normalize_header_text(line_text),
                    'category': category,
                    'term': term
                }
                
                headers['by_line'].append(header_info)
                headers['by_category'][category].append(header_info)
                
                if category in DETERMINATION_RELATED_HEADERS:
                    headers['determination_related'].add(i)
            
            i += 1
        
        # Sort headers by line number
        headers['by_line'].sort(key=lambda x: x['line_number'])
        
        return headers
    
    def extract_sections(self, text: str, headers: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Extract content between identified headers.
        
        Args:
            text: The document text
            headers: Dictionary of identified headers
            
        Returns:
            Dictionary with section contents by category
        """
        lines = text.split('\n')
        sections = {
            'by_header': {},
            'by_category': {}
        }
        
        # Initialize categories
        for category in HEADER_CATEGORIES:
            sections['by_category'][category] = {}
        
        # Get ordered list of headers
        ordered_headers = headers['by_line']
        
        # Extract content between headers
        for i, header_info in enumerate(ordered_headers):
            header_text = header_info['text']
            category = header_info['category']
            start_line = header_info['line_number'] + 1  # Start after header
            
            # Determine end line (next header or end of document)
            if i < len(ordered_headers) - 1:
                end_line = ordered_headers[i+1]['line_number']
            else:
                end_line = len(lines)
            
            # Extract section content
            if start_line < end_line:
                section_content = '\n'.join(lines[start_line:end_line]).strip()
                sections['by_header'][header_text] = section_content
                sections['by_category'][category][header_text] = section_content
        
        return sections
    

class OutcomeExtractor:
    """
    Extracts decision outcomes from text.
    """
    
    @staticmethod
    def extract_decision_outcomes(text: str) -> DecisionOutcome:
        """
        Extract decision outcomes using pattern matching.
        
        Args:
            text: Text to analyze (typically from decision sections)
            
        Returns:
            DecisionOutcome with extracted outcomes and confidence scores
        """
        outcome = DecisionOutcome()
        
        # Check for each outcome pattern
        for outcome_type, patterns in OUTCOME_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    setattr(outcome, outcome_type, True)
                    outcome.confidence = 0.9  # High confidence for direct matches
                    outcome.outcome_text = match.group(0)
                    return outcome
        
        # If no direct match, look for keywords
        for outcome_type, words in OUTCOME_KEYWORDS.items():
            for word in words:
                if re.search(r'(?i)\b' + re.escape(word) + r'\b', text):
                    setattr(outcome, outcome_type, True)
                    outcome.confidence = 0.6  # Lower confidence for keyword matches
                    
                    # Try to extract the sentence containing the keyword
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for sentence in sentences:
                        if re.search(r'(?i)\b' + re.escape(word) + r'\b', sentence):
                            outcome.outcome_text = sentence.strip()
                            break
                    
                    return outcome
        
        return outcome


class LegalTextPreprocessor:
    """
    An improved preprocessor for legal case texts with more robust pattern matching,
    better performance, and semantic grouping of section headers.
    """
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.header_detector = HeaderDetector()
        self.outcome_extractor = OutcomeExtractor()
    
    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Preprocess the legal case text with improved robustness and semantic grouping.
        
        Args:
            text: The raw case text
            
        Returns:
            Dictionary with cleaned text, extracted metadata, and categorized sections
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'cleaned_text': '',
                'metadata': None,
                'headers': {'by_line': [], 'by_category': {}},
                'sections': {'by_header': {}, 'by_category': {}},
                'decision_outcomes': None
            }
        
        # Store original text
        original_text = text
        
        # Extract metadata
        text, metadata = self.text_cleaner.extract_metadata(text)
        
        # Remove page pointers
        text = self.text_cleaner.remove_page_pointers(text)
        
        # Remove French content
        text = self.text_cleaner.remove_french_content(text)
        
        # Clean redacted information
        text = self.text_cleaner.clean_redacted_info(text)
        
        # Normalize whitespace
        text = self.text_cleaner.normalize_whitespace(text)
        
        # Identify section headers
        headers = self.header_detector.identify_section_headers(text)
        
        # Extract sections
        sections = self.header_detector.extract_sections(text, headers)
        
        # Extract decision outcomes
        decision_outcomes = None
        
        # Try to extract from decision headers first
        for header, content in sections['by_category'].get('decision_headers', {}).items():
            extracted_outcome = self.outcome_extractor.extract_decision_outcomes(content)
            if extracted_outcome.confidence > 0.5:
                decision_outcomes = extracted_outcome
                break
        
        # If not found or low confidence, try determination headers
        if not decision_outcomes or decision_outcomes.confidence < 0.7:
            for header, content in sections['by_category'].get('determination_headers', {}).items():
                extracted_outcome = self.outcome_extractor.extract_decision_outcomes(content)
                if extracted_outcome.confidence > 0.5:
                    if not decision_outcomes or extracted_outcome.confidence > decision_outcomes.confidence:
                        decision_outcomes = extracted_outcome
                    break
        
        # If still not found or low confidence, try conclusion headers
        if not decision_outcomes or decision_outcomes.confidence < 0.7:
            for header, content in sections['by_category'].get('conclusion_headers', {}).items():
                extracted_outcome = self.outcome_extractor.extract_decision_outcomes(content)
                if extracted_outcome.confidence > 0.5:
                    if not decision_outcomes or extracted_outcome.confidence > decision_outcomes.confidence:
                        decision_outcomes = extracted_outcome
                    break
        
        return {
            'original_text': original_text,
            'cleaned_text': text,
            'metadata': metadata,
            'headers': headers,
            'sections': sections,
            'decision_outcomes': decision_outcomes
        }


class DatasetProcessor:
    """
    Processes datasets of legal documents.
    """
    
    def __init__(self, text_column: str = 'full_text'):
        self.preprocessor = LegalTextPreprocessor()
        self.text_column = text_column
    
    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """
        Preprocess all texts in a dataframe with batching for better performance.

        Args:
            df: Input dataframe
            batch_size: Number of documents to process in each batch
            
        Returns:
            Dataframe with added preprocessing columns
        """
        # Create new columns for preprocessed data
        df['cleaned_text'] = None
        df['case_metadata'] = None
        df['general_metadata'] = None  # New column for CA IRB metadata

        # Create columns for key section categories
        essential_categories = [
            'decision_headers',
            'determination_headers',
            'analysis_headers',
            'reasons_headers',
            'conclusion_headers'
        ]

        # Add binary flag columns for each category
        for category in essential_categories:
            df[f'has_{category}'] = False
            df[f'{category}_text'] = None

        # Add columns for decision outcomes
        df['decision_outcome'] = None
        df['decision_text'] = None

        # Process in batches
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size

        start_time = time.time()

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_row = batch_idx * batch_size
            end_row = min(start_row + batch_size, total_rows)
            
            # Process each row in the batch
            for idx in range(start_row, end_row):
                row = df.iloc[idx]
                
                if pd.isna(row[self.text_column]) or not isinstance(row[self.text_column], str):
                    continue
                
                # Preprocess the text
                result = self.preprocessor.preprocess(row[self.text_column])
            
                # Update basic columns
                df.at[idx, 'cleaned_text'] = result['cleaned_text']
                df.at[idx, 'case_metadata'] = result['metadata']
                
                # Update section category columns and binary flags
                sections_by_category = result.get('sections', {}).get('by_category', {})
                for category in essential_categories:
                    category_sections = sections_by_category.get(category, {})
                    if category_sections:
                        df.at[idx, f'has_{category}'] = True
                        df.at[idx, f'{category}_text'] = json.dumps(category_sections)
                
                # Update decision outcome columns
                decision_outcomes = result.get('decision_outcomes')
                if decision_outcomes:
                    # Determine the primary outcome
                    primary_outcome = None
                    for outcome in ['allowed', 'dismissed', 'set_aside', 'granted', 'refused']:
                        if getattr(decision_outcomes, outcome, False):
                            primary_outcome = outcome
                            break
                    
                    if primary_outcome:
                        df.at[idx, 'decision_outcome'] = primary_outcome
                        df.at[idx, 'decision_text'] = decision_outcomes.outcome_text or ''
            
            # Print progress every few batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - start_time
                docs_per_sec = (batch_idx + 1) * batch_size / elapsed
                print(f"Processed {min((batch_idx + 1) * batch_size, total_rows)} documents in {elapsed:.2f}s ({docs_per_sec:.2f} docs/sec)")
        
        # After all regular processing is done, extract general metadata
        self._extract_general_metadata(df)
        
        self._print_statistics(df, essential_categories)
        
        return df
    
    def _extract_general_metadata(self, df: pd.DataFrame) -> None:
        """
        Extract general metadata and suspected first case paragraph by finding text before/after 
        first occurrence of "(CA IRB)". This runs after all other processing to avoid 
        interference with pattern matching.
        """
        # Initialize counters for statistics
        ca_irb_found_count = 0
        extracted_sentence_counts = []
        
        print("\nExtracting general metadata and suspected first case paragraph...")
        for idx in tqdm(range(len(df)), desc="Scanning for (CA IRB)"):
            row = df.iloc[idx]
            
            # Skip rows with missing text
            if pd.isna(row[self.text_column]) or not isinstance(row[self.text_column], str):
                continue
            
            # Get the original text for scanning
            original_text = row[self.text_column]
            
            # Split into sentences (simple heuristic split on periods followed by spaces)
            sentences = re.split(r'(?<=[.!?])\s+', original_text)
            
            # Limit to first 50 sentences for search
            first_sentences = sentences[:50]
            
            # Scan for "(CA IRB)"
            found_idx = -1
            for i, sentence in enumerate(first_sentences):
                if "(CA IRB)" in sentence:
                    found_idx = i
                    ca_irb_found_count += 1
                    break
            
            # Process if "(CA IRB)" was found
            if found_idx >= 0:
                # Split the sentence containing "(CA IRB)" at the exact position
                ca_irb_sentence = sentences[found_idx]
                split_pos = ca_irb_sentence.find("(CA IRB)")
                
                before_ca_irb = ca_irb_sentence[:split_pos].strip()
                from_ca_irb = ca_irb_sentence[split_pos:].strip()
                
                # 1. Extract all sentences from beginning to just before "(CA IRB)"
                prefix_sentences = sentences[:found_idx]
                if before_ca_irb:  # Add the part before (CA IRB) if it exists
                    general_metadata = " ".join(prefix_sentences + [before_ca_irb]).strip()
                else:
                    general_metadata = " ".join(prefix_sentences).strip()
                
                df.at[idx, 'general_metadata'] = general_metadata
                extracted_sentence_counts.append(len(prefix_sentences) + (1 if before_ca_irb else 0))
                
                # 2. Extract suspected first case paragraph ((CA IRB) and after + 7 more sentences)
                next_sentences = sentences[found_idx+1:found_idx+8]  # Get 7 sentences after the (CA IRB) sentence
                suspected_first_para = from_ca_irb + " " + " ".join(next_sentences).strip()
                df.at[idx, 'suspected_first_case_paragraph'] = suspected_first_para.strip()
                
                # 3. Remove all sentences up to and including the part before "(CA IRB)"
                if 'cleaned_text' in df.columns and not pd.isna(row['cleaned_text']):
                    # Get cleaned text
                    cleaned_text = row['cleaned_text']
                    
                    # Try to find the general metadata text in the cleaned text
                    if general_metadata and general_metadata in cleaned_text:
                        new_text = cleaned_text.replace(general_metadata, '', 1).strip()
                        
                        # If the remaining text starts with the (CA IRB) part, it worked properly
                        if new_text.startswith(from_ca_irb):
                            df.at[idx, 'cleaned_text'] = new_text
                        else:
                            # Fall back to simpler approach - look for (CA IRB) in the text
                            ca_irb_pos = cleaned_text.find("(CA IRB)")
                            if ca_irb_pos > 0:
                                df.at[idx, 'cleaned_text'] = cleaned_text[ca_irb_pos:].strip()
                    else:
                        # Fall back to simpler approach - look for (CA IRB) in the text
                        ca_irb_pos = cleaned_text.find("(CA IRB)")
                        if ca_irb_pos > 0:
                            df.at[idx, 'cleaned_text'] = cleaned_text[ca_irb_pos:].strip()
        
        # Calculate and display statistics
        not_found_count = len(df) - ca_irb_found_count
        print(f"\nGeneral Metadata Statistics:")
        print(f"  Documents with (CA IRB) found: {ca_irb_found_count} ({ca_irb_found_count/len(df):.1%})")
        print(f"  Documents without (CA IRB) in first 50 sentences: {not_found_count} ({not_found_count/len(df):.1%})")
        
        if extracted_sentence_counts:
            avg_sentences = sum(extracted_sentence_counts) / len(extracted_sentence_counts)
            print(f"  Average sentences extracted: {avg_sentences:.2f}")
            
            # Plot histogram if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.hist(extracted_sentence_counts, bins=range(1, max(extracted_sentence_counts) + 2))
                plt.xlabel('Number of Sentences Extracted')
                plt.ylabel('Frequency')
                plt.title('Distribution of Sentences Extracted for General Metadata')
                plt.tight_layout()
                
                # Save plot
                plot_path = Path(__file__).parent / "general_metadata_histogram.png"
                plt.savefig(plot_path)
                print(f"  Histogram saved to {plot_path}")
                
                # Display if in interactive environment
                try:
                    plt.show()
                except:
                    pass
            except ImportError:
                print("  Could not generate histogram: matplotlib not available")
    
    @staticmethod
    def _print_statistics(df: pd.DataFrame, essential_categories: List[str]) -> None:
        """Print statistics about the processed dataframe."""
        # Calculate statistics
        stats = {
            'total_documents': len(df),
            'documents_with_metadata': df['case_metadata'].notna().sum(),
            'documents_with_general_metadata': df['general_metadata'].notna().sum()
        }
        
        # Add stats for each essential section category
        for category in essential_categories:
            stats[f'documents_with_{category}'] = df[f'has_{category}'].sum()
        
        print("\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value} ({value/stats['total_documents']:.1%})")


def process_validation_set(validation_path: Path, output_path: Optional[Path] = None, 
                           sample_size: int = 10, batch_size: int = 50) -> pd.DataFrame:
    """Process validation dataset with option to run on sample first."""
    if not validation_path.exists():
        print(f"Error: Validation file not found at {validation_path}")
        print("Current directory:", os.getcwd())
        return pd.DataFrame()
    
    # Add this to ensure output directory exists
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load validation set
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded validation set with {len(validation_df)} cases")
    
    # Process a small sample if requested
    if sample_size > 0:
        sample_df = validation_df.sample(min(sample_size, len(validation_df)), random_state=42)
        print(f"Processing sample of {len(sample_df)} cases first...")
        
        processor = DatasetProcessor()
        sample_processed_df = processor.process_dataframe(sample_df, batch_size=min(5, batch_size))
        
        # Check if we should proceed with full dataset
        process_full = input("\nDo you want to process the full dataset? (y/n): ").strip().lower() == 'y'
        if not process_full:
            print("Skipping full dataset processing.")
            return sample_processed_df
    
    # Process full dataset
    print(f"Processing full dataset with {len(validation_df)} cases...")
    processor = DatasetProcessor()
    processed_df = processor.process_dataframe(validation_df, batch_size=batch_size)
    
    # Save the preprocessed dataframe if output path provided
    if output_path:
        processed_df.to_csv(output_path, index=False)
        print(f"Saved preprocessed validation set to {output_path}")
    
    return processed_df


def main():
    """Main function to process the validation set."""
    # Define validation file path
       # Use the correct path to validation_set.csv in pipeline/data directory
    base_dir = Path(__file__).parent.parent
    validation_path =  base_dir / "data" / "determination_extraction_set.csv"
    output_path = base_dir / "data" / "preprocessed_determination_extraction_set.csv"
    
    process_validation_set(validation_path, output_path)
        

if __name__ == "__main__":
    main()
