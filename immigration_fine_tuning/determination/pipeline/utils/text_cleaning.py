import re
import json
import ast
from typing import Any

def clean_text(text: Any) -> str:
    """
    Clean text robustly for processing by various extractors.
    
    Args:
        text: Input text to clean (can be string, list, dict, or other types)
        
    Returns:
        Cleaned text string
    """
    # 1. Handle non-string inputs or string representations of lists/dicts
    if not isinstance(text, str):
        if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
            text = text[0]  # Extract string from single-element list
        elif isinstance(text, dict):
            # Combine values from dict
            text = ' '.join(str(v) for v in text.values() if isinstance(v, str))
        else:
            text = str(text) # Force conversion

    # Handle cases where the string itself might be a JSON/list representation
    text = text.strip()
    if text.startswith("['") and text.endswith("']"):
        try:
            # Attempt to parse as a list literal
            parsed_list = ast.literal_eval(text)
            if isinstance(parsed_list, list) and len(parsed_list) == 1 and isinstance(parsed_list[0], str):
                text = parsed_list[0]
        except:
            # If parsing fails, just strip the outer brackets/quotes manually
            text = text.strip("[]'")
    elif text.startswith('{"') and text.endswith('"}'):
         try:
             # Attempt to parse as JSON dict
             parsed_dict = json.loads(text)
             if isinstance(parsed_dict, dict):
                 text = ' '.join(str(v) for v in parsed_dict.values() if isinstance(v, str))
         except:
             # If parsing fails, proceed with the raw string
             pass

    # 2. Basic substitutions (newlines, unicode)
    text = text.replace('\\n', ' ').replace('\n', ' ') # Handle escaped and actual newlines
    text = re.sub(r'\\u[0-9a-fA-F]{4}', ' ', text) # Remove unicode escapes

    # 3. Remove REDACTED patterns (more broadly)
    text = re.sub(r'\[REDACTED.*?\]', ' ', text, flags=re.IGNORECASE)

    # 4. Remove bracketed numbers (often paragraph markers)
    text = re.sub(r'\[\d+\]', ' ', text)

    # 5. Remove specific number sequences (like file IDs, often 5+ digits)
    # Be cautious not to remove meaningful numbers like section numbers (e.g., section 7)
    text = re.sub(r'\b\d{5,}\b', ' ', text) # Remove standalone sequences of 5 or more digits

    # 6. Remove Citations (CanLII)
    text = re.sub(r'\d{4}\s+CanLII\s+\d+\s*\(CA\s*IRB\)', ' ', text)

    # 7. Remove Dates (various formats)
    # Full month names
    text = re.sub(r'(?i)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', ' ', text)
    # Numeric dates (dd/mm/yyyy, mm/dd/yyyy, yyyy-mm-dd etc.) - adjust slashes/dashes as needed
    text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', ' ', text)
    # Standalone years (if problematic, e.g., "Decision Date 2012") - might be too broad
    # text = re.sub(r'\b(19|20)\d{2}\b', ' ', text) # Use with caution

    # 8. Remove file numbers and similar administrative text
    text = re.sub(r'(?i)IAD\s+File\s+No\.*\s*/\s*No\s+de\s+dossier\s+de\s+la\s+SAI\s*:.*?(?=\s\s|\n|$)', ' ', text)
    text = re.sub(r'(?i)(?:File|Client|UCI|Application)\s+(?:No|Number|ID)\.*\s*:.*?(?=\s\s|\n|$)', ' ', text)
    text = re.sub(r'(?i)Date\s*:\s*.*?(?=\s\s|\n|$)', ' ', text) # Remove "Date:" lines

    # 9. Remove Signature blocks / Boilerplate
    text = re.sub(r'(?i)\(signed\).*?(?=\s\s|\n|$)', ' ', text)
    text = re.sub(r'(?i)Judicial\s+Review\s+[-–]\s+Under\s+section\s+72.*?application\.', ' ', text, flags=re.DOTALL) # DOTALL allows '.' to match newline

    # 10. Final cleanup (whitespace, residual non-alphanumeric unless basic punctuation)
    text = re.sub(r'\s+', ' ', text) # Consolidate whitespace
    # Keep letters, numbers, spaces, and basic punctuation . , ! ? ; : - '
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text) # Consolidate again after character removal

    return text.strip() 