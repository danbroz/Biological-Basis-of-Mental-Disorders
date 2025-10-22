"""
Utility functions for citation cleaning and text processing.
"""

import re
import unicodedata
from typing import List, Tuple, Set


def normalize_token(s: str) -> str:
    """
    Normalize text for comparison by removing accents, converting to lowercase,
    and cleaning special characters.
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string for comparison
    """
    if not s:
        return ''
    
    # Convert to lowercase
    s = s.strip().lower()
    
    # Remove accents and diacritics
    s = ''.join(c for c in unicodedata.normalize('NFD', s) 
                if unicodedata.category(c) != 'Mn')
    
    # Remove special characters, keep only alphanumeric, hyphens, and spaces
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    
    return s


def validate_apa_author(apa_ref_text: str) -> bool:
    """
    Check if a reference has valid APA author format.
    
    Args:
        apa_ref_text: Reference text to validate
        
    Returns:
        True if valid APA format, False otherwise
    """
    # Extract the author segment before the year
    first_segment = apa_ref_text.split('(', 1)[0].strip()
    if not first_segment:
        return False
    
    # Check if it has the APA format "Surname, A. B." or just "Surname"
    if ',' in first_segment:
        # Has APA format - check for proper surname and initials
        parts = first_segment.split(',', 1)
        if len(parts) < 2:
            return False
        
        surname = parts[0].strip()
        initials_part = parts[1].strip()
        
        # Surname must be at least 2 letters
        surname_clean = re.sub(r"[^A-Za-zÀ-ÿ'-]", "", surname)
        if len(surname_clean) < 2:
            return False
        
        # Must have at least one initial (letter followed by period)
        if not re.search(r'[A-Z]\.', initials_part):
            return False
    else:
        # No comma - check if it's a proper surname (not just initials)
        surname_clean = re.sub(r"[^A-Za-zÀ-ÿ'-]", "", first_segment)
        if len(surname_clean) < 2:
            return False
        
        # Reject if it's just initials (like "D. S.")
        if re.match(r'^[A-Z]\.?\s*[A-Z]\.?$', first_segment.strip()):
            return False
        
        # Reject if it doesn't have proper APA format (no comma but no year in parentheses)
        if not re.search(r'\(\d{4}', apa_ref_text):
            return False
        
        # Reject if it doesn't have a comma (proper APA format requires "Surname, A. B.")
        return False
    
    return True


def extract_citation_parts(citation_text: str) -> List[Tuple[str, str]]:
    """
    Extract author and year pairs from citation text.
    
    Args:
        citation_text: Text containing citations like "(Author, Year)" or "(Author et al., Year)"
        
    Returns:
        List of (author, year) tuples
    """
    parts = [p.strip() for p in citation_text.split(';')]
    author_year_pairs = []
    
    for part in parts:
        year_match = re.search(r'\b(\d{4}[a-z]?)\b', part)
        if not year_match:
            continue
            
        year = year_match.group(1)
        text_before_year = part.split(year)[0].strip().rstrip(',').strip()
        
        # Remove leading parenthesis if present
        if text_before_year.startswith('('):
            text_before_year = text_before_year[1:].strip()
        
        # Handle different citation formats
        if ' et al.' in text_before_year:
            first_author = text_before_year.split(' et al.')[0].strip()
        elif ' & ' in text_before_year:
            first_author = text_before_year.split(' & ')[0].strip()
        elif ', ' in text_before_year:
            first_author = text_before_year.split(', ')[0].strip()
        else:
            first_author = text_before_year.strip()
            
        # Extract surname (last word)
        surname = first_author.split()[-1] if first_author else first_author
        if surname:
            author_year_pairs.append((surname.lower(), year))
    
    return author_year_pairs


def calculate_match_score(cited_surname: str, ref_authors: List[str]) -> int:
    """
    Calculate match score between cited surname and reference authors.
    
    Args:
        cited_surname: Surname from citation
        ref_authors: List of author surnames from reference
        
    Returns:
        Match score (0-100, higher is better)
    """
    norm_cited = normalize_token(cited_surname)
    norm_refs = [normalize_token(author) for author in ref_authors]
    
    # Exact match gets highest score
    if norm_cited in norm_refs:
        return 100
    
    # Check for prefix/variant matches
    for ref_author in norm_refs:
        if (norm_cited and len(norm_cited) >= 2 and 
            (ref_author.startswith(norm_cited) or norm_cited.startswith(ref_author))):
            return 50
    
    return 0


def clean_reference_text(ref: str) -> str:
    """
    Clean reference text by removing numbering and extra whitespace.
    
    Args:
        ref: Raw reference text
        
    Returns:
        Cleaned reference text
    """
    # Remove numbering if present (e.g., "1. Author, A. (2021)" -> "Author, A. (2021)")
    if re.match(r'^\d+\. ', ref):
        return re.sub(r'^\d+\. ', '', ref)
    return ref.strip()
