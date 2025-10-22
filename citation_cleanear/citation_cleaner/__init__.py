"""
Citation Cleaner Library

A Python library for filtering academic references to match only in-text citations,
eliminating hallucinated or non-cited references.

Usage:
    from citation_cleaner import CitationCleaner
    
    cleaner = CitationCleaner()
    filtered_refs = cleaner.clean_references(text_content, all_references)
"""

from .cleaner import CitationCleaner
from .utils import normalize_token, validate_apa_author

__version__ = "1.0.0"
__author__ = "Citation Cleaner Team"
__email__ = "citation.cleaner@example.com"

__all__ = [
    "CitationCleaner",
    "normalize_token", 
    "validate_apa_author"
]
