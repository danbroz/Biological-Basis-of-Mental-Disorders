"""
Core citation cleaning functionality.
"""

import re
from typing import List, Tuple, Set, Dict, Optional
from .utils import (
    normalize_token, 
    validate_apa_author, 
    extract_citation_parts,
    calculate_match_score,
    clean_reference_text
)


class CitationCleaner:
    """
    Main class for cleaning academic references to match only cited sources.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the citation cleaner.
        
        Args:
            strict_mode: If True, only include references with valid APA format
        """
        self.strict_mode = strict_mode
        self.citation_pattern = r'\(([^)]+\d{4}[a-z]?[^)]*)\)'
    
    def extract_citations(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all parenthetical citations from text.
        
        Args:
            text: Text content to extract citations from
            
        Returns:
            List of (surname, year) tuples from citations
        """
        citations = re.findall(self.citation_pattern, text)
        cited_pairs = set()
        
        for citation in citations:
            author_year_pairs = extract_citation_parts(citation)
            cited_pairs.update(author_year_pairs)
        
        return list(cited_pairs)
    
    def match_reference(self, cited_surname: str, cited_year: str, 
                       reference: str) -> int:
        """
        Calculate match score between a cited author+year and a reference.
        
        Args:
            cited_surname: Surname from citation
            cited_year: Year from citation
            reference: Full reference text
            
        Returns:
            Match score (0-100, higher is better)
        """
        # Clean reference text
        ref_text = clean_reference_text(reference)
        
        # Extract year from reference
        year_match = re.search(r'\((\d{4}[a-z]?)\)', ref_text)
        if not year_match or year_match.group(1) != cited_year:
            return 0
        
        # Extract author text before year
        authors_text = ref_text.split('(')[0].strip()
        
        # Extract author surnames
        ref_words = re.findall(r'\b[A-Za-zà-ÿ\-]+\b', authors_text)
        
        # Calculate match score
        return calculate_match_score(cited_surname, ref_words)
    
    def clean_references(self, text_content: str, all_references: List[str]) -> Dict:
        """
        Clean references to only include those actually cited in the text.
        
        Args:
            text_content: Text content with citations
            all_references: List of all available references
            
        Returns:
            Dictionary with cleaned references and statistics
        """
        # Extract cited author+year pairs from text
        cited_pairs = self.extract_citations(text_content)
        
        # Find best matches for each cited pair
        matched_references = []
        matched_pairs = set()
        match_scores = {}
        
        for cited_surname, cited_year in cited_pairs:
            best_match = None
            best_score = 0
            
            for ref in all_references:
                score = self.match_reference(cited_surname, cited_year, ref)
                
                if score > best_score:
                    # Validate reference format if in strict mode
                    if not self.strict_mode or validate_apa_author(ref):
                        best_match = ref
                        best_score = score
            
            if best_match and best_match not in matched_references:
                matched_references.append(best_match)
                matched_pairs.add((cited_surname, cited_year))
                match_scores[best_match] = best_score
        
        # Sort references alphabetically (APA style)
        matched_references.sort(key=lambda x: x.lower())
        
        # Create numbered references
        numbered_refs = [f"{i}. {ref}" for i, ref in enumerate(matched_references, 1)]
        
        # Calculate statistics
        stats = {
            'total_citations': len(cited_pairs),
            'matched_citations': len(matched_pairs),
            'unmatched_citations': len(cited_pairs) - len(matched_pairs),
            'total_references': len(all_references),
            'filtered_references': len(matched_references),
            'reduction_percentage': round((1 - len(matched_references) / len(all_references)) * 100, 1) if all_references else 0
        }
        
        return {
            'references': numbered_refs,
            'unmatched_references': matched_references,
            'statistics': stats,
            'match_scores': match_scores
        }
    
    def clean_document(self, content: str, reference_section: str) -> str:
        """
        Clean a document by replacing the references section with filtered references.
        
        Args:
            content: Full document content
            reference_section: References section to replace
            
        Returns:
            Document with cleaned references
        """
        # Split content at references section
        parts = content.split('\n## References\n')
        if len(parts) != 2:
            return content  # No references section found
        
        main_content, _ = parts
        
        # Extract all references from the section
        ref_lines = [line.strip() for line in reference_section.split('\n') 
                    if line.strip() and re.match(r'^\d+\.', line)]
        
        # Clean references
        result = self.clean_references(content, ref_lines)
        
        # Rebuild document
        cleaned_content = main_content + '\n\n## References\n\n'
        cleaned_content += '\n'.join(result['references'])
        
        return cleaned_content
    
    def get_unmatched_citations(self, text_content: str, all_references: List[str]) -> List[str]:
        """
        Get list of citations that couldn't be matched to references.
        
        Args:
            text_content: Text content with citations
            all_references: List of all available references
            
        Returns:
            List of unmatched citation strings
        """
        cited_pairs = self.extract_citations(text_content)
        unmatched = []
        
        for cited_surname, cited_year in cited_pairs:
            found_match = False
            for ref in all_references:
                if self.match_reference(cited_surname, cited_year, ref) > 0:
                    found_match = True
                    break
            
            if not found_match:
                unmatched.append(f"{cited_surname}, {cited_year}")
        
        return unmatched
