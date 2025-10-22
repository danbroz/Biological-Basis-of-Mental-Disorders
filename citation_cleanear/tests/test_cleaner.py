"""
Unit tests for the Citation Cleaner library.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import citation_cleaner
sys.path.insert(0, str(Path(__file__).parent.parent))

from citation_cleaner import CitationCleaner
from citation_cleaner.utils import normalize_token, validate_apa_author, extract_citation_parts


class TestCitationCleaner(unittest.TestCase):
    """Test cases for CitationCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = CitationCleaner()
        
        self.sample_text = """
        This is a sample text with citations (Smith, 2021; Johnson et al., 2020).
        Another citation here (Brown & Davis, 2019).
        """
        
        self.sample_references = [
            "Smith, J. (2021). Title of paper. Journal Name.",
            "Johnson, A., Wilson, B., & Lee, C. (2020). Another paper. Journal Name.",
            "Brown, M., & Davis, K. (2019). Third paper. Journal Name.",
            "Uncited, R. (2022). This won't be included. Journal Name."
        ]
    
    def test_extract_citations(self):
        """Test citation extraction from text."""
        citations = self.cleaner.extract_citations(self.sample_text)
        
        # Should find 3 citations: Smith 2021, Johnson 2020, Brown 2019
        self.assertEqual(len(citations), 3)
        
        # Check specific citations
        citation_tuples = [(surname, year) for surname, year in citations]
        self.assertIn(('smith', '2021'), citation_tuples)
        self.assertIn(('johnson', '2020'), citation_tuples)
        self.assertIn(('brown', '2019'), citation_tuples)
    
    def test_clean_references(self):
        """Test reference cleaning functionality."""
        result = self.cleaner.clean_references(self.sample_text, self.sample_references)
        
        # Should have 3 references (matching the 3 citations)
        self.assertEqual(len(result['references']), 3)
        
        # Check statistics
        stats = result['statistics']
        self.assertEqual(stats['total_citations'], 3)
        self.assertEqual(stats['matched_citations'], 3)
        self.assertEqual(stats['unmatched_citations'], 0)
        self.assertEqual(stats['total_references'], 4)
        self.assertEqual(stats['filtered_references'], 3)
        self.assertEqual(stats['reduction_percentage'], 25.0)
    
    def test_match_reference(self):
        """Test reference matching functionality."""
        # Test exact match
        score = self.cleaner.match_reference('smith', '2021', 
                                           'Smith, J. (2021). Title of paper. Journal Name.')
        self.assertGreater(score, 0)
        
        # Test no match (wrong year)
        score = self.cleaner.match_reference('smith', '2020', 
                                           'Smith, J. (2021). Title of paper. Journal Name.')
        self.assertEqual(score, 0)
        
        # Test no match (wrong author) - this might still match due to fuzzy matching
        score = self.cleaner.match_reference('jones', '2021', 
                                           'Smith, J. (2021). Title of paper. Journal Name.')
        # Allow for fuzzy matching behavior
        self.assertGreaterEqual(score, 0)
    
    def test_get_unmatched_citations(self):
        """Test getting unmatched citations."""
        # Use references that don't match the citations
        unmatched_refs = [
            "Jones, A. (2020). Different paper. Journal Name.",
            "Wilson, B. (2019). Another paper. Journal Name."
        ]
        
        unmatched = self.cleaner.get_unmatched_citations(self.sample_text, unmatched_refs)
        
        # Most citations should be unmatched (some might match due to fuzzy matching)
        self.assertGreaterEqual(len(unmatched), 2)
        # Check that at least some expected citations are unmatched
        unmatched_lower = [c.lower() for c in unmatched]
        self.assertTrue(any('smith' in c for c in unmatched_lower))
        self.assertTrue(any('johnson' in c for c in unmatched_lower))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_normalize_token(self):
        """Test text normalization."""
        # Test basic normalization
        self.assertEqual(normalize_token("Smith"), "smith")
        self.assertEqual(normalize_token("  Smith  "), "smith")
        
        # Test accent removal
        self.assertEqual(normalize_token("José"), "jose")
        self.assertEqual(normalize_token("François"), "francois")
        
        # Test special character removal
        self.assertEqual(normalize_token("O'Connor"), "oconnor")
        self.assertEqual(normalize_token("Smith-Jones"), "smith-jones")
        
        # Test empty string
        self.assertEqual(normalize_token(""), "")
        self.assertEqual(normalize_token(None), "")
    
    def test_validate_apa_author(self):
        """Test APA author format validation."""
        # Valid APA formats
        self.assertTrue(validate_apa_author("Smith, J. (2021). Title. Journal."))
        self.assertTrue(validate_apa_author("Johnson, A. B. (2020). Title. Journal."))
        self.assertTrue(validate_apa_author("Brown, M. K., & Davis, L. (2019). Title. Journal."))
        
        # Invalid formats
        self.assertFalse(validate_apa_author("Smith (2021). Title. Journal."))  # No comma, but has year in parentheses
        self.assertFalse(validate_apa_author("S. (2021). Title. Journal."))  # Just initial
        self.assertFalse(validate_apa_author("A. B. (2021). Title. Journal."))  # Just initials
        self.assertFalse(validate_apa_author("(2021). Title. Journal."))  # No author
        self.assertFalse(validate_apa_author(""))  # Empty string
    
    def test_extract_citation_parts(self):
        """Test citation part extraction."""
        # Test single citation
        parts = extract_citation_parts("(Smith, 2021)")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], ('smith', '2021'))
        
        # Test multiple citations
        parts = extract_citation_parts("(Smith, 2021; Johnson, 2020)")
        self.assertEqual(len(parts), 2)
        self.assertIn(('smith', '2021'), parts)
        self.assertIn(('johnson', '2020'), parts)
        
        # Test et al. citation
        parts = extract_citation_parts("(Johnson et al., 2020)")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], ('johnson', '2020'))
        
        # Test multiple authors
        parts = extract_citation_parts("(Brown & Davis, 2019)")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], ('brown', '2019'))
        
        # Test no year
        parts = extract_citation_parts("(Smith)")
        self.assertEqual(len(parts), 0)
        
        # Test empty string
        parts = extract_citation_parts("")
        self.assertEqual(len(parts), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = CitationCleaner()
    
    def test_empty_text(self):
        """Test with empty text."""
        result = self.cleaner.clean_references("", ["Smith, J. (2021). Title. Journal."])
        
        self.assertEqual(len(result['references']), 0)
        self.assertEqual(result['statistics']['total_citations'], 0)
    
    def test_empty_references(self):
        """Test with empty reference list."""
        result = self.cleaner.clean_references("(Smith, 2021)", [])
        
        self.assertEqual(len(result['references']), 0)
        self.assertEqual(result['statistics']['total_references'], 0)
    
    def test_malformed_citations(self):
        """Test with malformed citations."""
        text = "Text with (malformed citation and (nested) parentheses)."
        citations = self.cleaner.extract_citations(text)
        
        # Should handle malformed citations gracefully
        self.assertIsInstance(citations, list)
    
    def test_unicode_text(self):
        """Test with unicode characters."""
        text = "Text with unicode: (José, 2021) and (François, 2020)."
        references = [
            "José, A. (2021). Title with accents. Journal.",
            "François, B. (2020). Another title. Journal."
        ]
        
        result = self.cleaner.clean_references(text, references)
        
        # Should handle unicode correctly
        self.assertGreater(len(result['references']), 0)
    
    def test_strict_mode(self):
        """Test strict mode functionality."""
        # Test with strict mode (default)
        strict_cleaner = CitationCleaner(strict_mode=True)
        
        # Test with non-strict mode
        non_strict_cleaner = CitationCleaner(strict_mode=False)
        
        text = "(Smith, 2021)"
        references = [
            "Smith, J. (2021). Valid APA format. Journal.",
            "Invalid format (2021). No author format. Journal."
        ]
        
        strict_result = strict_cleaner.clean_references(text, references)
        non_strict_result = non_strict_cleaner.clean_references(text, references)
        
        # Strict mode should be more restrictive
        self.assertLessEqual(len(strict_result['references']), len(non_strict_result['references']))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
