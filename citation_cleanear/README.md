# Citation Cleaner Library

A Python library for filtering academic references to match only in-text citations, eliminating hallucinated or non-cited references.

## Features

- **Smart Citation Matching**: Accurately matches in-text citations to reference entries
- **Multiple Citation Formats**: Handles (Author, Year), (Author et al., Year), (Author & Author, Year)
- **Fuzzy Matching**: Supports name variants (Mc/Mac, hyphenated names, accents)
- **APA Format Validation**: Ensures references follow proper APA formatting
- **Detailed Statistics**: Provides match/unmatch statistics and reduction percentages
- **Format Agnostic**: Works with markdown, LaTeX, plain text, and other formats

## Installation

```bash
pip install citation-cleaner
```

Or install from source:

```bash
git clone https://github.com/example/citation-cleaner.git
cd citation-cleaner
pip install -e .
```

## Quick Start

```python
from citation_cleaner import CitationCleaner

# Initialize the cleaner
cleaner = CitationCleaner()

# Your text content with citations
text_content = """
This is a sample text with citations (Smith, 2021; Johnson et al., 2020).
Another citation here (Brown & Davis, 2019).
"""

# Your reference list
all_references = [
    "Smith, J. (2021). Title of paper. Journal Name.",
    "Johnson, A., Wilson, B., & Lee, C. (2020). Another paper. Journal Name.",
    "Brown, M., & Davis, K. (2019). Third paper. Journal Name.",
    "Uncited, R. (2022). This won't be included. Journal Name."
]

# Clean the references
result = cleaner.clean_references(text_content, all_references)

# Get filtered references
filtered_refs = result['references']
print("Filtered References:")
for ref in filtered_refs:
    print(ref)

# Get statistics
stats = result['statistics']
print(f"\nStatistics:")
print(f"Total citations: {stats['total_citations']}")
print(f"Matched citations: {stats['matched_citations']}")
print(f"References reduced: {stats['reduction_percentage']}%")
```

## API Reference

### CitationCleaner Class

#### `__init__(strict_mode=True)`
Initialize the citation cleaner.

- `strict_mode` (bool): If True, only include references with valid APA format

#### `clean_references(text_content, all_references)`
Clean references to only include those actually cited in the text.

**Parameters:**
- `text_content` (str): Text content with citations
- `all_references` (List[str]): List of all available references

**Returns:**
- Dictionary with:
  - `references`: List of numbered, filtered references
  - `unmatched_references`: List of matched references (unnumbered)
  - `statistics`: Dictionary with match statistics
  - `match_scores`: Dictionary of reference match scores

#### `extract_citations(text)`
Extract all parenthetical citations from text.

**Parameters:**
- `text` (str): Text content to extract citations from

**Returns:**
- List of (surname, year) tuples from citations

#### `clean_document(content, reference_section)`
Clean a document by replacing the references section with filtered references.

**Parameters:**
- `content` (str): Full document content
- `reference_section` (str): References section to replace

**Returns:**
- Document with cleaned references

#### `get_unmatched_citations(text_content, all_references)`
Get list of citations that couldn't be matched to references.

**Parameters:**
- `text_content` (str): Text content with citations
- `all_references` (List[str]): List of all available references

**Returns:**
- List of unmatched citation strings

## Examples

### Basic Usage

```python
from citation_cleaner import CitationCleaner

cleaner = CitationCleaner()
result = cleaner.clean_references(text, references)
```

### Batch Processing

```python
import os
from citation_cleaner import CitationCleaner

cleaner = CitationCleaner()

# Process multiple files
for filename in os.listdir('documents/'):
    if filename.endswith('.md'):
        with open(f'documents/{filename}', 'r') as f:
            content = f.read()
        
        # Extract references section
        parts = content.split('\n## References\n')
        if len(parts) == 2:
            main_content, refs_section = parts
            ref_lines = [line.strip() for line in refs_section.split('\n') 
                        if line.strip() and line.startswith(('1.', '2.', '3.'))]
            
            # Clean references
            result = cleaner.clean_references(content, ref_lines)
            
            # Save cleaned document
            cleaned_content = main_content + '\n\n## References\n\n'
            cleaned_content += '\n'.join(result['references'])
            
            with open(f'cleaned_{filename}', 'w') as f:
                f.write(cleaned_content)
```

### Advanced Configuration

```python
# Use non-strict mode to include references with formatting issues
cleaner = CitationCleaner(strict_mode=False)

# Get detailed statistics
result = cleaner.clean_references(text, references)
stats = result['statistics']

print(f"Reduced references from {stats['total_references']} to {stats['filtered_references']}")
print(f"Match rate: {stats['matched_citations']}/{stats['total_citations']} citations matched")
```

## Citation Formats Supported

The library recognizes these citation formats:

- `(Author, Year)` - Single author
- `(Author & Author, Year)` - Two authors  
- `(Author et al., Year)` - Multiple authors
- `(Author, Year; Author, Year)` - Multiple citations
- `(Author, Year, p. 123)` - Citations with page numbers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v1.0.0
- Initial release
- Core citation matching functionality
- APA format validation
- Fuzzy matching for name variants
- Detailed statistics and reporting
