#!/usr/bin/env python3
"""
Example script demonstrating how to use the Citation Cleaner library
to clean academic documents and remove non-cited references.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import citation_cleaner
sys.path.insert(0, str(Path(__file__).parent.parent))

from citation_cleaner import CitationCleaner


def clean_single_document(filepath: str) -> None:
    """
    Clean a single document file.
    
    Args:
        filepath: Path to the document file
    """
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into main content and references
        parts = content.split('\n## References\n')
        if len(parts) != 2:
            print(f"  ⚠️  No References section found in {filepath}")
            return
        
        main_content, refs_section = parts
        
        # Extract all references
        ref_lines = [line.strip() for line in refs_section.split('\n') 
                    if line.strip() and line.startswith(('1.', '2.', '3.'))]
        
        # Clean references
        cleaner = CitationCleaner()
        result = cleaner.clean_references(content, ref_lines)
        
        # Display statistics
        stats = result['statistics']
        print(f"  ✅ Cleaned: {stats['total_references']} → {stats['filtered_references']} references")
        print(f"  📊 Reduction: {stats['reduction_percentage']}%")
        print(f"  📝 Citations: {stats['matched_citations']}/{stats['total_citations']} matched")
        
        # Write cleaned document
        cleaned_content = main_content + '\n\n## References\n\n'
        cleaned_content += '\n'.join(result['references'])
        
        # Save to new file
        output_path = filepath.replace('.md', '_cleaned.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"  💾 Saved cleaned document to: {output_path}")
        
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")


def batch_clean_documents(directory: str) -> None:
    """
    Clean all markdown documents in a directory.
    
    Args:
        directory: Directory containing documents to clean
    """
    print(f"Cleaning all documents in: {directory}")
    
    # Find all markdown files
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md') and not file.endswith('_cleaned.md'):
                md_files.append(os.path.join(root, file))
    
    md_files.sort()
    
    if not md_files:
        print("No markdown files found.")
        return
    
    print(f"Found {len(md_files)} documents to process...")
    
    total_original = 0
    total_cleaned = 0
    
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into main content and references
            parts = content.split('\n## References\n')
            if len(parts) != 2:
                continue
            
            main_content, refs_section = parts
            
            # Extract all references
            ref_lines = [line.strip() for line in refs_section.split('\n') 
                        if line.strip() and line.startswith(('1.', '2.', '3.'))]
            
            if not ref_lines:
                continue
            
            # Clean references
            cleaner = CitationCleaner()
            result = cleaner.clean_references(content, ref_lines)
            
            # Update totals
            stats = result['statistics']
            total_original += stats['total_references']
            total_cleaned += stats['filtered_references']
            
            # Write cleaned document
            cleaned_content = main_content + '\n\n## References\n\n'
            cleaned_content += '\n'.join(result['references'])
            
            # Save to new file
            output_path = filepath.replace('.md', '_cleaned.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"  ✅ {os.path.basename(filepath)}: {stats['total_references']} → {stats['filtered_references']} references")
            
        except Exception as e:
            print(f"  ❌ Error processing {filepath}: {e}")
    
    # Display overall statistics
    if total_original > 0:
        reduction = round((1 - total_cleaned / total_original) * 100, 1)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total original references: {total_original}")
        print(f"  Total cleaned references: {total_cleaned}")
        print(f"  Overall reduction: {reduction}%")


def demonstrate_basic_usage():
    """
    Demonstrate basic usage of the Citation Cleaner library.
    """
    print("=== Citation Cleaner Library Demo ===\n")
    
    # Sample text with citations
    sample_text = """
    This is a sample academic text with various citations.
    
    Research has shown that cognitive abilities are important (Smith, 2021).
    Multiple studies have confirmed these findings (Johnson et al., 2020; Brown & Davis, 2019).
    However, some researchers disagree (Wilson, 2018).
    
    The methodology used in these studies was rigorous (Taylor, 2022).
    """
    
    # Sample reference list (including some uncited references)
    sample_references = [
        "Smith, J. (2021). Cognitive abilities in academic performance. Journal of Education.",
        "Johnson, A., Wilson, B., & Lee, C. (2020). Multiple studies on learning. Learning Journal.",
        "Brown, M., & Davis, K. (2019). Research methodology. Research Quarterly.",
        "Wilson, R. (2018). Alternative perspectives. Alternative Journal.",
        "Taylor, S. (2022). Rigorous methodology. Methodology Review.",
        "Uncited, A. (2023). This reference won't be included. Uncited Journal.",
        "Another, B. (2023). This one either. Another Journal."
    ]
    
    print("Sample text with citations:")
    print(sample_text)
    print("\nOriginal references:")
    for i, ref in enumerate(sample_references, 1):
        print(f"{i}. {ref}")
    
    # Clean the references
    cleaner = CitationCleaner()
    result = cleaner.clean_references(sample_text, sample_references)
    
    print(f"\n=== Cleaning Results ===")
    stats = result['statistics']
    print(f"Total citations found: {stats['total_citations']}")
    print(f"Matched citations: {stats['matched_citations']}")
    print(f"Unmatched citations: {stats['unmatched_citations']}")
    print(f"Original references: {stats['total_references']}")
    print(f"Filtered references: {stats['filtered_references']}")
    print(f"Reduction: {stats['reduction_percentage']}%")
    
    print(f"\nCleaned references:")
    for ref in result['references']:
        print(ref)
    
    # Show unmatched citations
    unmatched = cleaner.get_unmatched_citations(sample_text, sample_references)
    if unmatched:
        print(f"\nUnmatched citations:")
        for citation in unmatched:
            print(f"  - {citation}")


def main():
    """Main function to run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Citation Cleaner Examples')
    parser.add_argument('--demo', action='store_true', help='Run basic demo')
    parser.add_argument('--file', type=str, help='Clean a single file')
    parser.add_argument('--directory', type=str, help='Clean all files in directory')
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_basic_usage()
    elif args.file:
        clean_single_document(args.file)
    elif args.directory:
        batch_clean_documents(args.directory)
    else:
        print("Citation Cleaner Library Examples")
        print("Usage:")
        print("  python clean_document.py --demo                    # Run basic demo")
        print("  python clean_document.py --file document.md       # Clean single file")
        print("  python clean_document.py --directory ./docs/      # Clean directory")
        print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
