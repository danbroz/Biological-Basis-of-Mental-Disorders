#!/usr/bin/env python3
"""
Search for relevant papers for each DSM-5 disorder using OpenAlex RAG.
Processes mental-disorder-and-criteria.txt and exports 2000 papers per disorder.
"""

import os
import re
import json
import time
from datetime import datetime
from openalex_rag import OpenAlexRAG


def sanitize_filename(name):
    """
    Convert disorder name to valid filename.
    
    Args:
        name: Disorder name string
        
    Returns:
        Sanitized filename string
    """
    # Remove special characters
    name = re.sub(r'[^\w\s-]', '', name)
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[-\s]+', '_', name)
    # Convert to lowercase and strip
    name = name.lower().strip('_')
    # Truncate if too long (keep max 100 chars)
    if len(name) > 100:
        name = name[:100]
    return name


def parse_disorders_file(filepath):
    """
    Parse mental-disorder-and-criteria.txt to extract disorder blocks.
    
    Args:
        filepath: Path to the disorders file
        
    Returns:
        List of tuples (disorder_name, criteria_text)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    disorders = []
    
    # Split by double newlines to get blocks
    blocks = content.split('\n\n')
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.split('\n')
        
        # Skip very short blocks (likely not disorders)
        if len(lines) < 2:
            continue
        
        # First line is typically the disorder name
        disorder_name = lines[0].strip()
        
        # Skip if first line looks like criteria (starts with number, bullet, or is very long)
        if (disorder_name.startswith(tuple('0123456789')) or 
            len(disorder_name) > 150 or
            disorder_name.lower().startswith('note:') or
            disorder_name.lower().startswith('specify')):
            continue
        
        # Get all criteria text (rest of the lines)
        criteria_text = '\n'.join(lines[1:]).strip()
        
        # Skip if no criteria or if block is too short
        if not criteria_text or len(criteria_text) < 50:
            continue
        
        disorders.append((disorder_name, criteria_text))
    
    return disorders


def search_disorder_papers(disorder_name, criteria_text, rag, k=2000):
    """
    Search for papers relevant to a specific disorder.
    
    Args:
        disorder_name: Name of the disorder
        criteria_text: Full diagnostic criteria text
        rag: OpenAlexRAG instance
        k: Number of papers to retrieve
        
    Returns:
        List of Paper objects
    """
    # Combine name and criteria for comprehensive search
    query = f"{disorder_name}\n{criteria_text}"
    
    print(f"  🔍 Searching for {k} papers...")
    papers = rag.search_papers(query, k=k)
    
    return papers


def save_papers_to_json(papers, filename, disorder_name):
    """
    Save papers to JSON file.
    
    Args:
        papers: List of Paper objects
        filename: Output filename
        disorder_name: Name of disorder (for metadata)
    """
    # Convert papers to dictionaries
    papers_data = {
        'disorder': disorder_name,
        'paper_count': len(papers),
        'generated_at': datetime.now().isoformat(),
        'papers': [paper.to_dict() for paper in papers]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers_data, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved to: {filename}")


def main():
    """Main function to process all disorders."""
    print("=" * 80)
    print("DSM-5 Disorder Paper Search")
    print("=" * 80)
    
    # Configuration
    disorders_file = "mental-disorder-and-criteria.txt"
    output_dir = "disorder_papers"
    papers_per_disorder = 2000
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    # Initialize OpenAlex RAG (reuse for all searches)
    print("\n🚀 Initializing OpenAlex RAG...")
    try:
        rag = OpenAlexRAG(
            email="dan@danbroz.com",
            k=papers_per_disorder,
            use_mongodb=False
        )
        print("✅ RAG initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    # Parse disorders from file
    print(f"\n📖 Parsing disorders from: {disorders_file}")
    try:
        disorders = parse_disorders_file(disorders_file)
        print(f"✅ Found {len(disorders)} disorders")
    except Exception as e:
        print(f"❌ Failed to parse disorders file: {e}")
        return
    
    # Process each disorder
    print(f"\n🔬 Processing {len(disorders)} disorders...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for idx, (disorder_name, criteria_text) in enumerate(disorders, 1):
        try:
            print(f"\n[{idx}/{len(disorders)}] {disorder_name}")
            print("-" * 80)
            
            # Create sanitized filename
            safe_name = sanitize_filename(disorder_name)
            output_file = os.path.join(output_dir, f"{safe_name}.json")
            
            # Skip if already exists
            if os.path.exists(output_file):
                print(f"  ⏭️  Skipping (already exists): {output_file}")
                successful += 1
                continue
            
            # Search for papers
            disorder_start = time.time()
            papers = search_disorder_papers(
                disorder_name, 
                criteria_text, 
                rag, 
                k=papers_per_disorder
            )
            disorder_time = time.time() - disorder_start
            
            if papers:
                # Save to JSON
                save_papers_to_json(papers, output_file, disorder_name)
                print(f"  ✅ Retrieved {len(papers)} papers in {disorder_time:.1f}s")
                successful += 1
            else:
                print(f"  ⚠️  No papers found")
                failed += 1
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(disorders) - idx) * avg_time
            print(f"  ⏱️  Progress: {idx}/{len(disorders)} | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"Remaining: {remaining/60:.1f}m")
            
        except Exception as e:
            print(f"  ❌ Error processing {disorder_name}: {e}")
            failed += 1
            continue
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Total disorders: {len(disorders)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per disorder: {total_time/len(disorders):.1f}s")
    print(f"\n📁 Output directory: {output_dir}/")
    print("=" * 80)
    
    # Close RAG
    rag.close()
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()

