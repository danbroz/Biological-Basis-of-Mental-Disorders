#!/usr/bin/env python3
"""
Generate comprehensive book on the Biological Basis of Mental Health Disorders
using Gemini 2.5 Pro API with RAG from 2000 papers per disorder.

Each of 153 DSM-5 disorders becomes a chapter with full APA citations.
"""

import google.generativeai as genai
import json
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create a .env file with your API key.")
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 2.5 Pro (Free Tier)
# Note: Will use gemini-2.0-flash-exp if 2.5 not available
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("✅ Using Gemini 2.5 Pro")
except:
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    print("✅ Using Gemini 2.0 Flash (2M context)")

# Configuration
OUTPUT_DIR = "book_chapters"
PROGRESS_FILE = "book_progress.json"
FINAL_BOOK = "Biological_Basis_Mental_Health_Disorders.md"
RATE_LIMIT_DELAY = 6  # seconds between requests (10 RPM = 6s delay)


def generate_apa_citation(paper):
    """
    Generate APA format citation for a paper.
    
    Args:
        paper: Dictionary with paper details
        
    Returns:
        APA formatted citation string
    """
    authors = paper.get('authors', [])
    year = paper.get('year', 'n.d.')
    title = paper.get('title', 'No title')
    venue = paper.get('venue', '')
    doi = paper.get('doi', '')
    
    # Format authors
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]}, & {authors[1]}"
    elif len(authors) <= 20:
        author_str = ', '.join(authors[:-1]) + f", & {authors[-1]}"
    else:
        # 20+ authors: first 19, then ..., then last
        author_str = ', '.join(authors[:19]) + f", ... {authors[-1]}"
    
    # Build citation
    citation = f"{author_str} ({year}). {title}"
    if venue:
        citation += f". {venue}"
    if doi:
        citation += f". {doi}"
    
    return citation


def load_disorder_papers(json_filepath):
    """
    Load ALL papers from a disorder JSON file with FULL content.
    
    Args:
        json_filepath: Path to disorder JSON file
        
    Returns:
        Tuple of (papers_context_string, reference_list_string, paper_count, disorder_name)
    """
    print(f"  📖 Loading papers from: {json_filepath.name}")
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    disorder_name = data.get('disorder', json_filepath.stem.replace('_', ' ').title())
    papers = data.get('papers', [])
    
    print(f"  📚 Found {len(papers)} papers for {disorder_name}")
    
    # Format ALL papers for context (stuffed context window)
    context_parts = []
    reference_parts = []
    
    for i, paper in enumerate(papers, 1):
        # Full paper context
        authors_str = ', '.join(paper.get('authors', [])[:10])  # First 10 authors for context
        if len(paper.get('authors', [])) > 10:
            authors_str += ' et al.'
        
        context_part = f"""
Paper {i}:
Title: {paper.get('title', 'No title')}
Authors: {authors_str}
Year: {paper.get('year', 'n.d.')}
Venue: {paper.get('venue', 'Unknown')}
DOI: {paper.get('doi', 'N/A')}
Abstract: {paper.get('abstract', 'No abstract available')}
---
"""
        context_parts.append(context_part)
        
        # APA reference
        reference_parts.append(generate_apa_citation(paper))
    
    papers_context = '\n'.join(context_parts)
    reference_list = '\n'.join([f"{i+1}. {ref}" for i, ref in enumerate(reference_parts)])
    
    return papers_context, reference_list, len(papers), disorder_name


def build_chapter_prompt(disorder_name, papers_context, reference_list, paper_count):
    """
    Build comprehensive prompt for chapter generation.
    
    Args:
        disorder_name: Name of the disorder
        papers_context: Full text of all papers
        reference_list: APA formatted reference list
        paper_count: Number of papers included
        
    Returns:
        Complete prompt string
    """
    prompt = f"""You are writing a comprehensive academic chapter on the biological and neurological basis of {disorder_name}.

**Instructions:**

1. Write a detailed, evidence-based chapter (4000-6000 words) on the biological/neurological mechanisms underlying {disorder_name}

2. Cover topics including:
   - Neurotransmitter systems and neuropharmacology
   - Brain regions, neural circuits, and functional connectivity
   - Genetic factors, heritability, and molecular mechanisms
   - Cellular and synaptic mechanisms
   - Neuroimaging findings (structural and functional)
   - Developmental neurobiology and critical periods
   - Neuroplasticity, compensatory mechanisms, and treatment implications
   - Biological subtypes and heterogeneity

3. **CRITICAL - Citations (THIS IS MANDATORY):**
   - You MUST cite sources from the {paper_count} provided research papers
   - Use inline APA citations: (Author, Year) or (Author et al., Year)
   - Cite multiple papers when making broad claims: (Author1, Year1; Author2, Year2)
   - EVERY major statement, finding, or claim MUST be cited
   - Use the exact authors and years from the reference list provided below
   - Aim for 75-100+ citations throughout the chapter
   - Integrate citations naturally into the text
   - When discussing specific findings, cite the paper number if helpful for reference

4. Structure (use markdown headers):
   - ## Introduction
     Overview of {disorder_name} from a biological perspective
   
   - ## Neurobiological Systems
     Detailed coverage of implicated biological systems
   
   - ## Genetic and Molecular Basis
     Heritability, candidate genes, and molecular pathways
   
   - ## Brain Structure and Function
     Neuroimaging and neuroanatomical findings
   
   - ## Developmental Neurobiology
     How the disorder emerges across the lifespan
   
   - ## Treatment Mechanisms
     Biological basis of interventions
   
   - ## Future Directions
     Emerging research and unanswered questions
   
   - ## Conclusion
     Synthesis of biological understanding

5. Academic tone, suitable for a graduate-level neuroscience/psychiatry textbook

6. Write with authority and precision, but acknowledge uncertainties and controversies in the field

**ALL {paper_count} AVAILABLE RESEARCH PAPERS:**

{papers_context}

**COMPLETE REFERENCE LIST ({paper_count} papers - cite extensively):**
**only cite the papers that are directly relevant to the chapter**

{reference_list}

**Now write the chapter, ensuring every claim is supported by citations from the papers above. Begin with the chapter title and proceed through all sections:**
"""
    return prompt


    """
    Generate chapter with retry logic for rate limits and errors.
    
    Args:
        prompt: The prompt to send to Gemini
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated chapter text
    """
    for attempt in range(max_retries):
        try:
            print(f"  🤖 Generating chapter (attempt {attempt + 1}/{max_retries})...")
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192,
                )
            )
            
            chapter_text = response.text
            print(f"  ✅ Generated {len(chapter_text)} characters")
            
            return chapter_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ⚠️  Error: {error_msg}")
            
            # Handle rate limiting
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                wait_time = 60 * (attempt + 1)  # Exponential backoff
                print(f"  ⏱️  Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            # Other errors
            if attempt < max_retries - 1:
                print(f"  🔄 Retrying in {30 * (attempt + 1)}s...")
                time.sleep(30 * (attempt + 1))
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise


def load_progress():
    """Load progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed_chapters': [], 'chapters': []}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def save_chapter_file(chapter_num, disorder_name, content, reference_list):
    """Save individual chapter to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = f"{chapter_num:03d}_{disorder_name.lower().replace(' ', '_')}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Chapter {chapter_num}: {disorder_name}\n\n")
        f.write(content)
        f.write("\n\n## References\n\n")
        f.write(reference_list)
    
    print(f"  💾 Saved chapter to: {filepath}")
    return filepath


def generate_book():
    """Main function to generate all chapters."""
    print("=" * 80)
    print("BIOLOGICAL BASIS OF MENTAL HEALTH DISORDERS")
    print("Book Generation with Gemini 2.5 Pro (Free Tier)")
    print("=" * 80)
    
    # Load progress
    progress = load_progress()
    completed = set(progress.get('completed_chapters', []))
    
    # Get all disorder files
    disorder_files = sorted(Path('disorder_papers').glob('*.json'))
    print(f"\n📚 Found {len(disorder_files)} disorder files")
    print(f"✅ Already completed: {len(completed)} chapters")
    print(f"🔄 Remaining: {len(disorder_files) - len(completed)} chapters\n")
    
    start_time = time.time()
    
    for idx, disorder_file in enumerate(disorder_files, 1):
        disorder_key = disorder_file.stem
        
        # Skip if already completed
        if disorder_key in completed:
            print(f"[{idx}/{len(disorder_files)}] ⏭️  Skipping {disorder_key} (already completed)")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(disorder_files)}] Processing: {disorder_key}")
        print('=' * 80)
        
        try:
            # Load ALL papers with FULL content
            papers_context, reference_list, paper_count, disorder_name = load_disorder_papers(disorder_file)
            
            # Build comprehensive prompt
            prompt = build_chapter_prompt(disorder_name, papers_context, reference_list, paper_count)
            
            print(f"  📊 Prompt size: ~{len(prompt):,} characters")
            print(f"  🎯 Target: 4000-6000 words with 75+ citations")
            
            # Generate chapter
            chapter_start = time.time()
            chapter_content = generate_chapter_with_retry(prompt)
            chapter_time = time.time() - chapter_start
            
            # Save chapter file
            chapter_file = save_chapter_file(idx, disorder_name, chapter_content, reference_list)
            
            # Update progress
            chapter_info = {
                'number': idx,
                'disorder': disorder_name,
                'file': disorder_key,
                'paper_count': paper_count,
                'generation_time': chapter_time,
                'chapter_file': chapter_file,
                'timestamp': datetime.now().isoformat()
            }
            
            progress['chapters'].append(chapter_info)
            progress['completed_chapters'].append(disorder_key)
            save_progress(progress)
            
            print(f"  ⏱️  Chapter generated in {chapter_time:.1f}s")
            print(f"  ✅ Chapter {idx}/{len(disorder_files)} complete!")
            
            # Progress stats
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx - len(completed) + len(progress['chapters']))
            remaining = (len(disorder_files) - idx) * avg_time
            
            print(f"\n  📈 Progress: {len(progress['completed_chapters'])}/{len(disorder_files)} chapters")
            print(f"  ⏱️  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")
            
            # Rate limiting delay (Free tier ~10-15 RPM)
            if idx < len(disorder_files):
                print(f"  💤 Rate limit delay: {RATE_LIMIT_DELAY}s...")
                time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            print(f"  ❌ Error processing {disorder_name}: {e}")
            print(f"  💾 Progress saved. You can resume later.")
            continue
    
    # Generate final combined book
    print("\n" + "=" * 80)
    print("📖 Generating final combined book...")
    print("=" * 80)
    
    combine_chapters(progress['chapters'])
    
    total_time = time.time() - start_time
    print(f"\n🎉 Book generation complete!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Chapters generated: {len(progress['chapters'])}")
    print(f"📄 Final book: {FINAL_BOOK}")


def combine_chapters(chapters):
    """Combine all chapters into final book."""
    with open(FINAL_BOOK, 'w', encoding='utf-8') as book:
        # Title page
        book.write("# Biological Basis of Mental Health Disorders\n\n")
        book.write("*A Comprehensive Review of the Neurobiological Mechanisms Underlying DSM-5 Mental Disorders*\n\n")
        book.write(f"*Generated: {datetime.now().strftime('%B %d, %Y')}*\n\n")
        book.write("---\n\n")
        
        # Table of contents
        book.write("## Table of Contents\n\n")
        for chapter in chapters:
            book.write(f"{chapter['number']}. [{chapter['disorder']}](#chapter-{chapter['number']}-{chapter['disorder'].lower().replace(' ', '-')})\n")
        book.write("\n---\n\n")
        
        # Include all chapters
        for chapter in chapters:
            chapter_file = chapter['chapter_file']
            if os.path.exists(chapter_file):
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    book.write(content)
                    book.write("\n\n---\n\n")
        
        print(f"✅ Combined book saved to: {FINAL_BOOK}")


def main():
    """Entry point."""
    print("\n🚀 Starting book generation...")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"💾 Progress file: {PROGRESS_FILE}")
    print(f"📖 Final book: {FINAL_BOOK}")
    print(f"⏱️  Rate limit delay: {RATE_LIMIT_DELAY}s between chapters")
    print(f"💰 Cost: FREE (Gemini API Free Tier)")
    
    input("\n▶️  Press Enter to start generation (or Ctrl+C to cancel)...")
    
    try:
        generate_book()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("💾 Progress has been saved. Run script again to resume.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("💾 Progress has been saved. Run script again to resume.")


if __name__ == "__main__":
    main()

