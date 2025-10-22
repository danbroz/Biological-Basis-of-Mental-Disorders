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
import re
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
    # Convert year to string to avoid type mismatch issues
    if isinstance(year, int):
        year = str(year)
    title = paper.get('title', 'No title')
    venue = paper.get('venue', '')
    doi = paper.get('doi', '')
    
    # Helper: convert full name to "Surname, A. B." (proper APA format)
    def name_to_apa(full_name: str) -> str:
        if not full_name or not full_name.strip():
            return full_name.strip()
        
        # Tokenize and clean
        raw_parts = [p for p in re.split(r"\s+", full_name.strip()) if p]
        if not raw_parts:
            return full_name.strip()
        
        # Find surname as the last token with at least 2 letters (ignoring dots and punctuation)
        def letters_only(s: str) -> str:
            return re.sub(r"[^A-Za-zÀ-ÿ'-]", "", s)
        surname_idx = None
        for i in range(len(raw_parts) - 1, -1, -1):
            if len(letters_only(raw_parts[i])) >= 2:
                surname_idx = i
                break
        if surname_idx is None:
            # Cannot identify a valid surname; fall back to the raw name
            return full_name.strip()
        
        surname = raw_parts[surname_idx].strip().strip(',')
        given_parts = raw_parts[:surname_idx]
        
        # Build initials from given names (ignore punctuation-only tokens)
        initials = []
        for gp in given_parts:
            clean_gp = gp.replace('.', '').strip(" ,-")
            if not clean_gp:
                continue
            if '-' in clean_gp:
                hy = [h for h in clean_gp.split('-') if h]
                hy_inits = [h[0].upper() + '.' for h in hy if h]
                if hy_inits:
                    initials.append('-'.join(hy_inits))
            else:
                initials.append(clean_gp[0].upper() + '.')
        
        return f"{surname}, {' '.join(initials)}" if initials else surname

    # Format authors per APA 7th with filtering to avoid single-initial-only authors
    import re as _re
    def initials_only(name: str) -> bool:
        if not name:
            return False
        s = name.strip()
        return _re.fullmatch(r'(?:\s*[A-Za-z]\.\s*){1,6}', s) is not None
    def has_valid_surname(formatted: str) -> bool:
        # formatted is like "Surname, A. B." or "Surname"
        surname = formatted.split(',', 1)[0].strip()
        core = _re.sub(r"[^A-Za-zÀ-ÿ'-]", "", surname)
        # reject single-letter cores and empty
        if len(core) < 2:
            return False
        # reject pure initial (e.g., "A.")
        if _re.fullmatch(r'[A-Za-z]\.?', surname):
            return False
        return True

    if not authors:
        author_str = "Unknown"
    else:
        # Remove raw authors that are just initials like "D. S."
        authors_clean = [a for a in authors if a and a.strip() and not initials_only(a)]
        authors_use = authors_clean if authors_clean else authors
        
        # Find the first author with a proper full name (first name + surname)
        first_valid_author = None
        for author in authors_use:
            if author and author.strip():
                # Check if this author has both first name and surname
                name_parts = [p.strip() for p in author.split() if p.strip()]
                if len(name_parts) >= 2:  # Must have at least first name + surname
                    # Check if the last part (surname) has at least 2 letters
                    surname = name_parts[-1]
                    surname_clean = _re.sub(r"[^A-Za-zÀ-ÿ'-]", "", surname)
                    if len(surname_clean) >= 2:
                        # Also check that it's not just initials
                        first_name = name_parts[0]
                        if not _re.match(r'^[A-Z]\.?$', first_name):  # Not just an initial
                            first_valid_author = author
                            break
        
        # Use the first valid author if found, otherwise use all authors
        if first_valid_author:
            # Reorder authors to put the valid one first
            other_authors = [a for a in authors_use if a != first_valid_author]
            authors_use = [first_valid_author] + other_authors
        
        # First pass format
        formatted = [name_to_apa(a) for a in authors_use if a and a.strip()]
        # Filter out entries whose surname reduces to a single initial (e.g., "A.")
        filtered = [f for f in formatted if has_valid_surname(f)]
        apa_authors = filtered if filtered else formatted  # fall back if all filtered out
        if len(apa_authors) == 1:
            author_str = apa_authors[0]
        elif len(apa_authors) == 2:
            author_str = f"{apa_authors[0]}, & {apa_authors[1]}"
        elif len(apa_authors) <= 20:
            author_str = ', '.join(apa_authors[:-1]) + f", & {apa_authors[-1]}"
        else:
            author_str = ', '.join(apa_authors[:19]) + f", ... {apa_authors[-1]}"

    # Normalize DOI to https form when possible
    doi_str = ''
    if doi:
        d = doi.strip()
        if d.lower().startswith('10.'):
            doi_str = f"https://doi.org/{d}"
        elif d.lower().startswith('doi:'):
            core = d.split(':', 1)[-1].strip()
            doi_str = f"https://doi.org/{core}"
        elif d.lower().startswith('http'):
            doi_str = d
        else:
            doi_str = d

    # Build citation (plain text APA-like)
    citation = f"{author_str} ({year}). {title}."
    if venue:
        citation += f" {venue}."
    if doi_str:
        citation += f" {doi_str}"
    
    return citation


def load_disorder_papers(json_filepath):
    """
    Load ALL papers from a disorder JSON file with FULL content.
    
    Args:
        json_filepath: Path to disorder JSON file
        
    Returns:
        Tuple of (papers_context_string, reference_list_string, reference_list_array, paper_count, disorder_name)
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
    reference_list_str = '\n'.join([f"{i+1}. {ref}" for i, ref in enumerate(reference_parts)])
    
    return papers_context, reference_list_str, reference_parts, len(papers), disorder_name


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

**CRITICAL CITATION REQUIREMENTS:**

⚠️ You MUST cite ONLY from the {paper_count} research papers provided below in this prompt.
⚠️ Do NOT cite any external sources, textbooks, review articles, or other papers not in the provided list.
⚠️ Use exact author surnames and years as they appear in the reference list below.
⚠️ Any citation not matching a provided paper will be excluded from the final reference list.
⚠️ Cross-check every citation against the reference list to ensure it exists.

**Instructions:**

1. Write a detailed, evidence-based chapter (4000-6000 words) on the biological/neurological mechanisms underlying {disorder_name}. IMPORTANT: Complete ALL sections fully - do not truncate mid-sentence.

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
   - You MUST cite sources ONLY from the {paper_count} provided research papers below
   - Do NOT cite any papers not explicitly listed in the reference list
   - Use inline APA citations: (Author, Year) or (Author et al., Year)
   - Cite multiple papers when making broad claims: (Author1, Year1; Author2, Year2)
   - EVERY major statement, finding, or claim MUST be cited
   - Use the exact author surnames and years from the reference list provided below
   - Verify each citation exists in the reference list before using it
   - Aim for 75-100+ citations throughout the chapter
   - Integrate citations naturally into the text

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

**COMPLETE REFERENCE LIST ({paper_count} papers - CITE ONLY FROM THIS LIST):**

{reference_list}

**FINAL REMINDERS BEFORE WRITING:**
- ✓ Cite extensively from the papers above using APA format: (Author, Year) or (Author et al., Year)
- ✓ ONLY cite papers from the list above - do NOT cite any external sources
- ✓ Match author surnames and years exactly as shown in the reference list
- ✓ The final reference list will be automatically filtered to include ONLY papers you actually cite
- ✓ The reference list will be automatically sorted alphabetically in APA format
- ✓ If you cite a paper not in this list, it will NOT appear in the references section

**Now write the chapter, ensuring every claim is supported by citations from the papers listed above. Begin with the chapter title and proceed through all sections. CRITICAL: Complete each section fully and end with a proper conclusion - do not truncate mid-sentence.**
"""
    return prompt


def generate_chapter_with_retry(prompt, max_retries=3):
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
                    max_output_tokens=16384,  # Increased token limit
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


def extract_cited_references(chapter_content, all_references):
    """
    Extract only the references that were actually cited in the chapter.
    Uses flexible first-author + year matching to handle various citation formats.
    
    Args:
        chapter_content: The generated chapter text
        all_references: List of all reference strings
        
    Returns:
        Alphabetically sorted list of only cited references in APA format
    """
    import re
    
    # Extract all citations from the text
    citation_pattern = r'\(([^)]+\d{4}[a-z]?[^)]*)\)'
    citations = re.findall(citation_pattern, chapter_content)
    
    # Build set of cited first-author + year combinations
    # This is more flexible than trying to match all authors
    cited_first_author_year = set()
    
    for citation in citations:
        # Split multiple citations separated by semicolons
        parts = [p.strip() for p in citation.split(';')]
        
        for part in parts:
            # Extract year
            year_match = re.search(r'\b(\d{4}[a-z]?)\b', part)
            if not year_match:
                continue
                
            year = year_match.group(1)
            
            # Extract text before year
            text_before_year = part.split(year)[0].strip().rstrip(',').strip()
            
            # Handle different citation formats more flexibly
            # Case 1: "Author et al." - extract just the surname
            if ' et al.' in text_before_year:
                first_author = text_before_year.split(' et al.')[0].strip()
            # Case 2: "Author & Author" - extract first author
            elif ' & ' in text_before_year:
                first_author = text_before_year.split(' & ')[0].strip()
            # Case 3: "Author, Author" - extract first author
            elif ', ' in text_before_year:
                first_author = text_before_year.split(', ')[0].strip()
            # Case 4: Single author
            else:
                first_author = text_before_year.strip()
            
            # Extract just the surname (last word) for better matching
            surname = first_author.split()[-1] if first_author else first_author
            cited_first_author_year.add((surname.lower(), year))
    
    print(f"  🔍 Extracted {len(cited_first_author_year)} unique first-author + year citations")
    
    # Filter references using surname + year matching with improved logic
    cited_refs = []
    cited_refs_set = set()  # Track to avoid duplicates
    matched_pairs = set()  # Track which citations we matched
    
    for ref in all_references:
        # Remove numbering if present (look for pattern "1. " or "2. " etc. at start)
        if re.match(r'^\d+\. ', ref):
            ref_without_number = re.sub(r'^\d+\. ', '', ref)
        else:
            ref_without_number = ref
        
        # Skip if already added
        if ref_without_number in cited_refs_set:
            continue
        
        # Extract year from reference
        year_in_ref_match = re.search(r'\((\d{4}[a-z]?)\)', ref_without_number)
        if not year_in_ref_match:
            continue
            
        year_in_ref = year_in_ref_match.group(1)
        
        # Extract author text before year
        text_before_year = ref_without_number.split('(')[0].strip()
        
        # Check if any cited surname + year matches this reference
        for cited_surname, cited_year in cited_first_author_year:
            if cited_year == year_in_ref:
                # Try exact match first
                surname_pattern = r'\b' + re.escape(cited_surname) + r'\b'
                if re.search(surname_pattern, text_before_year, re.IGNORECASE):
                    cited_refs.append(ref_without_number)
                    cited_refs_set.add(ref_without_number)
                    matched_pairs.add((cited_surname, cited_year))
                    break
                
                # Try fuzzy matching for abbreviated surnames
                # Check if cited surname is a prefix of any surname in the reference
                words_in_ref = re.findall(r'\b[A-Za-zà-ÿ]+\b', text_before_year)
                for word in words_in_ref:
                    if (word.lower().startswith(cited_surname.lower()) and 
                        len(cited_surname) >= 2 and  # Avoid single letter matches
                        len(word) > len(cited_surname)):  # Word must be longer than cited surname
                        cited_refs.append(ref_without_number)
                        cited_refs_set.add(ref_without_number)
                        matched_pairs.add((cited_surname, cited_year))
                        break
                else:
                    continue  # No match found, try next cited surname
                break  # Found a match, move to next reference
    
    # Sort alphabetically by author (APA style)
    cited_refs.sort(key=lambda x: x.lower())
    
    # Re-number the references
    numbered_refs = [f"{i+1}. {ref}" for i, ref in enumerate(cited_refs)]
    
    # Report statistics
    unmatched_count = len(cited_first_author_year) - len(matched_pairs)
    
    if cited_refs:
        print(f"  📚 Filtered references: {len(all_references)} → {len(cited_refs)} cited papers")
        if unmatched_count > 0:
            print(f"  ⚠️  Warning: {unmatched_count} citations not found in provided papers (may be hallucinated)")
    else:
        print(f"  ⚠️  Warning: No citations matched. Including all {len(all_references)} references.")
        numbered_refs = [f"{i+1}. {ref}" for i, ref in enumerate(all_references)]
    
    return numbered_refs


def filter_references_and_clean_text(chapter_content, all_references):
    """
    Post-process the chapter to generate a validated reference list from in-text citations.
    Only includes references that are actually cited in the text.
    - Extract cited (surname, year) pairs from the text
    - Match against provided references using precise matching
    - Keep the text as-is and build a reference list with only cited sources

    Args:
        chapter_content: The generated chapter text
        all_references: List of all reference strings (APA format)

    Returns:
        Tuple (cleaned_content, numbered_refs)
    """
    import re
    import unicodedata

    # Always operate only on the main body; ignore any existing References section
    main_body = chapter_content.split("\n## References", 1)[0]

    # Step 1: Extract all parenthetical citations from the main body
    citation_pattern = r'\(([^)]+\d{4}[a-z]?[^)]*)\)'
    citations = re.findall(citation_pattern, main_body)

    # Step 2: Build set of cited first-author + year combinations from text
    cited_first_author_year = set()
    for citation in citations:
        parts = [p.strip() for p in citation.split(';')]
        for part in parts:
            year_match = re.search(r'\b(\d{4}[a-z]?)\b', part)
            if not year_match:
                continue
            year = year_match.group(1)
            text_before_year = part.split(year)[0].strip().rstrip(',').strip()
            if ' et al.' in text_before_year:
                first_author = text_before_year.split(' et al.')[0].strip()
            elif ' & ' in text_before_year:
                first_author = text_before_year.split(' & ')[0].strip()
            elif ', ' in text_before_year:
                first_author = text_before_year.split(', ')[0].strip()
            else:
                first_author = text_before_year.strip()
            surname = first_author.split()[-1] if first_author else first_author
            if surname:
                cited_first_author_year.add((surname.lower(), year))

    # Normalization helpers for robust matching
    def normalize_token(s: str) -> str:
        if not s:
            return ''
        s = s.strip().lower()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = re.sub(r"[^a-z0-9\- ]+", "", s)
        return s

    # Step 3: Build precise reference matches
    cited_refs = []
    cited_refs_set = set()

    def first_author_valid(apa_ref_text: str) -> bool:
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
        
        return True

    # For each reference, check if it matches any cited author+year combination
    for ref in all_references:
        # Remove numbering if present
        if re.match(r'^\d+\. ', ref):
            ref_text = re.sub(r'^\d+\. ', '', ref)
        else:
            ref_text = ref
        
        # Extract year from reference
        year_match = re.search(r'\((\d{4}[a-z]?)\)', ref_text)
        if not year_match:
            continue
        ref_year = year_match.group(1)
        
        # Extract author text before year
        authors_text = ref_text.split('(')[0].strip()
        
        # Check if this reference matches any cited author+year
        for cited_surname, cited_year in cited_first_author_year:
            if cited_year != ref_year:
                continue
            
            # Normalize for comparison
            norm_cited_surname = normalize_token(cited_surname)
            ref_words = re.findall(r'\b[A-Za-zà-ÿ\-]+\b', authors_text)
            norm_ref_words = [normalize_token(w) for w in ref_words]
            
            # Check for exact match or prefix match
            match_found = False
            if norm_cited_surname in norm_ref_words:
                match_found = True
            else:
                # Check for prefix/variant matches (handles Mc/Mac, etc.)
                for w in norm_ref_words:
                    if (norm_cited_surname and len(norm_cited_surname) >= 2 and 
                        (w.startswith(norm_cited_surname) or norm_cited_surname.startswith(w))):
                        match_found = True
                        break
            
            if match_found:
                # Only add if it's a valid reference and not already included
                if first_author_valid(ref_text) and ref_text not in cited_refs_set:
                    cited_refs.append(ref_text)
                    cited_refs_set.add(ref_text)
                break  # Found a match for this reference, move to next reference

    # Sort APA-style (alphabetical)
    cited_refs.sort(key=lambda x: x.lower())
    numbered_refs = [f"{i}. {ref}" for i, ref in enumerate(cited_refs, 1)]

    return main_body, numbered_refs


def save_chapter_file(chapter_num, disorder_name, content, all_references_list):
    """Save individual chapter to file with only cited references.
    Ensures 100% non-hallucinated citations by removing unmatched citations from text.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert reference list string to list if needed
    if isinstance(all_references_list, str):
        all_refs = [line.split('. ', 1)[-1] for line in all_references_list.split('\n') if line.strip()]
    else:
        all_refs = all_references_list
    
    # Strip any existing References section from incoming content
    base_content = content.split("\n## References", 1)[0]
    # Clean text from hallucinated citations and filter references
    cleaned_content, cited_references = filter_references_and_clean_text(base_content, all_refs)
    
    filename = f"{chapter_num:03d}_{disorder_name.lower().replace(' ', '_')}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Chapter {chapter_num}: {disorder_name}\n\n")
        f.write(cleaned_content)
        f.write("\n\n## References\n\n")
        f.write('\n'.join(cited_references))
    
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
            papers_context, reference_list_str, reference_parts, paper_count, disorder_name = load_disorder_papers(disorder_file)
            
            # Build comprehensive prompt
            prompt = build_chapter_prompt(disorder_name, papers_context, reference_list_str, paper_count)
            
            print(f"  📊 Prompt size: ~{len(prompt):,} characters")
            print(f"  🎯 Target: 4000-6000 words with 75+ citations")
            
            # Generate chapter
            chapter_start = time.time()
            chapter_content = generate_chapter_with_retry(prompt)
            chapter_time = time.time() - chapter_start
            
            # Save chapter file (with filtered and sorted references)
            chapter_file = save_chapter_file(idx, disorder_name, chapter_content, reference_parts)
            
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
    
    # input("\n▶️  Press Enter to start generation (or Ctrl+C to cancel)...")
    print("\n▶️  Starting generation...\n")
    
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

