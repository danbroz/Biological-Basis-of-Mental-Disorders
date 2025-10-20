#!/usr/bin/env python3
"""
Remove soft line breaks within diagnostic criteria while preserving hard breaks.
"""

import re

def is_criterion_marker(line):
    """Check if line starts a new criterion"""
    stripped = line.strip()
    if not stripped:
        return False
    
    # Main criteria: A., B., C., etc.
    if re.match(r'^[A-Z]\.\s+\S', stripped):
        return True
    # Numbered: 1., 2., 3., etc.
    if re.match(r'^\d+\.\s+\S', stripped):
        return True
    # Lettered sub: a., b., c., etc.
    if re.match(r'^[a-z]\.\s+\S', stripped):
        return True
    # Roman numerals
    if re.match(r'^(i{1,3}|iv|v|vi{1,3}|ix|x)\.\s+\S', stripped):
        return True
    
    return False

def is_section_marker(line):
    """Check if line is a section marker"""
    stripped = line.strip()
    if not stripped:
        return False
    
    # ICD codes
    if len(stripped) < 15 and re.match(r'^[FGRE]\d+\.', stripped):
        return True
    
    # Special keywords
    if stripped in ["Diagnostic Criteria"] or stripped.startswith(("Note:", "Specify", "Coding", "TABLE", "Subtypes", "Specifiers", "Recording Procedures")):
        return True
    
    return False

def fix_line_breaks(input_file, output_file):
    """Remove soft line breaks while preserving structure."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    result = []
    i = 0
    in_disorder = False
    
    while i < len(lines):
        current = lines[i].strip()
        
        # Handle empty lines
        if not current:
            # Add blank line to result if appropriate
            if result and result[-1] != '':
                result.append('')
            i += 1
            in_disorder = False
            continue
        
        # Section markers go on their own line
        if is_section_marker(lines[i]):
            result.append(current)
            i += 1
            in_disorder = True
            continue
        
        # Criterion markers - collect all continuation lines
        if is_criterion_marker(lines[i]):
            collected = [current]
            i += 1
            
            # Collect continuation lines (skip blank lines within criterion)
            while i < len(lines):
                next_line = lines[i].strip()
                
                # Skip blank lines within a criterion
                if not next_line:
                    i += 1
                    continue
                
                # Stop at next criterion/section marker  
                if is_criterion_marker(lines[i]) or is_section_marker(lines[i]):
                    break
                
                # Stop at what looks like a disorder name (after we were in criteria)
                # Disorder names typically start with capital letter and aren't too long
                if in_disorder and len(next_line) < 100 and not next_line[0].islower():
                    # Check if next line after this is "Diagnostic Criteria"
                    if i + 1 < len(lines) and lines[i + 1].strip() == "Diagnostic Criteria":
                        break
                
                collected.append(next_line)
                i += 1
            
            result.append(' '.join(collected))
            continue
        
        # Regular text - could be disorder name or introductory paragraph
        # Collect until blank line or marker
        collected = [current]
        i += 1
        
        while i < len(lines):
            next_line = lines[i].strip()
            
            # Stop at blank line, criterion, or section marker
            if not next_line or is_criterion_marker(lines[i]) or is_section_marker(lines[i]):
                break
            
            collected.append(next_line)
            i += 1
        
        result.append(' '.join(collected))
        # Don't set in_disorder here as regular text could be before "Diagnostic Criteria"
    
    # Clean up: ensure exactly one blank line between disorders
    cleaned = []
    prev_blank = False
    for line in result:
        if line == '':
            if not prev_blank:
                cleaned.append(line)
                prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
    
    return len(cleaned)

if __name__ == "__main__":
    input_file = "/home/dan/Biological Basis DSM Disorders/mental-disorder-and-criteria.txt"
    output_file = "/home/dan/Biological Basis DSM Disorders/mental-disorder-and-criteria.txt"
    
    print(f"Fixing line breaks in: {input_file}")
    lines = fix_line_breaks(input_file, output_file)
    print(f"Processed {lines} lines")
    print("Done! Soft line breaks have been removed.")
