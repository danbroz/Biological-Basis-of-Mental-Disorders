#!/usr/bin/env python3
"""
Test script for OpenAlex RAG with Intellectual Developmental Disorder query
"""

from openalex_rag import OpenAlexRAG

def main():
    # Initialize RAG (using OpenAlex API by default)
    print("🚀 Initializing OpenAlex RAG...")
    try:
        rag = OpenAlexRAG(
            email="dan@danbroz.com", 
            k=500,
            use_mongodb=False  # Use OpenAlex API instead of MongoDB
        )
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    # Query from DSM-5 Intellectual Developmental Disorder criteria
    query = """Intellectual Developmental Disorder (Intellectual Disability)
Deficits in intellectual functions, such as reasoning, problem solving, planning, abstract thinking, judgment, academic learning, and learning from experience, confirmed by both clinical assessment and individualized, standardized intelligence testing.
Deficits in adaptive functioning that result in failure to meet developmental and sociocultural standards for personal independence and social responsibility. Without ongoing support, the adaptive deficits limit functioning in one or more activities of daily life, such as communication, social participation, and independent living, across multiple environments, such as home, school, work, and community.
Onset of intellectual and adaptive deficits during the developmental period."""
    
    print(f"\n🔍 Testing Query:")
    print("-" * 80)
    print(query)
    print("-" * 80)
    
    # Search for papers
    papers = rag.search_papers(query, k=500)
    
    if papers:
        # Display formatted results
        print(f"\n{rag.format_results(papers)}")
        
        # Save to JSON
        filename = rag.save_to_json(papers, "intellectual_disability_papers.json")
        print(f"\n💾 Results saved to: {filename}")
    else:
        print("❌ No papers found")
    
    # Close connections
    rag.close()
    print("\n✅ Test completed")

if __name__ == "__main__":
    main()

