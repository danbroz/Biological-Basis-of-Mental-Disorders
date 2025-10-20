#!/usr/bin/env python3
"""
OpenAlex RAG Implementation with MongoDB and Model Context Protocol (MCP)
Uses trained embeddings for similarity search, then retrieves paper details from MongoDB.
Implements MCP for structured AI assistant integration.

Features:
- Similarity search using Faiss index
- MongoDB integration for fast metadata retrieval
- Model Context Protocol (MCP) server implementation
- No external API calls - all data served from local MongoDB
"""

import requests
import json
import re
import os
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import asyncio
from urllib.parse import quote

# Load environment variables
load_dotenv()

def convert_abstract_inverted_index(inverted_index: dict) -> str:
    """
    Convert OpenAlex abstract inverted index to readable text.
    
    Args:
        inverted_index: Dictionary mapping words to their positions in the abstract
        
    Returns:
        Reconstructed abstract text
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return "No abstract available"
    
    try:
        # Find max position to initialize list
        max_pos = max(max(positions) for positions in inverted_index.values() if positions)
        words = [''] * (max_pos + 1)
        
        # Place words at their positions
        for word, positions in inverted_index.items():
            for pos in positions:
                if pos <= max_pos:
                    words[pos] = word
        
        # Join and clean up
        return ' '.join(word for word in words if word)
    except (ValueError, TypeError) as e:
        print(f"⚠️  Error converting inverted index: {e}")
        return "No abstract available"

# MCP imports (using standard Python asyncio)
try:
    from mcp.server import Server
    from mcp.types import (
        Resource, Tool, TextContent, ImageContent, EmbeddedResource,
        ListResourcesResult, ReadResourceResult, CallToolResult
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not available. Install with: pip install mcp")

@dataclass
class Paper:
    """Data class for academic paper information"""
    title: str
    abstract: str
    authors: List[str]
    year: int
    download_url: str
    openalex_id: str
    doi: str = None
    venue: str = None
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

class OpenAlexRAG:
    """RAG implementation using similarity search and OpenAlex API"""
    
    def __init__(self, email: str = "dan@danbroz.com", k: int = 475, 
                 embeddings_dir: str = "all",
                 index_dir: str = "all",
                 model_name: str = "sentence-transformers/LaBSE",
                 use_mongodb: bool = False,
                 mongodb_uri: str = None,
                 mongodb_db: str = None,
                 mongodb_collection: str = None):
        self.email = email
        self.k = k
        self.use_mongodb = use_mongodb
        
        # MongoDB configuration (optional)
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.mongodb_db = mongodb_db or os.getenv('MONGODB_DB', 'openalex')
        self.mongodb_collection = mongodb_collection or os.getenv('MONGODB_COLLECTION', 'papers_all')
        
        # Embedding and index paths
        self.embeddings_dir = embeddings_dir
        self.index_dir = index_dir
        self.model_name = model_name
        
        # Initialize components
        self.model = None
        self.index = None
        self.ids = None
        self.mongo_client = None
        self.collection = None
        
        # OpenAlex API configuration
        self.api_base_url = "https://api.openalex.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'mailto:{self.email}',
            'Accept': 'application/json'
        })
        
        # Check if trained files exist
        self._check_trained_files()
        
        # Load the trained model and index
        self._load_model_and_index()
        
        # Connect to MongoDB (optional)
        if self.use_mongodb:
            self._connect_mongodb()
    
    def _check_trained_files(self):
        """Check if required trained files exist"""
        required_files = [
            os.path.join(self.index_dir, "index.faiss"),
            os.path.join(self.index_dir, "openalex_ids.npy")
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required trained files:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            print("\nPlease run the following commands to create the required files:")
            print("1. python3 build-both.py  # Generate embeddings and populate MongoDB")
            print("2. python3 train-both.py  # Create Faiss indices")
            raise FileNotFoundError("Required trained files not found")
        
        print("✅ All required trained files found")
        
    def _load_model_and_index(self):
        """Load the trained model and Faiss index"""
        print("🔄 Loading trained model and index...")
        
        # Load the sentence transformer model
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load the Faiss index
        try:
            index_path = os.path.join(self.index_dir, "index.faiss")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                print(f"✅ Loaded Faiss index from {index_path}")
            else:
                print(f"❌ Index file not found: {index_path}")
                raise FileNotFoundError(f"Index file not found: {index_path}")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            raise
        
        # Load the IDs
        try:
            ids_path = os.path.join(self.index_dir, "openalex_ids.npy")
            if os.path.exists(ids_path):
                self.ids = np.load(ids_path)
                print(f"✅ Loaded {len(self.ids)} IDs from {ids_path}")
            else:
                print(f"❌ IDs file not found: {ids_path}")
                raise FileNotFoundError(f"IDs file not found: {ids_path}")
        except Exception as e:
            print(f"❌ Error loading IDs: {e}")
            raise
        
        print(f"🎉 Successfully loaded model and index with {len(self.ids)} papers")
    
    def _connect_mongodb(self):
        """Connect to MongoDB"""
        print("🔄 Connecting to MongoDB...")
        try:
            self.mongo_client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            db = self.mongo_client[self.mongodb_db]
            self.collection = db[self.mongodb_collection]
            
            # Test connection
            self.mongo_client.server_info()
            doc_count = self.collection.count_documents({})
            print(f"✅ MongoDB connected: {self.mongodb_db}.{self.mongodb_collection}")
            print(f"📊 Total papers in MongoDB: {doc_count:,}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("Please ensure MongoDB is running and MONGODB_URI is set correctly")
            raise
    
    def _fetch_paper_from_api(self, openalex_id: str) -> Optional[Paper]:
        """
        Fetch a paper from OpenAlex API by ID
        
        Args:
            openalex_id: OpenAlex ID (e.g., 'https://openalex.org/W1234567890' or 'W1234567890')
            
        Returns:
            Paper object or None if not found
        """
        try:
            # Ensure we have the full URL format
            if not openalex_id.startswith('https://'):
                openalex_id = f'https://openalex.org/{openalex_id}'
            
            # Extract just the ID part for the API call
            work_id = openalex_id.split('/')[-1]
            url = f"{self.api_base_url}/works/{work_id}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Extract authors
            authors = []
            if 'authorships' in data and data['authorships']:
                for authorship in data['authorships']:
                    if authorship.get('author') and authorship['author'].get('display_name'):
                        authors.append(authorship['author']['display_name'])
            
            # Convert abstract inverted index to text
            abstract_text = "No abstract available"
            if 'abstract_inverted_index' in data and data['abstract_inverted_index']:
                abstract_text = convert_abstract_inverted_index(data['abstract_inverted_index'])
            
            # Extract other fields
            title = data.get('title', 'No Title Available')
            year = data.get('publication_year', 0) or 0
            doi = data.get('doi', '')
            
            # Get venue/journal name
            venue = ''
            if data.get('primary_location'):
                source = data['primary_location'].get('source')
                if source:
                    venue = source.get('display_name', '')
            
            # Get download URL (prefer DOI, then open access, then landing page)
            download_url = ''
            if doi:
                download_url = doi
            elif data.get('open_access') and data['open_access'].get('oa_url'):
                download_url = data['open_access']['oa_url']
            elif data.get('primary_location') and data['primary_location'].get('landing_page_url'):
                download_url = data['primary_location']['landing_page_url']
            
            return Paper(
                title=title,
                abstract=abstract_text,
                authors=authors,
                year=year,
                download_url=download_url,
                openalex_id=openalex_id,
                doi=doi,
                venue=venue
            )
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API request failed for {openalex_id}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error parsing API response for {openalex_id}: {e}")
            return None
    
    def _fetch_papers_batch_from_api(self, openalex_ids: List[str]) -> List[Paper]:
        """
        Fetch multiple papers from OpenAlex API in batch
        
        Args:
            openalex_ids: List of OpenAlex IDs
            
        Returns:
            List of Paper objects
        """
        if not openalex_ids:
            return []
        
        papers = []
        
        # OpenAlex API supports batch queries with OR filter
        # Format: filter=openalex_id:W1|W2|W3
        # But we need to handle the full URL format
        work_ids = []
        for oa_id in openalex_ids:
            if oa_id.startswith('https://'):
                work_ids.append(oa_id.split('/')[-1])
            else:
                work_ids.append(oa_id)
        
        # API has limits, so batch in groups of 50
        batch_size = 50
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i + batch_size]
            filter_str = '|'.join(batch)
            
            try:
                url = f"{self.api_base_url}/works"
                params = {
                    'filter': f'openalex_id:{filter_str}',
                    'per-page': batch_size
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Process results
                if 'results' in data:
                    for work in data['results']:
                        # Extract data similar to single fetch
                        authors = []
                        if 'authorships' in work and work['authorships']:
                            for authorship in work['authorships']:
                                if authorship.get('author') and authorship['author'].get('display_name'):
                                    authors.append(authorship['author']['display_name'])
                        
                        abstract_text = "No abstract available"
                        if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                            abstract_text = convert_abstract_inverted_index(work['abstract_inverted_index'])
                        
                        title = work.get('title', 'No Title Available')
                        year = work.get('publication_year', 0) or 0
                        doi = work.get('doi', '')
                        openalex_id = work.get('id', '')
                        
                        venue = ''
                        if work.get('primary_location'):
                            source = work['primary_location'].get('source')
                            if source:
                                venue = source.get('display_name', '')
                        
                        download_url = ''
                        if doi:
                            download_url = doi
                        elif work.get('open_access') and work['open_access'].get('oa_url'):
                            download_url = work['open_access']['oa_url']
                        elif work.get('primary_location') and work['primary_location'].get('landing_page_url'):
                            download_url = work['primary_location']['landing_page_url']
                        
                        papers.append(Paper(
                            title=title,
                            abstract=abstract_text,
                            authors=authors,
                            year=year,
                            download_url=download_url,
                            openalex_id=openalex_id,
                            doi=doi,
                            venue=venue
                        ))
                
                # Be polite - small delay between batches
                if i + batch_size < len(work_ids):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"⚠️  Batch API request failed: {e}")
                continue
        
        return papers
    
    def search_papers(self, query: str, k: Optional[int] = None) -> List[Paper]:
        """
        Search for papers using similarity search on trained embeddings
        
        Args:
            query: Search query string
            k: Number of results to return (defaults to self.k)
            
        Returns:
            List of Paper objects
        """
        if k is None:
            k = self.k
            
        print(f"🔍 Searching embeddings for: '{query}'")
        print(f"📊 Requesting top {k} results")
        
        # Generate embedding for the query
        try:
            query_embedding = self.model.encode([query])
            print(f"✅ Generated query embedding")
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            return []
        
        # Perform similarity search
        try:
            # Search the index
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            print(f"✅ Found {len(indices[0])} similar papers")
        except Exception as e:
            print(f"❌ Error searching index: {e}")
            return []
        
        # Get the OpenAlex IDs of the most similar papers
        similar_ids = []
        for idx in indices[0]:
            if idx < len(self.ids):
                openalex_id = self.ids[idx]
                similar_ids.append(str(openalex_id))
        
        print(f"📄 Retrieved {len(similar_ids)} OpenAlex IDs")
        
        # Fetch paper details from API or MongoDB
        papers = []
        
        if self.use_mongodb and self.collection:
            # Fetch from MongoDB (legacy mode)
            try:
                # Batch query MongoDB
                mongo_docs = self.collection.find(
                    {'openalex_id': {'$in': similar_ids}},
                    {'_id': 0}  # Exclude MongoDB _id field
                )
                
                # Create a mapping for quick lookup
                doc_map = {doc['openalex_id']: doc for doc in mongo_docs}
                
                # Maintain order from similarity search
                for openalex_id in similar_ids:
                    if openalex_id in doc_map:
                        doc = doc_map[openalex_id]
                        paper = self._parse_mongo_doc(doc)
                        if paper:
                            papers.append(paper)
                
                print(f"✅ Retrieved {len(papers)} papers from MongoDB")
                
            except Exception as e:
                print(f"❌ Error fetching from MongoDB: {e}")
                return []
        else:
            # Fetch from OpenAlex API (default mode)
            try:
                print(f"🌐 Fetching papers from OpenAlex API...")
                papers = self._fetch_papers_batch_from_api(similar_ids)
                
                # Maintain order from similarity search
                id_to_paper = {paper.openalex_id: paper for paper in papers}
                ordered_papers = []
                for openalex_id in similar_ids:
                    if openalex_id in id_to_paper:
                        ordered_papers.append(id_to_paper[openalex_id])
                papers = ordered_papers
                
                print(f"✅ Retrieved {len(papers)} papers from OpenAlex API")
                
            except Exception as e:
                print(f"❌ Error fetching from API: {e}")
                return []
        
        print(f"🎉 Successfully retrieved {len(papers)} papers")
        return papers[:k]
    
    def _parse_mongo_doc(self, doc: Dict[str, Any]) -> Optional[Paper]:
        """Parse a MongoDB document into a Paper object"""
        try:
            return Paper(
                title=doc.get('title', 'No Title Available'),
                abstract=doc.get('abstract', 'No abstract available'),
                authors=doc.get('authors', []),
                year=doc.get('year', 0) or 0,
                download_url=doc.get('download_url', ''),
                openalex_id=doc.get('openalex_id', ''),
                doi=doc.get('doi', ''),
                venue=doc.get('venue', '')
            )
        except Exception as e:
            print(f"⚠️  Error parsing MongoDB document: {e}")
            return None
    
    def get_paper_by_id(self, openalex_id: str) -> Optional[Paper]:
        """Get a specific paper by OpenAlex ID"""
        if self.use_mongodb and self.collection:
            # Fetch from MongoDB
            try:
                doc = self.collection.find_one({'openalex_id': openalex_id}, {'_id': 0})
                if doc:
                    return self._parse_mongo_doc(doc)
                return None
            except Exception as e:
                print(f"❌ Error fetching paper {openalex_id}: {e}")
                return None
        else:
            # Fetch from API
            return self._fetch_paper_from_api(openalex_id)
    
    def get_papers_by_author(self, author_name: str, limit: int = 100) -> List[Paper]:
        """Get papers by a specific author"""
        if self.use_mongodb and self.collection:
            # Fetch from MongoDB
            try:
                docs = self.collection.find(
                    {'authors': {'$regex': author_name, '$options': 'i'}},
                    {'_id': 0}
                ).limit(limit)
                
                papers = []
                for doc in docs:
                    paper = self._parse_mongo_doc(doc)
                    if paper:
                        papers.append(paper)
                
                print(f"✅ Found {len(papers)} papers by '{author_name}'")
                return papers
            except Exception as e:
                print(f"❌ Error searching by author: {e}")
                return []
        else:
            # Fetch from API
            try:
                url = f"{self.api_base_url}/works"
                params = {
                    'filter': f'author.search:{author_name}',
                    'per-page': min(limit, 200)
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                papers = []
                if 'results' in data:
                    for work in data['results']:
                        authors = []
                        if 'authorships' in work and work['authorships']:
                            for authorship in work['authorships']:
                                if authorship.get('author') and authorship['author'].get('display_name'):
                                    authors.append(authorship['author']['display_name'])
                        
                        abstract_text = "No abstract available"
                        if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                            abstract_text = convert_abstract_inverted_index(work['abstract_inverted_index'])
                        
                        papers.append(Paper(
                            title=work.get('title', 'No Title Available'),
                            abstract=abstract_text,
                            authors=authors,
                            year=work.get('publication_year', 0) or 0,
                            download_url=work.get('doi', '') or work.get('primary_location', {}).get('landing_page_url', ''),
                            openalex_id=work.get('id', ''),
                            doi=work.get('doi', ''),
                            venue=work.get('primary_location', {}).get('source', {}).get('display_name', '') if work.get('primary_location') else ''
                        ))
                
                print(f"✅ Found {len(papers)} papers by '{author_name}'")
                return papers[:limit]
            except Exception as e:
                print(f"❌ Error searching by author via API: {e}")
                return []
    
    def get_papers_by_year_range(self, start_year: int, end_year: int, limit: int = 1000) -> List[Paper]:
        """Get papers within a year range"""
        if self.use_mongodb and self.collection:
            # Fetch from MongoDB
            try:
                docs = self.collection.find(
                    {'year': {'$gte': start_year, '$lte': end_year}},
                    {'_id': 0}
                ).limit(limit)
                
                papers = []
                for doc in docs:
                    paper = self._parse_mongo_doc(doc)
                    if paper:
                        papers.append(paper)
                
                print(f"✅ Found {len(papers)} papers from {start_year}-{end_year}")
                return papers
            except Exception as e:
                print(f"❌ Error searching by year range: {e}")
                return []
        else:
            # Fetch from API
            try:
                url = f"{self.api_base_url}/works"
                params = {
                    'filter': f'publication_year:{start_year}-{end_year}',
                    'per-page': min(limit, 200)
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                papers = []
                if 'results' in data:
                    for work in data['results']:
                        authors = []
                        if 'authorships' in work and work['authorships']:
                            for authorship in work['authorships']:
                                if authorship.get('author') and authorship['author'].get('display_name'):
                                    authors.append(authorship['author']['display_name'])
                        
                        abstract_text = "No abstract available"
                        if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                            abstract_text = convert_abstract_inverted_index(work['abstract_inverted_index'])
                        
                        papers.append(Paper(
                            title=work.get('title', 'No Title Available'),
                            abstract=abstract_text,
                            authors=authors,
                            year=work.get('publication_year', 0) or 0,
                            download_url=work.get('doi', '') or work.get('primary_location', {}).get('landing_page_url', ''),
                            openalex_id=work.get('id', ''),
                            doi=work.get('doi', ''),
                            venue=work.get('primary_location', {}).get('source', {}).get('display_name', '') if work.get('primary_location') else ''
                        ))
                
                print(f"✅ Found {len(papers)} papers from {start_year}-{end_year}")
                return papers[:limit]
            except Exception as e:
                print(f"❌ Error searching by year range via API: {e}")
                return []
    
    def format_results(self, papers: List[Paper]) -> str:
        """Format the results for display"""
        if not papers:
            return "❌ No papers found."
        
        output = []
        output.append(f"📚 OpenAlex RAG Results ({len(papers)} papers)")
        output.append("=" * 80)
        
        for i, paper in enumerate(papers, 1):
            output.append(f"\n📄 Paper #{i}")
            output.append(f"Title: {paper.title}")
            output.append(f"Authors: {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}")
            output.append(f"Year: {paper.year}")
            output.append(f"Venue: {paper.venue or 'Unknown'}")
            output.append(f"DOI: {paper.doi or 'Not available'}")
            
            # Truncate abstract if too long
            abstract = paper.abstract
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            output.append(f"Abstract: {abstract}")
            
            output.append(f"Download: {paper.download_url}")
            output.append("-" * 60)
        
        return "\n".join(output)
    
    def save_to_json(self, papers: List[Paper], filename: str = None) -> str:
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openalex_rag_results_{timestamp}.json"
        
        # Convert papers to dictionaries
        papers_data = [paper.to_dict() for paper in papers]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def close(self):
        """Close connections"""
        if self.mongo_client:
            self.mongo_client.close()
            print("✅ MongoDB connection closed")
        if self.session:
            self.session.close()
            print("✅ API session closed")


class OpenAlexMCPServer:
    """Model Context Protocol (MCP) Server for OpenAlex RAG"""
    
    def __init__(self, rag: OpenAlexRAG):
        self.rag = rag
        if not MCP_AVAILABLE:
            raise ImportError("MCP not available. Install with: pip install mcp")
        
        self.server = Server("openalex-rag")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP handlers"""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available RAG resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="openalex://search",
                        name="OpenAlex Semantic Search",
                        description="Search academic papers using semantic similarity",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="openalex://paper/{id}",
                        name="Get Paper by ID",
                        description="Retrieve a specific paper by OpenAlex ID",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="openalex://author/{name}",
                        name="Search by Author",
                        description="Find papers by author name",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="openalex://year-range/{start}/{end}",
                        name="Search by Year Range",
                        description="Find papers within a year range",
                        mimeType="application/json"
                    )
                ]
            )
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource"""
            if uri.startswith("openalex://search?q="):
                query = uri.split("q=")[1]
                papers = self.rag.search_papers(query)
                content = json.dumps([p.to_dict() for p in papers], indent=2)
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=content)]
                )
            
            elif uri.startswith("openalex://paper/"):
                paper_id = uri.split("/")[-1]
                paper = self.rag.get_paper_by_id(paper_id)
                if paper:
                    content = json.dumps(paper.to_dict(), indent=2)
                    return ReadResourceResult(
                        contents=[TextContent(type="text", text=content)]
                    )
                else:
                    return ReadResourceResult(
                        contents=[TextContent(type="text", text="Paper not found")]
                    )
            
            elif uri.startswith("openalex://author/"):
                author = uri.split("/")[-1]
                papers = self.rag.get_papers_by_author(author)
                content = json.dumps([p.to_dict() for p in papers], indent=2)
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=content)]
                )
            
            elif uri.startswith("openalex://year-range/"):
                parts = uri.split("/")
                start_year = int(parts[-2])
                end_year = int(parts[-1])
                papers = self.rag.get_papers_by_year_range(start_year, end_year)
                content = json.dumps([p.to_dict() for p in papers], indent=2)
                return ReadResourceResult(
                    contents=[TextContent(type="text", text=content)]
                )
            
            else:
                return ReadResourceResult(
                    contents=[TextContent(type="text", text="Unknown resource URI")]
                )
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="search_papers",
                    description="Search for academic papers using semantic similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 100
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_paper",
                    description="Get a specific paper by OpenAlex ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "openalex_id": {
                                "type": "string",
                                "description": "OpenAlex ID of the paper"
                            }
                        },
                        "required": ["openalex_id"]
                    }
                ),
                Tool(
                    name="search_by_author",
                    description="Search papers by author name",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "author_name": {
                                "type": "string",
                                "description": "Author name to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 100
                            }
                        },
                        "required": ["author_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> CallToolResult:
            """Execute a tool"""
            try:
                if name == "search_papers":
                    query = arguments["query"]
                    k = arguments.get("k", 100)
                    papers = self.rag.search_papers(query, k=k)
                    results = [p.to_dict() for p in papers]
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=json.dumps(results, indent=2)
                        )]
                    )
                
                elif name == "get_paper":
                    openalex_id = arguments["openalex_id"]
                    paper = self.rag.get_paper_by_id(openalex_id)
                    if paper:
                        return CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=json.dumps(paper.to_dict(), indent=2)
                            )]
                        )
                    else:
                        return CallToolResult(
                            content=[TextContent(type="text", text="Paper not found")]
                        )
                
                elif name == "search_by_author":
                    author_name = arguments["author_name"]
                    limit = arguments.get("limit", 100)
                    papers = self.rag.get_papers_by_author(author_name, limit=limit)
                    results = [p.to_dict() for p in papers]
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text=json.dumps(results, indent=2)
                        )]
                    )
                
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                    )
            
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
    
    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Main function to demonstrate the RAG implementation"""
    print("🚀 OpenAlex RAG Implementation with MongoDB")
    print("=" * 60)
    
    # Initialize RAG
    try:
        rag = OpenAlexRAG(email="dan@danbroz.com", k=475)
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    # Example queries
    queries = [
        "machine learning transformers",
        "quantum computing algorithms", 
        "climate change renewable energy"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        # Search for papers
        papers = rag.search_papers(query, k=5)
        
        if papers:
            # Display results
            print(f"\n📊 Top {len(papers)} results:")
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                print(f"   Year: {paper.year}")
                print(f"   Venue: {paper.venue or 'Unknown'}")
                print(f"   Abstract: {paper.abstract[:150]}...")
            
            print(f"\n✅ Retrieved {len(papers)} papers for query: '{query}'")
        else:
            print(f"❌ No papers found for query: '{query}'")
        
        print("\n" + "="*80 + "\n")
    
    # Close connections
    rag.close()


def start_mcp_server():
    """Start the MCP server"""
    print("🚀 Starting OpenAlex RAG MCP Server")
    print("=" * 60)
    
    try:
        # Initialize RAG
        rag = OpenAlexRAG(email="dan@danbroz.com", k=475)
        
        # Create and run MCP server
        mcp_server = OpenAlexMCPServer(rag)
        asyncio.run(mcp_server.run())
        
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return
    finally:
        if 'rag' in locals():
            rag.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        start_mcp_server()
    else:
        main()
