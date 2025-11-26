"""
Search Orchestration Service
Main service that orchestrates the complete semantic search pipeline.

Pipeline:
1. Fetch HTML from URL
2. Parse and clean HTML
3. Tokenize into chunks
4. Generate embeddings
5. Index in Pinecone
6. Search with query
7. Return ranked results
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from modules.fetcher import HTMLFetcher, HTMLFetchError
from modules.parser import HTMLParser, HTMLParseError
from modules.tokenizer import TokenizerService, TokenizationError
from modules.embedder import EmbedderService, EmbeddingError
from modules.vector_db import PineconeService, PineconeError


class SearchServiceError(Exception):
    """Custom exception for search service errors"""
    pass


class SearchService:
    """
    Complete semantic search orchestration service.
    
    Provides end-to-end pipeline from URL to search results.
    """
    
    def __init__(
        self,
        use_gemini: bool = True,
        use_tiktoken: bool = False,
        max_html_size: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_chunks: int = None
    ):
        """
        Initialize search service
        
        Args:
            use_gemini: Use Gemini for embeddings (default: True)
            use_tiktoken: Use tiktoken for faster tokenization (default: False)
            max_html_size: Max HTML size in chars (defaults to env or 500k)
            chunk_size: Tokens per chunk (defaults to env or 500)
            chunk_overlap: Overlap tokens (defaults to env or 50)
            max_chunks: Max chunks to prevent OOM (defaults to env or 1000)
        """
        # Get config from env
        self.max_html_size = max_html_size or int(os.getenv('MAX_HTML_SIZE', 500000))
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 200))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 50))
        self.max_chunks = max_chunks or int(os.getenv('MAX_CHUNKS', 1000))
        
        # Initialize all services
        print("Initializing Search Service...")
        
        try:
            # HTML Fetcher
            self.fetcher = HTMLFetcher(max_html_size=self.max_html_size)
            print("✓ HTML Fetcher ready")
            
            # HTML Parser
            self.parser = HTMLParser()
            print("✓ HTML Parser ready")
            
            # Tokenizer
            self.tokenizer = TokenizerService(use_tiktoken=use_tiktoken)
            print("✓ Tokenizer ready")
            
            # Embedder (Gemini or local)
            self.embedder = EmbedderService(use_gemini=use_gemini)
            print("✓ Embedder ready")
            
            # Pinecone Vector DB
            self.vector_db = PineconeService(
                dimension=self.embedder.get_embedding_dimension()
            )
            print("✓ Vector DB ready")
            
            print(f"✅ Search Service initialized successfully!")
            
        except Exception as e:
            raise SearchServiceError(f"Failed to initialize search service: {str(e)}")
    
    def process_and_index(self, url: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Process URL and index chunks into Pinecone
        
        Pipeline:
        1. Check if URL already indexed (skip if exists unless force_reindex=True)
        2. Fetch HTML
        3. Parse and clean
        4. Tokenize into chunks
        5. Generate embeddings
        6. Index in Pinecone
        
        Args:
            url: URL to process
            force_reindex: Force re-indexing even if URL exists (default: False)
            
        Returns:
            Dictionary with processing stats
        """
        try:
            print(f"\n🔄 Processing URL: {url}")
            
            # Check if URL already exists
            if not force_reindex and self.vector_db.url_exists(url):
                print(f"⚡ URL already indexed, skipping re-index (use force_reindex=True to override)")
                return {
                    'success': True,
                    'url': url,
                    'cached': True,
                    'message': 'URL already indexed, skipped processing'
                }
            
            if force_reindex:
                print(f"🔄 Force re-indexing enabled, will process URL")
            
            # Step 1: Fetch HTML
            print("📥 Fetching HTML...")
            html_content, final_url = self.fetcher.fetch_html(url)
            print(f"✓ Fetched {len(html_content):,} characters")
            
            # Step 2: Parse and clean HTML
            print("🧹 Parsing HTML...")
            try:
                clean_text = self.parser.clean_html(html_content)
                print(f"✓ Extracted {len(clean_text):,} characters of text")
            except Exception as parse_error:
                print(f"❌ HTML parsing failed: {str(parse_error)}")
                print(f"HTML content length: {len(html_content)}")
                print(f"HTML content preview (first 500 chars): {html_content[:500]}")
                raise parse_error
            
            # Step 3: Tokenize into chunks
            print("✂️  Tokenizing into chunks...")
            chunks = self.tokenizer.chunk_text(
                clean_text,
                max_tokens=self.chunk_size,
                overlap=self.chunk_overlap,
                max_chunks=self.max_chunks
            )
            print(f"✓ Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            print("🧠 Generating embeddings...")
            chunk_texts = self.tokenizer.get_chunk_texts(chunks)
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            print(f"✓ Generated {len(embeddings)} embeddings")
            
            # Step 5: Index in Pinecone
            print("💾 Indexing in Pinecone...")
            
            # Convert embeddings to list format if needed
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            result = self.vector_db.index_chunks(
                url=final_url,
                chunks=chunks,
                embeddings=embeddings_list
            )
            print(f"✓ Indexed {result['upserted_count']} vectors")
            
            return {
                'success': True,
                'url': final_url,
                'html_size': len(html_content),
                'text_size': len(clean_text),
                'chunk_count': len(chunks),
                'indexed_count': result['upserted_count'],
                'cached': False
            }
            
        except HTMLFetchError as e:
            raise SearchServiceError(f"HTML fetch failed: {str(e)}")
        except HTMLParseError as e:
            raise SearchServiceError(f"HTML parse failed: {str(e)}")
        except TokenizationError as e:
            raise SearchServiceError(f"Tokenization failed: {str(e)}")
        except EmbeddingError as e:
            raise SearchServiceError(f"Embedding generation failed: {str(e)}")
        except PineconeError as e:
            raise SearchServiceError(f"Pinecone indexing failed: {str(e)}")
        except Exception as e:
            raise SearchServiceError(f"Unexpected error: {str(e)}")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of results (defaults to env or 10)
            score_threshold: Min score (defaults to env or 0.3)
            
        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            # Get defaults from env
            if top_k is None:
                top_k = int(os.getenv('TOP_K_RESULTS', 10))
            
            if score_threshold is None:
                score_threshold = float(os.getenv('MIN_SCORE_THRESHOLD', 0.3))
            
            print(f"\n🔍 Searching for: '{query}'")
            
            # Step 1: Generate query embedding
            print("🧠 Generating query embedding...")
            query_embedding = self.embedder.generate_embeddings(query)
            
            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            print(f"✓ Query embedding generated ({len(query_embedding)} dims)")
            
            # Step 2: Search in Pinecone
            print(f"🔎 Searching Pinecone (top_k={top_k}, threshold={score_threshold})...")
            results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if not results:
                print("⚠️  No results found above threshold")
                return []
            
            print(f"✅ Found {len(results)} relevant chunks")
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'rank': i,
                    'score': round(result['score'], 4),
                    'text': result['text'],
                    'url': result['url'],
                    'chunk_id': result['chunk_id'],
                    'token_count': result['token_count']
                })
            
            return formatted_results
            
        except EmbeddingError as e:
            raise SearchServiceError(f"Query embedding failed: {str(e)}")
        except PineconeError as e:
            raise SearchServiceError(f"Search failed: {str(e)}")
        except Exception as e:
            raise SearchServiceError(f"Unexpected search error: {str(e)}")
    
    def process_and_search(
        self,
        url: str,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Complete pipeline: process URL and search in one call
        
        Args:
            url: URL to process
            query: Search query
            top_k: Number of results
            score_threshold: Min similarity score
            force_reindex: Force re-indexing even if URL exists
            
        Returns:
            Dictionary with processing stats and search results
        """
        try:
            # Process and index
            process_result = self.process_and_index(url, force_reindex=force_reindex)
            
            # Search
            search_results = self.search(query, top_k=top_k, score_threshold=score_threshold)
            
            return {
                'success': True,
                'processing': process_result,
                'results': search_results,
                'query': query,
                'url': url,
                'total_results': len(search_results)
            }
            
        except SearchServiceError:
            raise
        except Exception as e:
            raise SearchServiceError(f"Pipeline failed: {str(e)}")
    
    def delete_url(self, url: str):
        """
        Delete all indexed chunks for a URL
        
        Args:
            url: URL to delete
        """
        try:
            self.vector_db.delete_by_url(url)
            print(f"✓ Deleted all chunks for: {url}")
        except PineconeError as e:
            raise SearchServiceError(f"Failed to delete URL: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            return self.vector_db.get_index_stats()
        except PineconeError as e:
            raise SearchServiceError(f"Failed to get stats: {str(e)}")


# Convenience function for quick usage
def search_url(url: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Quick search: process URL and search in one call
    
    Args:
        url: URL to search
        query: Search query
        top_k: Number of results
        
    Returns:
        List of search results
    """
    service = SearchService()
    result = service.process_and_search(url, query, top_k=top_k)
    return result['results']
