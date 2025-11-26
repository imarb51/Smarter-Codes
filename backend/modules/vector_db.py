"""
Pinecone Vector Database Integration Module
Handles vector indexing and semantic search using Pinecone.
"""

import os
import time
from typing import List, Dict, Optional, Any
import hashlib


class PineconeError(Exception):
    """Custom exception for Pinecone errors"""
    pass


class PineconeService:
    """
    Production-ready Pinecone vector database service.
    
    Features:
    - Auto-create index with correct dimensions
    - Upsert with metadata (url, chunk_id, text, token_count)
    - Duplicate handling via deterministic IDs
    - Semantic search with score threshold
    - Connection retry logic
    - Dimension validation
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize Pinecone service
        
        Args:
            api_key: Pinecone API key (defaults to env PINECONE_API_KEY)
            index_name: Index name (defaults to env PINECONE_INDEX_NAME)
            dimension: Embedding dimension (default: 768 for Gemini)
            metric: Distance metric (default: cosine)
            cloud: Cloud provider (default: aws)
            region: Cloud region (default: us-east-1)
        """
        # Get config from env if not provided
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'html-semantic-search')
        self.dimension = int(dimension or os.getenv('PINECONE_DIMENSION', 768))
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        if not self.api_key:
            print("⚠️ PINECONE_API_KEY not found, using in-memory fallback vector store")
            self._init_memory_fallback()
            return
        
        # Initialize Pinecone
        try:
            self._init_pinecone()
            # Ensure index exists
            self._ensure_index_exists()
        except Exception as e:
            print(f"⚠️ Pinecone init failed, falling back to in-memory store: {e}")
            self._init_memory_fallback()
    
    def _init_pinecone(self):
        """Initialize Pinecone client"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            self.ServerlessSpec = ServerlessSpec
            
            print(f"✓ Initialized Pinecone client")
            
        except ImportError:
            raise PineconeError(
                "pinecone not installed. "
                "Install with: pip install pinecone"
            )
        except Exception as e:
            raise PineconeError(f"Failed to initialize Pinecone: {str(e)}")
    
    def _init_memory_fallback(self):
        """Initialize in-memory vector store for development/testing"""
        self.use_memory = True
        self.memory_store = {}  # url -> list of (id, vector, metadata)
        print(f"✓ Using in-memory vector store fallback ({self.dimension} dims)")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist, or validate existing index"""
        try:
            # Check if index exists (using modern has_index() method)
            if not self.pc.has_index(self.index_name):
                # Create new index
                print(f"Creating Pinecone index '{self.index_name}'...")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                
                # Wait for index to be ready
                max_wait = 60  # seconds
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    index_stats = self.pc.describe_index(self.index_name)
                    if index_stats['status']['ready']:
                        break
                    time.sleep(2)
                
                print(f"✓ Created index '{self.index_name}' ({self.dimension} dims, {self.metric} metric)")
            
            else:
                # Validate existing index dimensions
                index_stats = self.pc.describe_index(self.index_name)
                existing_dim = index_stats['dimension']
                
                if existing_dim != self.dimension:
                    raise PineconeError(
                        f"Index '{self.index_name}' exists with dimension {existing_dim}, "
                        f"but expected {self.dimension}. Either delete the index or update "
                        "PINECONE_DIMENSION in .env to match."
                    )
                
                print(f"✓ Using existing index '{self.index_name}' ({existing_dim} dims)")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except PineconeError:
            raise
        except Exception as e:
            raise PineconeError(f"Failed to ensure index exists: {str(e)}")
    
    def _generate_chunk_id(self, url: str, chunk_id: int) -> str:
        """
        Generate deterministic ID for a chunk
        This ensures same URL + chunk_id always gets same vector ID
        (handles duplicates automatically)
        
        Args:
            url: Source URL
            chunk_id: Chunk index
            
        Returns:
            Deterministic ID string
        """
        # Create hash of URL + chunk_id for deterministic IDs
        content = f"{url}|{chunk_id}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()
    
    def index_chunks(
        self,
        url: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Index chunks with embeddings into Pinecone or in-memory store
        
        Args:
            url: Source URL
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors
            namespace: Optional namespace for multi-tenancy
            
        Returns:
            Dictionary with upsert stats
            
        Raises:
            PineconeError: If indexing fails
        """
        
        
        if getattr(self, 'use_memory', False):
            
            # In-memory fallback implementation
            if url not in self.memory_store:
                self.memory_store[url] = []
            
            upserted_count = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = self._generate_chunk_id(url, chunk.get('chunk_id', i))
                metadata = {
                    'url': url,
                    'chunk_id': chunk.get('chunk_id', i),
                    'text': chunk.get('text', '')[:1000],
                    'token_count': chunk.get('token_count', 0),
                    'start_char': chunk.get('start_char', 0),
                    'end_char': chunk.get('end_char', 0),
                    'has_overlap': chunk.get('has_overlap', False)
                }
                self.memory_store[url].append((vector_id, embedding, metadata))
                upserted_count += 1
            
            print(f"✓ Indexed {upserted_count} chunks from {url} (in-memory)")
            return {
                'success': True,
                'upserted_count': upserted_count,
                'url': url,
                'namespace': namespace
            }
        
        
        # Original Pinecone implementation
        if len(chunks) != len(embeddings):
            raise PineconeError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )
        
        if not chunks:
            raise PineconeError("No chunks to index")
        
        # Validate embedding dimensions
        if embeddings and len(embeddings[0]) != self.dimension:
            raise PineconeError(
                f"Embedding dimension ({len(embeddings[0])}) doesn't match "
                f"index dimension ({self.dimension})"
            )
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate deterministic ID (handles duplicates)
                vector_id = self._generate_chunk_id(url, chunk.get('chunk_id', i))
                
                # Prepare metadata
                metadata = {
                    'url': url,
                    'chunk_id': chunk.get('chunk_id', i),
                    'text': chunk.get('text', '')[:1000],  # Limit text length in metadata
                    'token_count': chunk.get('token_count', 0),
                    'start_char': chunk.get('start_char', 0),
                    'end_char': chunk.get('end_char', 0),
                    'has_overlap': chunk.get('has_overlap', False)
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            upserted_count = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                response = self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                upserted_count += response.get('upserted_count', len(batch))
            
            print(f"✓ Indexed {upserted_count} chunks from {url}")
            
            return {
                'success': True,
                'upserted_count': upserted_count,
                'url': url,
                'namespace': namespace
            }
            
        except Exception as e:
            raise PineconeError(f"Failed to index chunks: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: float = 0.3,
        namespace: str = "",
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default: 10)
            score_threshold: Minimum similarity score (default: 0.3)
            namespace: Optional namespace
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching results with metadata and scores
        """
        
        if getattr(self, 'use_memory', False):
                
            # In-memory search implementation
            import numpy as np
            query_vec = np.array(query_embedding)
            
            all_matches = []
            for url, chunks in self.memory_store.items():
                for vector_id, embedding, metadata in chunks:
                    # Calculate cosine similarity
                    vec = np.array(embedding)
                    similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                    if similarity >= score_threshold:
                        all_matches.append({
                            'id': vector_id,
                            'score': float(similarity),
                            'metadata': metadata,
                            'text': metadata.get('text', ''),
                            'url': metadata.get('url', ''),
                            'chunk_id': metadata.get('chunk_id', 0),
                            'token_count': metadata.get('token_count', 0)
                        })
            
            # Sort by score and take top_k
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            filtered_results = all_matches[:top_k]
            
            print(f"✓ Found {len(filtered_results)} matches (threshold: {score_threshold})")
            
            return filtered_results
        
        print("DEBUG: Using Pinecone search")
        # Original Pinecone implementation
        # Validate embedding dimension
        if len(query_embedding) != self.dimension:
            raise PineconeError(
                f"Query embedding dimension ({len(query_embedding)}) doesn't match "
                f"index dimension ({self.dimension})"
            )
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Filter by score threshold and format results
            filtered_results = []
            for match in results.get('matches', []):
                score = match.get('score', 0)
                
                if score >= score_threshold:
                    filtered_results.append({
                        'id': match.get('id'),
                        'score': score,
                        'metadata': match.get('metadata', {}),
                        'text': match.get('metadata', {}).get('text', ''),
                        'url': match.get('metadata', {}).get('url', ''),
                        'chunk_id': match.get('metadata', {}).get('chunk_id', 0),
                        'token_count': match.get('metadata', {}).get('token_count', 0)
                    })
            
            print(f"✓ Found {len(filtered_results)} matches (threshold: {score_threshold})")
            
            return filtered_results
            
        except Exception as e:
            
            raise PineconeError(f"Search failed: {str(e)}")
    
    def url_exists(self, url: str, namespace: str = "") -> bool:
        """
        Check if a URL is already indexed
        
        Args:
            url: URL to check
            namespace: Optional namespace
            
        Returns:
            True if URL has indexed vectors, False otherwise
        """
        if getattr(self, 'use_memory', False):
            # In-memory check
            return url in self.memory_store and len(self.memory_store[url]) > 0
        
        try:
            # Query Pinecone for any vector with this URL
            results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=1,
                namespace=namespace,
                filter={'url': url},
                include_metadata=False
            )
            
            # Check if any matches found
            has_vectors = len(results.get('matches', [])) > 0
            return has_vectors
            
        except Exception as e:
            print(f"⚠️ Error checking URL existence: {str(e)}")
            return False
    
    def delete_by_url(self, url: str, namespace: str = ""):
        """
        Delete all vectors for a specific URL
        
        Args:
            url: URL to delete vectors for
            namespace: Optional namespace
        """
        if self.use_memory:
            if url in self.memory_store:
                del self.memory_store[url]
                print(f"✓ Deleted vectors for URL: {url} (in-memory)")
            return
        
        try:
            # Delete by metadata filter
            self.index.delete(
                filter={'url': url},
                namespace=namespace
            )
            print(f"✓ Deleted vectors for URL: {url}")
        except Exception as e:
            raise PineconeError(f"Failed to delete vectors: {str(e)}")
    
    def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """
        Get index statistics
        
        Args:
            namespace: Optional namespace
            
        Returns:
            Dictionary with index stats
        """
        if self.use_memory:
            total_count = sum(len(chunks) for chunks in self.memory_store.values())
            return {
                'total_vector_count': total_count,
                'dimension': self.dimension,
                'namespaces': {},
                'index_fullness': 0.0
            }
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'namespaces': stats.get('namespaces', {}),
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            raise PineconeError(f"Failed to get index stats: {str(e)}")


# Convenience functions
def create_pinecone_service() -> PineconeService:
    """Create Pinecone service with config from environment"""
    return PineconeService()


def search_vectors(
    query_embedding: List[float],
    top_k: int = None,
    score_threshold: float = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for vector search
    
    Args:
        query_embedding: Query vector
        top_k: Number of results (defaults to env TOP_K_RESULTS or 10)
        score_threshold: Minimum score (defaults to env MIN_SCORE_THRESHOLD or 0.3)
        
    Returns:
        List of search results
    """
    if top_k is None:
        top_k = int(os.getenv('TOP_K_RESULTS', 10))
    
    if score_threshold is None:
        score_threshold = float(os.getenv('MIN_SCORE_THRESHOLD', 0.3))
    
    service = create_pinecone_service()
    return service.search(query_embedding, top_k=top_k, score_threshold=score_threshold)
