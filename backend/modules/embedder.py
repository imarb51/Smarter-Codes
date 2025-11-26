"""
Embedding Generation Module
Generates semantic embeddings using Google Gemini API (text-embedding-004) or local sentence-transformers.

Correct implementation using google-generativeai package as per official docs:
https://ai.google.dev/gemini-api/docs/embeddings
"""

import os
import time
from typing import List, Union, Optional
import numpy as np


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors"""
    pass


class EmbedderService:
    """
    Production-ready embedder using Google Gemini text-embedding-004 (768 dims)
    with local sentence-transformers fallback.
    
    Features:
    - Google Gemini API (text-embedding-004, 768 dimensions)
    - Local fallback (sentence-transformers all-MiniLM-L6-v2, 384 dimensions)
    - Batch processing with configurable batch size
    - Exponential backoff for rate limiting
    - GPU/CPU auto-detection for local model
    - Async support for FastAPI
    """
    
    def __init__(
        self,
        use_gemini: bool = True,
        model_name: str = "models/text-embedding-004",
        local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_retries: int = 3
    ):
        """
        Initialize embedder service
        
        Args:
            use_gemini: Use Gemini API (default: True)
            model_name: Gemini model - "models/text-embedding-004" (768 dims)
            local_model: Local model for fallback
            batch_size: Batch size for processing (default: 32)
            max_retries: Max retry attempts for rate limits (default: 3)
        """
        self.use_gemini = use_gemini
        self.model_name = model_name
        self.local_model_name = local_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize appropriate backend
        if use_gemini:
            self._init_gemini()
        else:
            self._init_local_model()
    
    def _init_gemini(self):
        """
        Initialize Google Gemini API using google-generativeai package
        
        CORRECT usage per official docs:
        import google.generativeai as genai
        genai.configure(api_key=key)
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        """
        try:
            import google.generativeai as genai
            
            # Get API key from environment
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise EmbeddingError(
                    "GEMINI_API_KEY not found in environment. "
                    "Set it in your .env file."
                )
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.genai = genai
            
            # Set embedding dimension
            # text-embedding-004 outputs 768 dimensions
            self.embedding_dimension = 768
            
            print(f"✓ Initialized Gemini embeddings: {self.model_name} (768 dims)")
            
        except ImportError:
            raise EmbeddingError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Gemini: {str(e)}")
    
    def _init_local_model(self):
        """Initialize local sentence-transformers model with GPU/CPU detection"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Auto-detect GPU/CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ No GPU, using CPU (slower)")
            
            # Load model
            print(f"Loading {self.local_model_name}...")
            self.model = SentenceTransformer(self.local_model_name, device=device)
            self.device = device
            
            # Get dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            print(f"✓ Local model loaded on {device} ({self.embedding_dimension} dims)")
            
        except ImportError as e:
            raise EmbeddingError(
                f"sentence-transformers or torch not installed: {e}. "
                "Install with: pip install sentence-transformers torch"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load local model: {e}")
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Normalize input
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Filter empty strings
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            raise EmbeddingError("No valid non-empty texts to embed")
        
        # Generate based on backend
        if self.use_gemini:
            embeddings = self._generate_gemini_embeddings(texts)
        else:
            embeddings = self._generate_local_embeddings(texts)
        
        # Return single or array
        if is_single:
            return embeddings[0]
        return embeddings
    
    def _generate_gemini_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Google Gemini API with safe chunking
        for free-tier limits (~1500 chars max per request).
        """

        MAX_CHARS = 1500  # safe limit for free Gemini
        all_embeddings = []

        for original_text in texts:
            # --- Step 1: Split into smaller chunks ---
            chunks = []
            start = 0
            while start < len(original_text):
                end = start + MAX_CHARS
                chunks.append(original_text[start:end])
                start = end

            chunk_embeddings = []

            # --- Step 2: Send each chunk to Gemini ---
            for chunk in chunks:
                success = False
                for attempt in range(self.max_retries):
                    try:
                        result = self.genai.embed_content(
                            model=self.model_name,
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        emb = result["embedding"]
                        chunk_embeddings.append(emb)
                        success = True
                        break

                    except Exception as e:
                        error_msg = str(e).lower()

                        if any(k in error_msg for k in ["rate", "quota", "429"]):
                            wait = (2 ** attempt)
                            print(f"⚠ Rate-limit. Waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        
                        raise EmbeddingError(f"Gemini error: {e}")

                if not success:
                    raise EmbeddingError("Failed after retries")

            # --- Step 3: Combine chunk embeddings into one vector ---
            final_vector = np.mean(np.array(chunk_embeddings), axis=0)
            all_embeddings.append(final_vector)

        return np.array(all_embeddings, dtype=np.float32)

    
    def _generate_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local sentence-transformers model"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (768 for Gemini, 384 for local)"""
        return self.embedding_dimension
    
    async def generate_embeddings_async(
        self,
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """Async version for FastAPI"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings, texts)


# Convenience functions
def generate_embeddings(
    texts: Union[str, List[str]],
    use_gemini: Optional[bool] = None
) -> np.ndarray:
    """
    Generate embeddings (convenience wrapper)
    
    Args:
        texts: Single text or list of texts
        use_gemini: Use Gemini API (defaults to env USE_GEMINI or True)
        
    Returns:
        Embeddings array
    """
    if use_gemini is None:
        use_gemini = os.getenv('USE_GEMINI', 'true').lower() in ('true', '1', 'yes')
    
    embedder = EmbedderService(use_gemini=use_gemini)
    return embedder.generate_embeddings(texts)


def get_embedding_dimension(use_gemini: Optional[bool] = None) -> int:
    """Get embedding dimension for backend"""
    if use_gemini is None:
        use_gemini = os.getenv('USE_GEMINI', 'true').lower() in ('true', '1', 'yes')
    
    return 768 if use_gemini else 384
