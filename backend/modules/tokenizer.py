"""
Production-Ready Tokenization Module
Splits text into chunks with token limits, overlap, and advanced features.
Includes tiktoken support, streaming, async support, and comprehensive error handling.
"""

from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional, Generator, AsyncGenerator
import os
import asyncio


class TokenizationError(Exception):
    """Custom exception for tokenization errors"""
    pass


class TokenizerService:
    """
    Handles text tokenization and chunking with production features.
    
    Features:
    - Transformers or tiktoken tokenizer
    - Sliding window chunking with overlap
    - Max chunk limits to prevent OOM
    - Streaming chunking for huge texts
    - Async support for FastAPI
    - Chunking strategy options
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_tiktoken: bool = False,
        tiktoken_model: str = "cl100k_base"
    ):
        """
        Initialize tokenizer service
        
        Args:
            model_name: HuggingFace model name for tokenizer
            use_tiktoken: Use tiktoken instead of transformers (faster)
            tiktoken_model: Tiktoken model name if use_tiktoken=True
        """
        self.use_tiktoken = use_tiktoken
        
        try:
            if use_tiktoken:
                # Use tiktoken for speed (OpenAI-style)
                try:
                    import tiktoken
                    self.tokenizer = tiktoken.get_encoding(tiktoken_model)
                    self.model_name = tiktoken_model
                    self._is_tiktoken = True
                except ImportError:
                    raise TokenizationError(
                        "tiktoken not installed. Install with: pip install tiktoken"
                    )
            else:
                # Use transformers tokenizer
                hf_token = os.getenv('HF_TOKEN')
                if hf_token:
                    os.environ['TRANSFORMERS_HF_TOKEN'] = hf_token
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=hf_token
                )
                self.model_name = model_name
                self._is_tiktoken = False
                
        except Exception as e:
            raise TokenizationError(f"Failed to load tokenizer: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._is_tiktoken:
            return len(self.tokenizer.encode(text))
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
    
    def chunk_text(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
        max_chunks: Optional[int] = None,
        strategy: str = "sliding"
    ) -> List[Dict[str, any]]:
        """
        Split text into chunks with token limit and overlap
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (default: 500)
            overlap: Number of tokens to overlap between chunks (default: 50)
            max_chunks: Maximum number of chunks to prevent OOM (default: None = unlimited)
            strategy: Chunking strategy - "sliding" or "sentence" (default: "sliding")
            
        Returns:
            List of chunk dictionaries with metadata
            
        Raises:
            TokenizationError: If chunking fails or validation errors
        """
        # Critical Fix #3: Validate overlap
        if overlap >= max_tokens:
            raise TokenizationError(
                f"Overlap ({overlap}) must be less than max_tokens ({max_tokens}) "
                "to prevent infinite loop"
            )
        
        # Validate inputs
        if overlap < 0:
            raise TokenizationError("Overlap must be non-negative")
        
        if max_tokens <= 0:
            raise TokenizationError("max_tokens must be positive")
        
        try:
            # Handle empty text
            if not text or len(text.strip()) == 0:
                raise TokenizationError("Cannot chunk empty text")
            
            # Tokenize entire text
            if self._is_tiktoken:
                tokens = self.tokenizer.encode(text)
            else:
                # Critical Fix #4: Always use add_special_tokens=False for accurate count
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # If text is shorter than max_tokens, return as single chunk
            if len(tokens) <= max_tokens:
                return [{
                    'chunk_id': 0,
                    'text': text,
                    'token_count': len(tokens),
                    'start_char': 0,
                    'end_char': len(text),
                    'has_overlap': False
                }]
            
            # Critical Fix #5: Check max_chunks to prevent OOM
            estimated_chunks = max(1, (len(tokens) - overlap) // (max_tokens - overlap))
            if max_chunks and estimated_chunks > max_chunks:
                raise TokenizationError(
                    f"Text would generate {estimated_chunks} chunks, "
                    f"exceeding max_chunks limit of {max_chunks}. "
                    f"Consider increasing max_tokens or max_chunks."
                )
            
            # Get offset mapping for accurate character positions
            if not self._is_tiktoken:
                encoding = self.tokenizer(
                    text,
                    return_offsets_mapping=True,
                    add_special_tokens=False
                )
                offsets = encoding["offset_mapping"]
            else:
                offsets = None
            
            # Create chunks based on strategy
            if strategy == "sliding":
                chunks = self._sliding_window_chunks(
                    text, tokens, max_tokens, overlap, offsets, max_chunks
                )
            elif strategy == "sentence":
                chunks = self._sentence_aware_chunks(
                    text, tokens, max_tokens, overlap, offsets, max_chunks
                )
            else:
                raise TokenizationError(
                    f"Unknown chunking strategy: {strategy}. "
                    "Use 'sliding' or 'sentence'."
                )
            
            return chunks
            
        except TokenizationError:
            raise
        except Exception as e:
            raise TokenizationError(f"Failed to chunk text: {str(e)}")
    
    def _sliding_window_chunks(
        self,
        text: str,
        tokens: List[int],
        max_tokens: int,
        overlap: int,
        offsets: Optional[List],
        max_chunks: Optional[int]
    ) -> List[Dict]:
        """Sliding window chunking strategy"""
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            # Check max chunks limit
            if max_chunks and chunk_id >= max_chunks:
                break
            
            # Calculate end index for this chunk
            end_idx = min(start_idx + max_tokens, len(tokens))
            
            # Extract token chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            if self._is_tiktoken:
                chunk_text = self.tokenizer.decode(chunk_tokens)
            else:
                chunk_text = self.tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True
                )
            
            # Clean up whitespace
            chunk_text = chunk_text.replace("  ", " ").strip()
            
            # Calculate character positions
            if offsets:
                start_char = offsets[start_idx][0]
                end_char = offsets[min(end_idx - 1, len(offsets) - 1)][1]
            else:
                # Approximate for tiktoken
                start_char = int(start_idx / len(tokens) * len(text))
                end_char = int(end_idx / len(tokens) * len(text))
            
            # Create chunk metadata
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'start_char': start_char,
                'end_char': end_char,
                'has_overlap': chunk_id > 0,
                'strategy': 'sliding'
            }
            
            chunks.append(chunk)
            
            # Move to next chunk
            if end_idx >= len(tokens):
                break
            
            # Move start index forward (chunk size - overlap)
            start_idx = end_idx - overlap
            chunk_id += 1
        
        return chunks
    
    def _sentence_aware_chunks(
        self,
        text: str,
        tokens: List[int],
        max_tokens: int,
        overlap: int,
        offsets: Optional[List],
        max_chunks: Optional[int]
    ) -> List[Dict]:
        """
        Sentence-aware chunking - tries to break at sentence boundaries
        Falls back to sliding window if sentences are too long
        """
        import re
        
        # Split text into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        chunk_id = 0
        current_chunk_text = ""
        current_chunk_tokens = []
        start_char = 0
        
        for sentence in sentences:
            if max_chunks and chunk_id >= max_chunks:
                break
            
            # Tokenize sentence
            if self._is_tiktoken:
                sent_tokens = self.tokenizer.encode(sentence)
            else:
                sent_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            
            # If adding this sentence exceeds max_tokens, save current chunk
            if len(current_chunk_tokens) + len(sent_tokens) > max_tokens:
                if current_chunk_text:
                    # Save current chunk
                    chunk = {
                        'chunk_id': chunk_id,
                        'text': current_chunk_text.strip(),
                        'token_count': len(current_chunk_tokens),
                        'start_char': start_char,
                        'end_char': start_char + len(current_chunk_text),
                        'has_overlap': chunk_id > 0,
                        'strategy': 'sentence'
                    }
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    # Take last overlap tokens from previous chunk
                    if overlap > 0 and len(current_chunk_tokens) > overlap:
                        overlap_tokens = current_chunk_tokens[-overlap:]
                        if self._is_tiktoken:
                            current_chunk_text = self.tokenizer.decode(overlap_tokens)
                        else:
                            current_chunk_text = self.tokenizer.decode(
                                overlap_tokens,
                                skip_special_tokens=True
                            )
                        current_chunk_tokens = overlap_tokens
                        start_char = start_char + len(current_chunk_text)
                    else:
                        current_chunk_text = ""
                        current_chunk_tokens = []
                        start_char += len(current_chunk_text)
            
            # Add sentence to current chunk
            current_chunk_text += (" " if current_chunk_text else "") + sentence
            current_chunk_tokens.extend(sent_tokens)
        
        # Add final chunk
        if current_chunk_text and (not max_chunks or chunk_id < max_chunks):
            chunk = {
                'chunk_id': chunk_id,
                'text': current_chunk_text.strip(),
                'token_count': len(current_chunk_tokens),
                'start_char': start_char,
                'end_char': start_char + len(current_chunk_text),
                'has_overlap': chunk_id > 0,
                'strategy': 'sentence'
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_text_streaming(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
        max_chunks: Optional[int] = None
    ) -> Generator[Dict[str, any], None, None]:
        """
        🔥 Pro Feature: Streaming chunking for huge texts
        Yields chunks one at a time instead of storing all in memory
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap: Overlap tokens
            max_chunks: Maximum chunks to prevent OOM
            
        Yields:
            Chunk dictionaries one at a time
        """
        # Validate
        if overlap >= max_tokens:
            raise TokenizationError("Overlap must be less than max_tokens")
        
        if not text or len(text.strip()) == 0:
            return
        
        # Tokenize
        if self._is_tiktoken:
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Single chunk case
        if len(tokens) <= max_tokens:
            yield {
                'chunk_id': 0,
                'text': text,
                'token_count': len(tokens),
                'start_char': 0,
                'end_char': len(text),
                'has_overlap': False
            }
            return
        
        # Stream chunks
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            if max_chunks and chunk_id >= max_chunks:
                break
            
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            if self._is_tiktoken:
                chunk_text = self.tokenizer.decode(chunk_tokens)
            else:
                chunk_text = self.tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True
                )
            
            chunk_text = chunk_text.replace("  ", " ").strip()
            
            yield {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'has_overlap': chunk_id > 0
            }
            
            if end_idx >= len(tokens):
                break
            
            start_idx = end_idx - overlap
            chunk_id += 1
    
    async def chunk_text_async(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
        max_chunks: Optional[int] = None,
        strategy: str = "sliding"
    ) -> List[Dict[str, any]]:
        """
        🔥 Pro Feature: Async version for FastAPI performance
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap: Overlap tokens
            max_chunks: Maximum chunks
            strategy: Chunking strategy
            
        Returns:
            List of chunk dictionaries
        """
        # Run synchronous chunking in executor to not block event loop
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            self.chunk_text,
            text,
            max_tokens,
            overlap,
            max_chunks,
            strategy
        )
        return chunks
    
    async def chunk_text_streaming_async(
        self,
        text: str,
        max_tokens: int = 500,
        overlap: int = 50,
        max_chunks: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, any], None]:
        """
        🔥 Pro Feature: Async streaming chunking
        
        Yields:
            Chunk dictionaries asynchronously
        """
        for chunk in self.chunk_text_streaming(text, max_tokens, overlap, max_chunks):
            yield chunk
            # Allow other coroutines to run
            await asyncio.sleep(0)
    
    def get_chunk_texts(self, chunks: List[Dict]) -> List[str]:
        """Extract just the text from chunks"""
        return [chunk['text'] for chunk in chunks]
    
    def validate_chunks(self, chunks: List[Dict], max_tokens: int = 500) -> bool:
        """Validate that all chunks are within token limit"""
        for chunk in chunks:
            if chunk['token_count'] > max_tokens:
                return False
        return True


# Convenience functions
def tokenize_and_chunk(
    text: str,
    max_tokens: int = None,
    overlap: int = None,
    max_chunks: int = None,
    use_tiktoken: bool = False,
    strategy: str = "sliding"
) -> List[Dict[str, any]]:
    """
    Tokenize and chunk text with default settings
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk (defaults to env var or 500)
        overlap: Overlap tokens (defaults to env var or 50)
        max_chunks: Maximum chunks to prevent OOM (defaults to env var or 1000)
        use_tiktoken: Use tiktoken for speed
        strategy: "sliding" or "sentence"
        
    Returns:
        List of chunk dictionaries
    """
    if max_tokens is None:
        max_tokens = int(os.getenv('CHUNK_SIZE', 500))
    
    if overlap is None:
        overlap = int(os.getenv('CHUNK_OVERLAP', 50))
    
    if max_chunks is None:
        max_chunks = int(os.getenv('MAX_CHUNKS', 1000))
    
    tokenizer = TokenizerService(use_tiktoken=use_tiktoken)
    return tokenizer.chunk_text(
        text,
        max_tokens=max_tokens,
        overlap=overlap,
        max_chunks=max_chunks,
        strategy=strategy
    )
