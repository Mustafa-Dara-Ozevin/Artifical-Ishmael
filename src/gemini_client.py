"""Google AI Studio Gemini client for Moby-Dick GraphRAG (using new google-genai SDK)."""

import time
import logging
from typing import Any, Iterator

from google import genai
from google.genai import types

from .config import GeminiConfig, get_config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API client with rate limiting and retry logic."""
    
    def __init__(self, config: GeminiConfig | None = None):
        """Initialize Gemini client.
        
        Args:
            config: Gemini configuration. If None, loads from environment.
        """
        self.config = config or get_config().gemini
        self._client: genai.Client | None = None
    
    @property
    def client(self) -> genai.Client:
        """Get or create the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client
    
    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry.
        
        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Function result.
            
        Raises:
            Last exception if all retries fail.
        """
        last_exception = None
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    raise e
                elif "unavailable" in error_str or "503" in error_str:
                    last_exception = e
                    logger.warning(f"Service unavailable, attempt {attempt + 1}/{self.config.max_retries}. Waiting {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
        
        raise last_exception

    def _retry_embedding_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute embedding function with stricter backoff to prevent quota exhaustion.

        Uses max_embedding_retries instead of max_retries to be more conservative
        with embedding API calls since quota is shared across the free tier.
        """
        last_exception = None
        delay = self.config.embedding_retry_delay

        for attempt in range(self.config.max_embedding_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    last_exception = e
                    logger.warning(
                        f"Embedding rate limited, attempt {attempt + 1}/{self.config.max_embedding_retries}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                elif "unavailable" in error_str or "503" in error_str:
                    last_exception = e
                    logger.warning(
                        f"Service unavailable, attempt {attempt + 1}/{self.config.max_embedding_retries}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

        raise last_exception

    def check_quota_remaining(self) -> bool:
        """Check if quota is available using a lightweight API call."""
        try:
            self.count_tokens("test")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "resource_exhausted" in error_str:
                logger.warning("Quota check failed: quota likely exhausted")
                return False
            logger.debug(f"Quota check encountered non-quota error: {e}")
            return True
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.
            temperature: Optional temperature override.
            
        Returns:
            Generated text.
        """
        def _generate():
            config = types.GenerateContentConfig(
                temperature=temperature or 0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
            
            if system_instruction:
                config.system_instruction = system_instruction
            
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=config
            )
            return response.text
        
        return self._retry_with_backoff(_generate)
    
    def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None
    ) -> Iterator[str]:
        """Generate text with streaming.
        
        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.
            
        Yields:
            Text chunks as they are generated.
        """
        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        if system_instruction:
            config.system_instruction = system_instruction
        
        for chunk in self.client.models.generate_content_stream(
            model=self.config.model,
            contents=prompt,
            config=config
        ):
            if chunk.text:
                yield chunk.text
    
    def embed(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed.
            task_type: Embedding task type. Options:
                - "RETRIEVAL_QUERY": For query embeddings
                - "RETRIEVAL_DOCUMENT": For document embeddings
                - "SEMANTIC_SIMILARITY": For similarity comparisons
                - "CLASSIFICATION": For classification tasks
                - "CLUSTERING": For clustering tasks
                
        Returns:
            Embedding vector as list of floats.
        """
        def _embed():
            response = self.client.models.embed_content(
                model=self.config.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return response.embeddings[0].values
        
        return self._retry_embedding_with_backoff(_embed)
    
    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            task_type: Embedding task type.
            
        Returns:
            List of embedding vectors.
        """
        def _embed_batch():
            response = self.client.models.embed_content(
                model=self.config.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return [emb.values for emb in response.embeddings]

        return self._retry_embedding_with_backoff(_embed_batch)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Token count.
        """
        response = self.client.models.count_tokens(
            model=self.config.model,
            contents=text
        )
        return response.total_tokens


# Singleton instance
_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get or create the singleton Gemini client."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
