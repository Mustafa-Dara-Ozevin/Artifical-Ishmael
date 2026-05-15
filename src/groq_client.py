"""Groq client for Moby-Dick GraphRAG (using groq SDK)."""

import time
import logging
from typing import Any, Iterator

from groq import Groq

from .config import GroqConfig, get_config

logger = logging.getLogger(__name__)


class GroqClient:
    """Groq API client with rate limiting and retry logic."""
    
    def __init__(self, config: GroqConfig | None = None):
        """Initialize Groq client.
        
        Args:
            config: Groq configuration. If None, loads from environment.
        """
        self.config = config or get_config().groq
        self._client: Groq | None = None
    
    @property
    def client(self) -> Groq:
        """Get or create the Groq client."""
        if self._client is None:
            self._client = Groq(api_key=self.config.api_key)
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
                # Rate limit errors usually contain "rate" or "429"
                if "rate" in error_str or "429" in error_str:
                    last_exception = e
                    logger.warning(f"Groq rate limited, attempt {attempt + 1}/{self.config.max_retries}. Waiting {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                elif "unavailable" in error_str or "503" in error_str or "500" in error_str:
                    last_exception = e
                    logger.warning(f"Service issue, attempt {attempt + 1}/{self.config.max_retries}. Waiting {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
        
        raise last_exception

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
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        def _call():
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=False,
                temperature=temperature or 0.1,
                max_tokens=2048
            )
            
        response = self._retry_with_backoff(_call)
        return response.choices[0].message.content

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
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        def _call():
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=2048
            )
            
        response = self._retry_with_backoff(_call)
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Singleton instance
_groq_client = None

def get_groq_client() -> GroqClient:
    """Get the global Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
