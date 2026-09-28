"""Cumhurbaşkanlığı Savunma Sanayii Başkanlığı (SSB) EVREN API client for Moby-Dick GraphRAG.

EVREN is Turkey's sovereign AI platform providing an OpenAI-compatible API
for inference with open-weight models (e.g., DeepSeek, GLM, Qwen).
"""

import time
import logging
from typing import Any, Iterator

import requests
from openai import OpenAI

from .config import EvrenConfig, get_config

logger = logging.getLogger(__name__)


class EvrenClient:
    """EVREN SSB API client with rate limiting, retry logic, and terms acceptance handling."""
    
    def __init__(self, config: EvrenConfig | None = None):
        """Initialize EVREN client.
        
        Args:
            config: Evren configuration. If None, loads from environment.
        """
        self.config = config or get_config().evren
        self._client: OpenAI | None = None
    
    @property
    def client(self) -> OpenAI:
        """Get or create the OpenAI client configured for EVREN SSB API."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        return self._client

    def get_terms_status(self) -> dict[str, Any]:
        """Check terms of service status on EVREN API.
        
        Returns:
            Dictionary with terms status (e.g. current_version, accepted).
        """
        try:
            url = f"{self.config.base_url.rstrip('/')}/terms/status"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10
            )
            if resp.ok:
                return resp.json()
            return {"error": resp.text, "status_code": resp.status_code}
        except Exception as e:
            logger.warning(f"Failed to check EVREN terms status: {e}")
            return {"error": str(e)}

    def accept_terms(self, version: int | None = None) -> bool:
        """Accept EVREN API terms of service.
        
        Required once per account/key to use the API without getting 403 terms_not_accepted.
        
        Args:
            version: Terms version to accept. If None, queries /terms/status for current_version.
            
        Returns:
            True if accepted successfully, False otherwise.
        """
        try:
            base = self.config.base_url.rstrip('/')
            if version is None:
                status = self.get_terms_status()
                version = status.get("current_version", 1)
            
            url = f"{base}/terms/accept"
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                json={"version": version},
                timeout=10
            )
            if resp.ok:
                logger.info(f"Successfully accepted EVREN terms version {version}")
                return True
            else:
                logger.warning(f"Failed to accept EVREN terms: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Error accepting EVREN terms: {e}")
            return False

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry.
        
        Handles:
            - 429 Rate limiting / concurrency limits
            - 500/502/503 Service issues
            - 403 terms_not_accepted (attempts auto-acceptance)
            
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
        terms_acceptance_attempted = False
        
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle terms_not_accepted error (403)
                if ("terms_not_accepted" in error_str or "terms not accepted" in error_str or "403" in error_str) and not terms_acceptance_attempted:
                    logger.warning("EVREN API returned terms_not_accepted. Attempting auto-acceptance of terms...")
                    terms_acceptance_attempted = True
                    if self.accept_terms():
                        time.sleep(1.0)
                        continue
                
                # Handle rate limiting / quota cooldown
                if "rate" in error_str or "429" in error_str or "too many requests" in error_str or "cooldown" in error_str:
                    last_exception = e
                    logger.warning(
                        f"EVREN rate limited, attempt {attempt + 1}/{self.config.max_retries}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                elif "unavailable" in error_str or "503" in error_str or "500" in error_str or "502" in error_str:
                    last_exception = e
                    logger.warning(
                        f"EVREN service issue, attempt {attempt + 1}/{self.config.max_retries}. Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
        
        raise last_exception

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.
            temperature: Optional temperature override.
            max_tokens: Maximum tokens in response.
            
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
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens
            )
            
        response = self._retry_with_backoff(_call)
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048
    ) -> Iterator[str]:
        """Generate text with streaming.
        
        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.
            temperature: Optional temperature override.
            max_tokens: Maximum tokens in response.
            
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
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens
            )
            
        response = self._retry_with_backoff(_call)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, text: str, model: str = "qwen3-embedding-8b") -> list[float]:
        """Generate embeddings using EVREN API.
        
        Args:
            text: Text to embed.
            model: Embedding model name.
            
        Returns:
            List of floats representing the embedding vector.
        """
        def _call():
            resp = self.client.embeddings.create(input=text, model=model)
            return resp.data[0].embedding
        return self._retry_with_backoff(_call)

    def embed_batch(self, texts: list[str], model: str = "qwen3-embedding-8b") -> list[list[float]]:
        """Generate batch embeddings using EVREN API.
        
        Args:
            texts: List of texts to embed.
            model: Embedding model name.
            
        Returns:
            List of embedding vectors.
        """
        def _call():
            resp = self.client.embeddings.create(input=texts, model=model)
            return [d.embedding for d in resp.data]
        return self._retry_with_backoff(_call)

    def list_models(self) -> list[str]:
        """List available models on the EVREN platform.
        
        Returns:
            List of model IDs.
        """
        def _call():
            models = self.client.models.list()
            return [m.id for m in models.data]
        return self._retry_with_backoff(_call)


# Singleton instance
_evren_client: EvrenClient | None = None


def get_evren_client() -> EvrenClient:
    """Get the global EVREN client instance."""
    global _evren_client
    if _evren_client is None:
        _evren_client = EvrenClient()
    return _evren_client
