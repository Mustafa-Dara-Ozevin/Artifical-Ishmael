"""Configuration module for Moby-Dick GraphRAG Encyclopedia."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class Neo4jConfig:
    """Neo4j Aura connection configuration."""
    uri: str = os.getenv("NEO4J_URI", "")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    
    def validate(self) -> bool:
        """Validate that all required Neo4j credentials are set."""
        return all([self.uri, self.user, self.password])


@dataclass
class GeminiConfig:
    """Google AI Studio Gemini API configuration."""
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def validate(self) -> bool:
        """Validate that the API key is set."""
        return bool(self.api_key) and self.api_key != "your_gemini_api_key_here"


@dataclass
class AppConfig:
    """Main application configuration."""
    neo4j: Neo4jConfig
    gemini: GeminiConfig
    
    # Retrieval settings
    max_graph_results: int = 10
    max_vector_results: int = 5
    similarity_threshold: float = 0.7
    
    # Response settings
    include_citations: bool = True
    stream_responses: bool = True


def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig(
        neo4j=Neo4jConfig(),
        gemini=GeminiConfig()
    )


def validate_config(config: AppConfig) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    if not config.neo4j.validate():
        errors.append("Neo4j credentials are not properly configured. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
    
    if not config.gemini.validate():
        errors.append("Gemini API key is not configured. Set GEMINI_API_KEY in .env")
    
    return errors
