"""Configuration module for Moby-Dick GraphRAG Encyclopedia."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from httplib2 import URI
from neo4j import GraphDatabase

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass
class Neo4jConfig:
    """Neo4j Aura connection configuration."""
    uri: str = os.getenv("NEO4J_URI", "").strip()
    user: str = os.getenv("NEO4J_USER", "").strip()
    password: str = os.getenv("NEO4J_PASSWORD", "").strip()
    
    def validate(self) -> bool:
        """Validate that all required Neo4j credentials are set."""
        return all([self.uri, self.user, self.password])


@dataclass
class GeminiConfig:
    """Google AI Studio Gemini API configuration."""
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Quota safety
    max_embedding_retries: int = 1  # Stricter limit for embeddings to prevent exhaustion
    embedding_retry_delay: float = 0.5
    
    def validate(self) -> bool:
        """Validate that the API key is set."""
        return bool(self.api_key) and self.api_key != "your_gemini_api_key_here"


@dataclass
class GroqConfig:
    """Groq API configuration."""
    api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def validate(self) -> bool:
        """Validate that the API key is set."""
        return bool(self.api_key) and self.api_key != "your_groq_api_key_here"


@dataclass
class SelectionLayerConfig:
    """Selection layer configuration for rhetorical filtering."""
    min_grounded: int = int(os.getenv("SELECTION_MIN_GROUNDED", "1"))
    min_facts: int = int(os.getenv("SELECTION_MIN_FACTS", "2"))
    min_analysis: int = int(os.getenv("SELECTION_MIN_ANALYSIS", "1"))
    relationship_weight: float = float(os.getenv("SELECTION_RELATIONSHIP_WEIGHT", "0.3"))
    cross_layer_bonus: float = float(os.getenv("SELECTION_CROSS_LAYER_BONUS", "0.2"))
    grounded_bonus: float = float(os.getenv("SELECTION_GROUNDED_BONUS", "0.15"))


@dataclass
class QuoteBudgetConfig:
    """Quote budget configuration for limiting quotations in responses."""
    max_quotes: int = int(os.getenv("QUOTE_BUDGET_MAX", "3"))
    max_quote_length: int = int(os.getenv("QUOTE_MAX_LENGTH", "150"))
    enabled: bool = os.getenv("QUOTE_BUDGET_ENABLED", "true").lower() == "true"


@dataclass
class SynthesisConfig:
    """Synthesis mode configuration for connection-focused responses."""
    enabled: bool = os.getenv("SYNTHESIS_MODE", "true").lower() == "true"
    require_cross_layer_insight: bool = os.getenv("SYNTHESIS_REQUIRE_CROSS_LAYER", "true").lower() == "true"
    prefer_prose: bool = os.getenv("SYNTHESIS_PREFER_PROSE", "true").lower() == "true"


@dataclass
class VectorRetrieverConfig:
    """Vector retriever quota safety configuration."""
    max_fallback_candidates: int = int(os.getenv("VECTOR_MAX_FALLBACK_CANDIDATES", "15"))
    enable_quota_checks: bool = os.getenv("VECTOR_ENABLE_QUOTA_CHECKS", "true").lower() == "true"
    fallback_to_graph_on_quota_low: bool = os.getenv("VECTOR_FALLBACK_ON_QUOTA_LOW", "true").lower() == "true"

@dataclass
class AppConfig:
    """Main application configuration."""
    neo4j: Neo4jConfig
    gemini: GeminiConfig
    groq: GroqConfig
    
    # LLM Provider selection
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Retrieval settings
    max_graph_results: int = 10
    max_vector_results: int = 5
    similarity_threshold: float = 0.7
    
    # Response settings
    include_citations: bool = True
    stream_responses: bool = True
    
    # Quality improvement features
    selection_layer: SelectionLayerConfig = None
    quote_budget: QuoteBudgetConfig = None
    synthesis: SynthesisConfig = None
    
    # Vector retriever quota safety
    vector_retriever: VectorRetrieverConfig = None
    
    
    def __post_init__(self):
        """Initialize nested configs with defaults if not provided."""
        if self.selection_layer is None:
            self.selection_layer = SelectionLayerConfig()
        if self.quote_budget is None:
            self.quote_budget = QuoteBudgetConfig()
        if self.synthesis is None:
            self.synthesis = SynthesisConfig()
        if self.vector_retriever is None:
            self.vector_retriever = VectorRetrieverConfig()

def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig(
        neo4j=Neo4jConfig(),
        gemini=GeminiConfig(),
        groq=GroqConfig()
    )


def validate_config(config: AppConfig) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    if not config.neo4j.validate():
        errors.append("Neo4j credentials are not properly configured. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
    
    if config.llm_provider == "gemini":
        if not config.gemini.validate():
            errors.append("Gemini API key is not configured. Set GEMINI_API_KEY in .env")
    elif config.llm_provider == "groq":
        if not config.groq.validate():
            errors.append("Groq API key is not configured. Set GROQ_API_KEY in .env")
    else:
        errors.append(f"Unsupported LLM provider: {config.llm_provider}. Use 'gemini' or 'groq'.")
    
    return errors

if __name__ == "__main__":
    config = get_config()
    validation_errors = validate_config(config)
    if validation_errors:
        print("Configuration validation failed with the following errors:")
        for error in validation_errors:
            print(f"- {error}")
    else:
        print("Configuration is valid.")
        AUTH = (config.neo4j.user, config.neo4j.password)
        with GraphDatabase.driver(uri=config.neo4j.uri, auth=AUTH) as driver:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j Aura!")