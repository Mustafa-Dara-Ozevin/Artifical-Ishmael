"""Configuration module for Moby-Dick GraphRAG Encyclopedia."""

import os
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Helpers for configuration cleaning
def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().strip("'\"").strip()

def _clean_uri(val: Any) -> str:
    val = _clean_str(val)
    return val.rstrip("/")

# Bridge Streamlit Cloud secrets to os.environ if available
try:
    import streamlit as st
    from collections.abc import Mapping
    if hasattr(st, "secrets"):
        def _bridge_secrets(mapping: Mapping, prefix: str = ""):
            for k, v in mapping.items():
                if isinstance(v, Mapping):
                    _bridge_secrets(v, prefix=f"{k}_")
                    sec = k.lower()
                    if sec == "neo4j":
                        for sk, sv in v.items():
                            clean_sk = sk.lower().removeprefix("neo4j_").removeprefix("neo4j.")
                            if clean_sk in ("uri", "url"):
                                os.environ.setdefault("NEO4J_URI", _clean_uri(sv))
                            elif clean_sk in ("user", "username"):
                                os.environ.setdefault("NEO4J_USER", _clean_str(sv))
                            elif clean_sk in ("password", "pass"):
                                os.environ.setdefault("NEO4J_PASSWORD", _clean_str(sv))
                            elif clean_sk in ("database", "db"):
                                os.environ.setdefault("NEO4J_DATABASE", _clean_str(sv))
                    elif sec == "evren":
                        for sk, sv in v.items():
                            clean_sk = sk.lower().removeprefix("evren_")
                            if clean_sk in ("api_key", "key"):
                                os.environ.setdefault("EVREN_API_KEY", _clean_str(sv))
                            elif clean_sk in ("base_url", "url"):
                                os.environ.setdefault("EVREN_BASE_URL", _clean_str(sv))
                            elif clean_sk == "model":
                                os.environ.setdefault("EVREN_MODEL", _clean_str(sv))
                    elif sec == "gemini":
                        for sk, sv in v.items():
                            clean_sk = sk.lower().removeprefix("gemini_")
                            if clean_sk in ("api_key", "key"):
                                os.environ.setdefault("GEMINI_API_KEY", _clean_str(sv))
                    elif sec == "groq":
                        for sk, sv in v.items():
                            clean_sk = sk.lower().removeprefix("groq_")
                            if clean_sk in ("api_key", "key"):
                                os.environ.setdefault("GROQ_API_KEY", _clean_str(sv))
                elif isinstance(v, (str, int, float, bool)):
                    clean_val = _clean_str(v)
                    key_upper = k.upper()
                    os.environ.setdefault(key_upper, clean_val)
                    if prefix:
                        os.environ.setdefault(f"{prefix}{k}".upper(), clean_val)

        _bridge_secrets(st.secrets)
except Exception:
    pass


@dataclass
class Neo4jConfig:
    """Neo4j Aura connection configuration."""
    uri: str = _clean_uri(os.getenv("NEO4J_URI", ""))
    user: str = _clean_str(os.getenv("NEO4J_USER", ""))
    password: str = _clean_str(os.getenv("NEO4J_PASSWORD", ""))
    database: str | None = os.getenv("NEO4J_DATABASE", None)
    
    def validate(self) -> bool:
        """Validate that all required Neo4j credentials are set."""
        if not all([self.uri, self.user, self.password]):
            return False
        if any(p in self.uri.lower() for p in ("your_neo4j", "<instance-id>", "<dbid>", "your_uri")):
            return False
        return True


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
class EvrenConfig:
    """Cumhurbaşkanlığı Savunma Sanayii Başkanlığı (SSB) EVREN API configuration."""
    api_key: str = os.getenv("EVREN_API_KEY", "").strip()
    base_url: str = os.getenv("EVREN_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1").strip()
    model: str = os.getenv("EVREN_MODEL", "deepseek-v4-flash").strip()
    
    # Rate limiting & retry
    max_retries: int = 3
    retry_delay: float = 2.0
    
    def validate(self) -> bool:
        """Validate that the EVREN API key is set."""
        return bool(self.api_key) and self.api_key != "your_evren_api_key_here"


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
    evren: EvrenConfig = None
    
    # LLM Provider selection ('evren', 'groq', 'gemini')
    llm_provider: str = os.getenv("LLM_PROVIDER", "evren").lower()
    
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
        if self.evren is None:
            self.evren = EvrenConfig()
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
        groq=GroqConfig(),
        evren=EvrenConfig()
    )


def validate_config(config: AppConfig) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    if not config.neo4j.validate():
        errors.append("Neo4j credentials are not properly configured. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env or Streamlit Secrets.")
    elif not any(config.neo4j.uri.startswith(scheme) for scheme in ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")):
        errors.append(f"Invalid NEO4J_URI scheme: '{config.neo4j.uri}'. It must start with 'neo4j+s://' (recommended for Neo4j Aura) or 'bolt+s://'.")
    
    if config.llm_provider == "evren":
        if not config.evren.validate():
            errors.append("Evren API key is not configured. Set EVREN_API_KEY in .env or Streamlit Secrets.")
    elif config.llm_provider == "gemini":
        if not config.gemini.validate():
            errors.append("Gemini API key is not configured. Set GEMINI_API_KEY in .env or Streamlit Secrets.")
    elif config.llm_provider == "groq":
        if not config.groq.validate():
            errors.append("Groq API key is not configured. Set GROQ_API_KEY in .env or Streamlit Secrets.")
    else:
        errors.append(f"Unsupported LLM provider: {config.llm_provider}. Use 'evren', 'groq', or 'gemini'.")
    
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