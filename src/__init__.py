"""Moby-Dick GraphRAG Encyclopedia - A Gemini-powered knowledge base."""

__version__ = "0.1.0"

# Core components
from .query_engine import QueryEngine, QueryResult, QueryType, get_query_engine
from .hybrid_retriever import HybridRetriever, HybridResults, get_hybrid_retriever
from .selection_layer import SelectionLayer, SelectionConfig, get_selection_layer
from .prompts import (
    RetrievedContext,
    QuoteBudget,
    SYSTEM_INSTRUCTION,
    SYNTHESIS_INSTRUCTIONS,
    build_query_prompt,
    build_synthesis_prompt,
)
from .config import (
    AppConfig,
    SelectionLayerConfig,
    QuoteBudgetConfig,
    SynthesisConfig,
    get_config,
)

__all__ = [
    # Query engine
    "QueryEngine",
    "QueryResult", 
    "QueryType",
    "get_query_engine",
    # Retrieval
    "HybridRetriever",
    "HybridResults",
    "get_hybrid_retriever",
    # Selection layer
    "SelectionLayer",
    "SelectionConfig",
    "get_selection_layer",
    # Prompts
    "RetrievedContext",
    "QuoteBudget",
    "SYSTEM_INSTRUCTION",
    "SYNTHESIS_INSTRUCTIONS",
    "build_query_prompt",
    "build_synthesis_prompt",
    # Config
    "AppConfig",
    "SelectionLayerConfig",
    "QuoteBudgetConfig",
    "SynthesisConfig",
    "get_config",
]
