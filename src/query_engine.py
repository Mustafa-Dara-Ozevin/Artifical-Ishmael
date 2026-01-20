"""Query engine orchestrating retrieval and generation for Moby-Dick GraphRAG."""

from typing import Any, Iterator
from dataclasses import dataclass
from enum import Enum
import logging
import re

from .hybrid_retriever import HybridRetriever, HybridResults, get_hybrid_retriever
from .gemini_client import GeminiClient, get_gemini_client
from .prompts import (
    SYSTEM_INSTRUCTION,
    build_query_prompt,
    build_character_prompt,
    build_theme_prompt,
    build_chapter_prompt,
    RetrievedContext
)

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the engine can handle."""
    GENERAL = "general"
    CHARACTER = "character"
    CHAPTER = "chapter"
    THEME = "theme"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"


@dataclass
class QueryResult:
    """Container for query results."""
    query: str
    query_type: QueryType
    answer: str
    context: HybridResults
    sources: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "answer": self.answer,
            "num_facts_used": len(self.context.facts),
            "num_analysis_used": len(self.context.analysis),
            "sources": self.sources
        }


class QueryEngine:
    """Orchestrates retrieval and generation for encyclopedia queries."""
    
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        gemini: GeminiClient | None = None,
        include_sources: bool = True
    ):
        """Initialize the query engine.
        
        Args:
            retriever: Hybrid retriever instance.
            gemini: Gemini client instance.
            include_sources: Whether to include source citations.
        """
        self.retriever = retriever or get_hybrid_retriever()
        self.gemini = gemini or get_gemini_client()
        self.include_sources = include_sources
    
    def query(
        self,
        user_query: str,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> QueryResult:
        """Process a user query and generate a response.
        
        Args:
            user_query: Natural language question.
            max_facts: Maximum fact nodes to retrieve.
            max_analysis: Maximum analysis nodes to retrieve.
            
        Returns:
            Query result with answer and sources.
        """
        logger.info(f"Processing query: {user_query}")
        
        # 1. Classify query type
        query_type = self._classify_query(user_query)
        logger.info(f"Query type: {query_type.value}")
        
        # 2. Retrieve context using specialized retrieval
        context = self._retrieve_context(
            user_query, 
            query_type, 
            max_facts, 
            max_analysis
        )
        
        # 3. Build prompt based on query type
        prompt = self._build_prompt(user_query, query_type, context)
        
        # 4. Generate response
        answer = self.gemini.generate(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7
        )
        
        # 5. Extract sources
        sources = self._extract_sources(context)
        
        return QueryResult(
            query=user_query,
            query_type=query_type,
            answer=answer,
            context=context,
            sources=sources
        )
    
    def query_stream(
        self,
        user_query: str,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> Iterator[str]:
        """Process a query with streaming response.
        
        Args:
            user_query: Natural language question.
            max_facts: Maximum fact nodes to retrieve.
            max_analysis: Maximum analysis nodes to retrieve.
            
        Yields:
            Response text chunks as they are generated.
        """
        logger.info(f"Processing query (streaming): {user_query}")
        
        # 1. Classify and retrieve
        query_type = self._classify_query(user_query)
        context = self._retrieve_context(
            user_query, 
            query_type, 
            max_facts, 
            max_analysis
        )
        
        # 2. Build prompt
        prompt = self._build_prompt(user_query, query_type, context)
        
        # 3. Stream response
        for chunk in self.gemini.generate_stream(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION
        ):
            yield chunk
    
    def ask_about_character(self, character_name: str) -> QueryResult:
        """Specialized query for character information.
        
        Args:
            character_name: Name of the character.
            
        Returns:
            Detailed character information.
        """
        query = f"Tell me about the character {character_name} in Moby-Dick."
        context = self.retriever.retrieve_for_character(character_name)
        
        prompt = build_character_prompt(
            character_name, 
            context.to_context()
        )
        
        answer = self.gemini.generate(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7
        )
        
        return QueryResult(
            query=query,
            query_type=QueryType.CHARACTER,
            answer=answer,
            context=context,
            sources=self._extract_sources(context)
        )
    
    def ask_about_chapter(self, chapter_num: int) -> QueryResult:
        """Specialized query for chapter information.
        
        Args:
            chapter_num: Chapter number.
            
        Returns:
            Detailed chapter information.
        """
        query = f"What happens in Chapter {chapter_num} of Moby-Dick?"
        context = self.retriever.retrieve_for_chapter(chapter_num)
        
        prompt = build_chapter_prompt(
            chapter_num,
            context.to_context()
        )
        
        answer = self.gemini.generate(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7
        )
        
        return QueryResult(
            query=query,
            query_type=QueryType.CHAPTER,
            answer=answer,
            context=context,
            sources=self._extract_sources(context)
        )
    
    def ask_about_theme(self, theme: str) -> QueryResult:
        """Specialized query for theme/concept information.
        
        Args:
            theme: Theme or concept name.
            
        Returns:
            Thematic analysis.
        """
        query = f"Explain the theme of {theme} in Moby-Dick."
        context = self.retriever.retrieve_for_theme(theme)
        
        prompt = build_theme_prompt(
            theme,
            context.to_context()
        )
        
        answer = self.gemini.generate(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7
        )
        
        return QueryResult(
            query=query,
            query_type=QueryType.THEME,
            answer=answer,
            context=context,
            sources=self._extract_sources(context)
        )
    
    def compare(self, entity1: str, entity2: str) -> QueryResult:
        """Compare two entities (characters, concepts, etc.).
        
        Args:
            entity1: First entity name.
            entity2: Second entity name.
            
        Returns:
            Comparative analysis.
        """
        query = f"Compare {entity1} and {entity2} in Moby-Dick."
        
        # Retrieve context for both entities
        context1 = self.retriever.retrieve(entity1, max_facts=5, max_analysis=3)
        context2 = self.retriever.retrieve(entity2, max_facts=5, max_analysis=3)
        
        # Merge contexts
        merged_context = HybridResults(
            facts=context1.facts + context2.facts,
            analysis=context1.analysis + context2.analysis,
            query=query
        )
        
        # Build comparison prompt
        prompt = self._build_comparison_prompt(
            entity1, 
            entity2, 
            merged_context.to_context()
        )
        
        answer = self.gemini.generate(
            prompt,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7
        )
        
        return QueryResult(
            query=query,
            query_type=QueryType.COMPARISON,
            answer=answer,
            context=merged_context,
            sources=self._extract_sources(merged_context)
        )
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query.
        
        Args:
            query: User query.
            
        Returns:
            Detected query type.
        """
        query_lower = query.lower()
        
        # Character patterns
        character_patterns = [
            r"who is (\w+)",
            r"tell me about (\w+)",
            r"describe (\w+)",
            r"(\w+)'s? role",
            r"character\s+(\w+)",
            r"what does (\w+) do"
        ]
        for pattern in character_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CHARACTER
        
        # Chapter patterns
        if re.search(r"chapter\s*\d+", query_lower):
            return QueryType.CHAPTER
        
        # Theme/symbol patterns
        theme_keywords = [
            "theme", "symbol", "meaning", "represent", 
            "significance", "metaphor", "allegory"
        ]
        if any(kw in query_lower for kw in theme_keywords):
            return QueryType.THEME
        
        # Comparison patterns
        comparison_keywords = [
            "compare", "contrast", "difference", "similar",
            "versus", " vs ", " and "
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            # Check for two entities
            entities = re.findall(r"(?:compare|between)\s+(\w+)\s+(?:and|with|to|vs)\s+(\w+)", query_lower)
            if entities:
                return QueryType.COMPARISON
        
        # Relationship patterns
        relationship_keywords = [
            "relationship", "connection", "interact",
            "relate", "know each other"
        ]
        if any(kw in query_lower for kw in relationship_keywords):
            return QueryType.RELATIONSHIP
        
        return QueryType.GENERAL
    
    def _retrieve_context(
        self,
        query: str,
        query_type: QueryType,
        max_facts: int,
        max_analysis: int
    ) -> HybridResults:
        """Retrieve context based on query type.
        
        Args:
            query: User query.
            query_type: Classified query type.
            max_facts: Maximum fact nodes.
            max_analysis: Maximum analysis nodes.
            
        Returns:
            Retrieved context.
        """
        query_lower = query.lower()
        
        if query_type == QueryType.CHARACTER:
            # Extract character name
            char_match = re.search(
                r"(?:who is|about|describe|character)\s+(\w+)", 
                query_lower
            )
            if char_match:
                return self.retriever.retrieve_for_character(
                    char_match.group(1),
                    max_facts=max_facts,
                    max_analysis=max_analysis
                )
        
        elif query_type == QueryType.CHAPTER:
            # Extract chapter number
            chapter_match = re.search(r"chapter\s*(\d+)", query_lower)
            if chapter_match:
                return self.retriever.retrieve_for_chapter(
                    int(chapter_match.group(1)),
                    max_facts=max_facts,
                    max_analysis=max_analysis
                )
        
        elif query_type == QueryType.THEME:
            # Use the query itself for theme retrieval
            return self.retriever.retrieve_for_theme(
                query,
                max_facts=max_facts,
                max_analysis=max_analysis
            )
        
        # Default: general hybrid retrieval
        return self.retriever.retrieve(
            query,
            max_facts=max_facts,
            max_analysis=max_analysis
        )
    
    def _build_prompt(
        self,
        query: str,
        query_type: QueryType,
        context: HybridResults
    ) -> str:
        """Build the appropriate prompt for the query type.
        
        Args:
            query: User query.
            query_type: Classified query type.
            context: Retrieved context.
            
        Returns:
            Formatted prompt string.
        """
        retrieved_context = context.to_context()
        
        if query_type == QueryType.CHARACTER:
            # Extract character name for specialized prompt
            match = re.search(r"(?:who is|about|describe)\s+(\w+)", query.lower())
            if match:
                return build_character_prompt(match.group(1), retrieved_context)
        
        elif query_type == QueryType.CHAPTER:
            match = re.search(r"chapter\s*(\d+)", query.lower())
            if match:
                return build_chapter_prompt(int(match.group(1)), retrieved_context)
        
        elif query_type == QueryType.THEME:
            return build_theme_prompt(query, retrieved_context)
        
        # Default prompt
        return build_query_prompt(retrieved_context)
    
    def _build_comparison_prompt(
        self,
        entity1: str,
        entity2: str,
        context: RetrievedContext
    ) -> str:
        """Build a comparison prompt.
        
        Args:
            entity1: First entity.
            entity2: Second entity.
            context: Retrieved context.
            
        Returns:
            Comparison prompt string.
        """
        base_prompt = build_query_prompt(context)
        
        comparison_instructions = f"""
## COMPARISON FOCUS: {entity1} vs. {entity2}

When comparing these two entities, address:
1. **Nature**: What type of entity is each? (character, concept, symbol, etc.)
2. **Similarities**: What do they share in common?
3. **Differences**: How do they differ?
4. **Relationship**: How do they interact or relate to each other?
5. **Thematic Connection**: What themes connect or contrast them?
6. **Significance**: What is the significance of comparing them?"""
        
        return base_prompt + "\n\n" + comparison_instructions
    
    def _extract_sources(self, context: HybridResults) -> list[dict[str, Any]]:
        """Extract source citations from context.
        
        Args:
            context: Retrieved context.
            
        Returns:
            List of source citations.
        """
        sources = []
        
        for result in context.facts + context.analysis:
            node = result.node
            source = {
                "type": result.node_type,
                "name": node.get("name", node.get("title", node.get("id", "Unknown"))),
                "layer": "Facts" if result.layer == 1 else "Analysis",
                "retrieval_source": result.source,
                "score": round(result.score, 3)
            }
            
            # Add chapter reference if available
            if "chapter_ref" in node:
                source["chapter"] = node["chapter_ref"]
            elif "chapter_refs" in node:
                source["chapters"] = node["chapter_refs"]
            
            sources.append(source)
        
        return sources


# Singleton instance
_engine: QueryEngine | None = None


def get_query_engine() -> QueryEngine:
    """Get or create the singleton query engine."""
    global _engine
    if _engine is None:
        _engine = QueryEngine()
    return _engine
