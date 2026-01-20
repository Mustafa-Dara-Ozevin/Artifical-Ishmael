"""Hybrid retriever combining graph and vector search with re-ranking."""

from typing import Any
from dataclasses import dataclass, field
import logging
import re

from .graph_retriever import GraphRetriever, NodeLayer, get_graph_retriever
from .vector_retriever import VectorRetriever, get_vector_retriever
from .prompts import RetrievedContext

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with scoring."""
    node: dict[str, Any]
    source: str  # "graph" or "vector"
    score: float
    node_type: str
    layer: int  # 1 = facts, 2 = analysis
    
    @property
    def node_id(self) -> str:
        """Get unique identifier for deduplication."""
        return self.node.get("id", self.node.get("name", str(hash(str(self.node)))))


@dataclass
class HybridResults:
    """Container for hybrid retrieval results."""
    facts: list[RetrievalResult] = field(default_factory=list)
    analysis: list[RetrievalResult] = field(default_factory=list)
    query: str = ""
    
    def to_context(self) -> RetrievedContext:
        """Convert to RetrievedContext for prompt building."""
        return RetrievedContext(
            facts=[r.node for r in self.facts],
            analysis=[r.node for r in self.analysis],
            query=self.query
        )
    
    @property
    def total_results(self) -> int:
        return len(self.facts) + len(self.analysis)


class HybridRetriever:
    """Combines graph traversal and vector similarity for optimal retrieval."""
    
    # Node type to layer mapping
    FACT_TYPES = {"Character", "Event", "Location", "Object", "Chapter", "Glossary"}
    ANALYSIS_TYPES = {"Concept", "Symbol", "Allusion", "Commentary"}
    
    def __init__(
        self,
        graph_retriever: GraphRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        graph_weight: float = 0.6,
        vector_weight: float = 0.4
    ):
        """Initialize hybrid retriever.
        
        Args:
            graph_retriever: Graph retrieval instance.
            vector_retriever: Vector retrieval instance.
            graph_weight: Weight for graph-based results in ranking.
            vector_weight: Weight for vector-based results in ranking.
        """
        self.graph = graph_retriever or get_graph_retriever()
        self.vector = vector_retriever or get_vector_retriever()
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight
    
    def retrieve(
        self,
        query: str,
        max_facts: int = 10,
        max_analysis: int = 5,
        use_vector: bool = True
    ) -> HybridResults:
        """Perform hybrid retrieval for a query.
        
        Args:
            query: Natural language query.
            max_facts: Maximum fact nodes to return.
            max_analysis: Maximum analysis nodes to return.
            use_vector: Whether to include vector search.
            
        Returns:
            Combined and ranked results.
        """
        logger.info(f"Hybrid retrieval for: {query}")
        
        results = HybridResults(query=query)
        all_results: list[RetrievalResult] = []
        
        # Detect query type and extract entities
        query_info = self._analyze_query(query)
        
        # 1. Graph-based retrieval (structured)
        graph_results = self._graph_retrieve(query, query_info)
        all_results.extend(graph_results)
        
        # 2. Vector-based retrieval (semantic)
        if use_vector:
            vector_results = self._vector_retrieve(query)
            all_results.extend(vector_results)
        
        # 3. Deduplicate and merge scores
        merged = self._deduplicate_and_merge(all_results)
        
        # 4. Re-rank by combined score
        ranked = sorted(merged, key=lambda r: r.score, reverse=True)
        
        # 5. Separate by layer
        for result in ranked:
            if result.layer == 1 and len(results.facts) < max_facts:
                results.facts.append(result)
            elif result.layer == 2 and len(results.analysis) < max_analysis:
                results.analysis.append(result)
        
        logger.info(f"Retrieved {len(results.facts)} facts, {len(results.analysis)} analysis nodes")
        return results
    
    def retrieve_for_character(
        self,
        character_name: str,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> HybridResults:
        """Specialized retrieval for character queries.
        
        Args:
            character_name: Name of the character.
            max_facts: Maximum fact nodes.
            max_analysis: Maximum analysis nodes.
            
        Returns:
            Character-focused results.
        """
        results = HybridResults(query=f"Character: {character_name}")
        
        # Get character node with relationships
        character = self.graph.get_character(character_name)
        if character:
            result = RetrievalResult(
                node=character,
                source="graph",
                score=1.0,
                node_type="Character",
                layer=1
            )
            results.facts.append(result)
            
            # Add related characters
            relationships = self.graph.get_character_relationships(
                character_name, 
                limit=max_facts - 1
            )
            for rel in relationships:
                rel_result = RetrievalResult(
                    node={
                        "type": "Relationship",
                        "from": rel["character"],
                        "to": rel["other_name"],
                        "relationship": {
                            "type": rel["relationship_type"],
                            "context": rel.get("context"),
                            "weight": rel.get("weight", 0.5)
                        },
                        "connected_to": rel["other_name"]
                    },
                    source="graph",
                    score=rel.get("weight", 0.5),
                    node_type="Relationship",
                    layer=rel.get("layer", 1)
                )
                
                if rel_result.layer == 1 and len(results.facts) < max_facts:
                    results.facts.append(rel_result)
                elif rel_result.layer == 2 and len(results.analysis) < max_analysis:
                    results.analysis.append(rel_result)
        
        # Add semantic search for thematic connections
        semantic_results = self.vector.semantic_search(
            f"{character_name} character themes symbolism",
            layer=NodeLayer.ANALYSIS,
            limit=max_analysis
        )
        
        for node in semantic_results:
            if len(results.analysis) >= max_analysis:
                break
            result = RetrievalResult(
                node=node,
                source="vector",
                score=node.get("similarity_score", 0.5),
                node_type=node.get("type", "Unknown"),
                layer=2
            )
            results.analysis.append(result)
        
        return results
    
    def retrieve_for_chapter(
        self,
        chapter_num: int,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> HybridResults:
        """Specialized retrieval for chapter queries.
        
        Args:
            chapter_num: Chapter number.
            max_facts: Maximum fact nodes.
            max_analysis: Maximum analysis nodes.
            
        Returns:
            Chapter-focused results.
        """
        results = HybridResults(query=f"Chapter {chapter_num}")
        
        # Get chapter node
        chapter = self.graph.get_chapter(chapter_num)
        if chapter:
            result = RetrievalResult(
                node=chapter,
                source="graph",
                score=1.0,
                node_type="Chapter",
                layer=1
            )
            results.facts.append(result)
        
        # Get events in chapter
        events = self.graph.get_events_in_chapter(chapter_num)
        for event in events[:max_facts - 1]:
            result = RetrievalResult(
                node=event,
                source="graph",
                score=0.9,
                node_type="Event",
                layer=1
            )
            results.facts.append(result)
        
        # Get analysis related to chapter via semantic search
        analysis_results = self.vector.semantic_search(
            f"Chapter {chapter_num} themes analysis",
            layer=NodeLayer.ANALYSIS,
            limit=max_analysis
        )
        
        for node in analysis_results:
            result = RetrievalResult(
                node=node,
                source="vector",
                score=node.get("similarity_score", 0.5),
                node_type=node.get("type", "Unknown"),
                layer=2
            )
            results.analysis.append(result)
        
        return results
    
    def retrieve_for_theme(
        self,
        theme: str,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> HybridResults:
        """Specialized retrieval for theme/concept queries.
        
        Args:
            theme: Theme or concept name.
            max_facts: Maximum fact nodes.
            max_analysis: Maximum analysis nodes.
            
        Returns:
            Theme-focused results.
        """
        results = HybridResults(query=f"Theme: {theme}")
        
        # Get concept node
        concept = self.graph.get_concept(theme)
        if concept:
            result = RetrievalResult(
                node=concept,
                source="graph",
                score=1.0,
                node_type="Concept",
                layer=2
            )
            results.analysis.append(result)
        
        # Get symbol
        symbol = self.graph.get_symbol(theme)
        if symbol:
            result = RetrievalResult(
                node=symbol,
                source="graph",
                score=0.95,
                node_type="Symbol",
                layer=2
            )
            results.analysis.append(result)
        
        # Semantic search for related analysis
        semantic_analysis = self.vector.semantic_search(
            theme,
            layer=NodeLayer.ANALYSIS,
            limit=max_analysis
        )
        
        for node in semantic_analysis:
            if len(results.analysis) >= max_analysis:
                break
            # Skip if already added
            if node.get("id") == (concept or {}).get("id"):
                continue
            if node.get("id") == (symbol or {}).get("id"):
                continue
            
            result = RetrievalResult(
                node=node,
                source="vector",
                score=node.get("similarity_score", 0.5),
                node_type=node.get("type", "Unknown"),
                layer=2
            )
            results.analysis.append(result)
        
        # Get characters that embody the theme
        semantic_facts = self.vector.semantic_search(
            f"{theme} character example",
            layer=NodeLayer.FACTS,
            limit=max_facts
        )
        
        for node in semantic_facts:
            result = RetrievalResult(
                node=node,
                source="vector",
                score=node.get("similarity_score", 0.5),
                node_type=node.get("type", "Unknown"),
                layer=1
            )
            results.facts.append(result)
        
        return results
    
    def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query to extract entities and intent.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary with extracted information.
        """
        query_lower = query.lower()
        
        info = {
            "is_character_query": False,
            "is_chapter_query": False,
            "is_theme_query": False,
            "is_symbol_query": False,
            "is_relationship_query": False,
            "entities": [],
            "chapter_num": None
        }
        
        # Detect character names from the query (look for capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        # Known character names
        known_characters = {'ishmael', 'queequeg', 'ahab', 'starbuck', 'stubb', 
                           'flask', 'tashtego', 'daggoo', 'pip', 'fedallah',
                           'elijah', 'mapple', 'bildad', 'peleg', 'bulkington'}
        for pn in proper_nouns:
            if pn.lower() in known_characters:
                info["entities"].append(pn.lower())
        
        # Detect character queries
        character_patterns = [
            r"who is (\w+)",
            r"tell me about (\w+)",
            r"(\w+)'s? (?:role|character|personality)",
            r"character (\w+)"
        ]
        for pattern in character_patterns:
            match = re.search(pattern, query_lower)
            if match:
                info["is_character_query"] = True
                info["entities"].append(match.group(1))
        
        # Detect relationship queries (between characters)
        relationship_patterns = [
            r"relationship between (\w+) and (\w+)",
            r"(\w+) and (\w+)(?:'s)? relationship",
            r"(\w+)(?:'s)? (?:relationship|friendship|bond) with (\w+)"
        ]
        for pattern in relationship_patterns:
            match = re.search(pattern, query_lower)
            if match:
                info["is_relationship_query"] = True
                info["entities"].extend([match.group(1), match.group(2)])
        
        # Detect chapter queries
        chapter_match = re.search(r"chapter\s*(\d+)", query_lower)
        if chapter_match:
            info["is_chapter_query"] = True
            info["chapter_num"] = int(chapter_match.group(1))
        
        # Detect theme/symbol queries - expanded with more abstract concept keywords
        theme_keywords = ["theme", "symbol", "meaning", "represents", "symbolize", "significance",
                         "concept", "idea", "otherness", "self", "identity", "knowledge",
                         "prejudice", "fear", "race", "friendship", "bond", "soul",
                         "meditation", "melancholy", "isolation", "fate", "obsession"]
        if any(kw in query_lower for kw in theme_keywords):
            info["is_theme_query"] = True
        
        symbol_keywords = ["whale", "sea", "whiteness", "leg", "coffin", "harpoon"]
        if any(kw in query_lower for kw in symbol_keywords):
            info["is_symbol_query"] = True
        
        # Deduplicate entities
        info["entities"] = list(set(info["entities"]))
        
        return info
    
    def _graph_retrieve(
        self,
        query: str,
        query_info: dict[str, Any]
    ) -> list[RetrievalResult]:
        """Perform graph-based retrieval.
        
        Args:
            query: User query.
            query_info: Analyzed query information.
            
        Returns:
            List of retrieval results from graph.
        """
        results = []
        
        # Use specialized retrievers based on query type
        if query_info["is_chapter_query"] and query_info["chapter_num"]:
            chapter = self.graph.get_chapter(query_info["chapter_num"])
            if chapter:
                results.append(RetrievalResult(
                    node=chapter,
                    source="graph",
                    score=1.0,
                    node_type="Chapter",
                    layer=1
                ))
            
            events = self.graph.get_events_in_chapter(query_info["chapter_num"])
            for event in events:
                results.append(RetrievalResult(
                    node=event,
                    source="graph",
                    score=0.9,
                    node_type="Event",
                    layer=1
                ))
        
        # Search by extracted entities - get characters
        entities = query_info.get("entities", [])
        for entity in entities:
            char = self.graph.get_character(entity)
            if char:
                results.append(RetrievalResult(
                    node=char,
                    source="graph",
                    score=1.0,
                    node_type="Character",
                    layer=1
                ))
        
        # For relationship queries between characters, get allusions mentioning both characters
        if query_info.get("is_relationship_query") and len(entities) >= 2:
            # Get allusions for each character and find ones mentioning both
            for entity in entities[:2]:
                allusion_results = self.graph.get_allusions_for_character(entity, limit=10)
                for node in allusion_results:
                    # Check if the other entity is also mentioned in description
                    desc = str(node.get("description", "")).lower()
                    other_entities = [e for e in entities if e != entity]
                    if any(other.lower() in desc for other in other_entities):
                        results.append(RetrievalResult(
                            node=node,
                            source="graph",
                            score=0.98,  # Very high score for allusions mentioning both characters
                            node_type=node.get("type", "Allusion"),
                            layer=2
                        ))
                    else:
                        # Still include allusions about individual characters
                        results.append(RetrievalResult(
                            node=node,
                            source="graph",
                            score=0.85,
                            node_type=node.get("type", "Allusion"),
                            layer=2
                        ))
        # For queries mentioning characters (even if not explicitly relationship)
        elif entities:
            for entity in entities:
                allusion_results = self.graph.get_allusions_for_character(entity, limit=5)
                for node in allusion_results:
                    results.append(RetrievalResult(
                        node=node,
                        source="graph",
                        score=0.9,  # High score for character-related allusions
                        node_type=node.get("type", "Allusion"),
                        layer=2
                    ))
        
        # For theme queries, search concepts and allusions specifically
        if query_info.get("is_theme_query"):
            # Search analysis layer for abstract concepts
            analysis_results = self.graph.fulltext_search(query, layer=NodeLayer.ANALYSIS, limit=8)
            for node in analysis_results:
                results.append(RetrievalResult(
                    node=node,
                    source="graph",
                    score=0.85,  # Good score for theme-related analysis
                    node_type=node.get("type", "Unknown"),
                    layer=2
                ))
        
        # General fulltext search
        fulltext_results = self.graph.fulltext_search(query, limit=10)
        for node in fulltext_results:
            node_type = node.get("type", "Unknown")
            layer = 1 if node_type in self.FACT_TYPES else 2
            
            results.append(RetrievalResult(
                node=node,
                source="graph",
                score=0.7,  # Lower score for fulltext matches
                node_type=node_type,
                layer=layer
            ))
        
        return results
    
    def _vector_retrieve(self, query: str) -> list[RetrievalResult]:
        """Perform vector-based retrieval.
        
        Args:
            query: User query.
            
        Returns:
            List of retrieval results from vector search.
        """
        results = []
        
        # Search both layers
        for layer in [NodeLayer.FACTS, NodeLayer.ANALYSIS]:
            layer_num = 1 if layer == NodeLayer.FACTS else 2
            
            semantic_results = self.vector.semantic_search(
                query,
                layer=layer,
                limit=5
            )
            
            for node in semantic_results:
                results.append(RetrievalResult(
                    node=node,
                    source="vector",
                    score=node.get("similarity_score", 0.5),
                    node_type=node.get("type", "Unknown"),
                    layer=layer_num
                ))
        
        return results
    
    def _deduplicate_and_merge(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Deduplicate results and merge scores.
        
        Args:
            results: All retrieval results.
            
        Returns:
            Deduplicated results with merged scores.
        """
        seen: dict[str, RetrievalResult] = {}
        
        for result in results:
            node_id = result.node_id
            
            if node_id in seen:
                # Merge scores using weighted combination
                existing = seen[node_id]
                
                if result.source == "graph":
                    new_score = (
                        existing.score * self.vector_weight +
                        result.score * self.graph_weight
                    )
                else:
                    new_score = (
                        existing.score * self.graph_weight +
                        result.score * self.vector_weight
                    )
                
                # Boost for appearing in multiple sources
                new_score = min(new_score * 1.2, 1.0)
                
                seen[node_id] = RetrievalResult(
                    node=result.node,
                    source="hybrid",
                    score=new_score,
                    node_type=result.node_type,
                    layer=result.layer
                )
            else:
                seen[node_id] = result
        
        return list(seen.values())


# Singleton instance
_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the singleton hybrid retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
