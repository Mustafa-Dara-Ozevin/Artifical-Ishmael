"""Graph-based retriever using Cypher queries for Moby-Dick knowledge graph."""

from typing import Any
from enum import Enum
import logging

from .neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)


class NodeLayer(Enum):
    """Knowledge graph layers."""
    FACTS = 1  # Characters, Events, Locations, Objects, Chapters
    ANALYSIS = 2  # Concepts, Symbols, Allusions, Commentary
    ALL = 0


class GraphRetriever:
    """Retrieves context from Neo4j using structured Cypher queries."""
    
    # Node labels by layer
    FACT_LABELS = ["Character", "Event", "Location", "Object", "Chapter", "Glossary"]
    ANALYSIS_LABELS = ["Concept", "Symbol", "Allusion", "Commentary"]
    
    def __init__(self, client: Neo4jClient | None = None):
        """Initialize the graph retriever.
        
        Args:
            client: Neo4j client instance. If None, uses singleton.
        """
        self.client = client or get_neo4j_client()
    
    def _get_labels_for_layer(self, layer: NodeLayer) -> list[str]:
        """Get node labels for a specific layer.
        
        Args:
            layer: The layer to get labels for.
            
        Returns:
            List of node label strings.
        """
        if layer == NodeLayer.FACTS:
            return self.FACT_LABELS
        elif layer == NodeLayer.ANALYSIS:
            return self.ANALYSIS_LABELS
        else:  # NodeLayer.ALL
            return self.FACT_LABELS + self.ANALYSIS_LABELS
    
    def _format_nodes(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format Neo4j query results into standardized node dictionaries.
        
        Args:
            results: Raw query results from Neo4j.
            
        Returns:
            List of formatted node dictionaries.
        """
        formatted = []
        for record in results:
            node = record.get("n") or record
            labels = record.get("labels", [])
            
            # Convert Neo4j node to dict if needed
            if hasattr(node, "items"):
                node_dict = dict(node)
            else:
                node_dict = node if isinstance(node, dict) else {}
            
            # Add type from labels
            if labels:
                node_dict["type"] = labels[0] if isinstance(labels, list) else labels
            elif "type" not in node_dict:
                node_dict["type"] = "Unknown"
            
            # Ensure we have a name field
            if "name" not in node_dict:
                node_dict["name"] = node_dict.get("title") or node_dict.get("term") or node_dict.get("id", "Unknown")
            
            formatted.append(node_dict)
        
        return formatted
    
    def search_by_name(
        self,
        name: str,
        layer: NodeLayer = NodeLayer.ALL,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search for nodes by name or title.
        
        Args:
            name: Name or partial name to search for.
            layer: Which layer to search in.
            limit: Maximum results to return.
            
        Returns:
            List of matching nodes with their properties.
        """
        labels = self._get_labels_for_layer(layer)
        label_filter = "|".join(labels) if labels else ""
        
        query = f"""
        MATCH (n{':' + label_filter if label_filter else ''})
        WHERE toLower(n.name) CONTAINS toLower($name)
           OR toLower(n.title) CONTAINS toLower($name)
           OR toLower(n.id) CONTAINS toLower($name)
        RETURN n, labels(n) as labels
        LIMIT $limit
        """
        
        results = self.client.execute_query(query, {"name": name, "limit": limit})
        return self._format_nodes(results)
    
    def get_character(self, name: str) -> dict[str, Any] | None:
        """Get detailed character information.
        
        Args:
            name: Character name to search for.
            
        Returns:
            Character data with relationships, or None if not found.
        """
        query = """
        MATCH (c:Character)
        WHERE toLower(c.name) CONTAINS toLower($name)
           OR toLower(c.id) CONTAINS toLower($name)
        OPTIONAL MATCH (c)-[r]-(other)
        RETURN c, 
               collect(DISTINCT {
                   type: type(r),
                   direction: CASE WHEN startNode(r) = c THEN 'outgoing' ELSE 'incoming' END,
                   other_name: coalesce(other.name, other.title, other.id),
                   other_labels: labels(other),
                   context: r.context,
                   weight: r.weight,
                   layer: r.layer
               }) as relationships
        LIMIT 1
        """
        
        results = self.client.execute_query(query, {"name": name})
        if not results:
            return None
        
        result = results[0]
        char_data = dict(result["c"])
        char_data["type"] = "Character"
        char_data["relationships"] = [
            r for r in result["relationships"] 
            if r["other_name"] is not None
        ]
        return char_data
    
    def get_character_relationships(
        self,
        name: str,
        relationship_types: list[str] | None = None,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get character relationships with context.
        
        Args:
            name: Character name.
            relationship_types: Optional filter for relationship types.
            limit: Maximum relationships to return.
            
        Returns:
            List of relationships with context.
        """
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH (c:Character)-[r{rel_filter}]-(other)
        WHERE toLower(c.name) CONTAINS toLower($name)
        RETURN c.name as character,
               type(r) as relationship_type,
               CASE WHEN startNode(r) = c THEN 'outgoing' ELSE 'incoming' END as direction,
               coalesce(other.name, other.title, other.id) as other_name,
               labels(other) as other_labels,
               r.context as context,
               r.weight as weight,
               r.layer as layer
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        
        return self.client.execute_query(query, {"name": name, "limit": limit})
    
    def get_chapter(self, chapter_num: int) -> dict[str, Any] | None:
        """Get chapter information with related elements.
        
        Args:
            chapter_num: Chapter number.
            
        Returns:
            Chapter data with related nodes.
        """
        query = """
        MATCH (ch:Chapter {number: $chapter_num})
        OPTIONAL MATCH (ch)-[r]-(related)
        RETURN ch,
               collect(DISTINCT {
                   type: type(r),
                   name: coalesce(related.name, related.title, related.id),
                   labels: labels(related),
                   context: r.context,
                   weight: r.weight
               }) as related_elements
        LIMIT 1
        """
        
        results = self.client.execute_query(query, {"chapter_num": chapter_num})
        if not results:
            # Try with string id
            query_by_id = """
            MATCH (ch:Chapter)
            WHERE ch.id = $chapter_id OR ch.number = $chapter_num
            OPTIONAL MATCH (ch)-[r]-(related)
            RETURN ch,
                   collect(DISTINCT {
                       type: type(r),
                       name: coalesce(related.name, related.title, related.id),
                       labels: labels(related),
                       context: r.context,
                       weight: r.weight
                   }) as related_elements
            LIMIT 1
            """
            results = self.client.execute_query(
                query_by_id, 
                {"chapter_id": f"ch_{chapter_num}", "chapter_num": chapter_num}
            )
            if not results:
                return None
        
        result = results[0]
        chapter_data = dict(result["ch"])
        chapter_data["type"] = "Chapter"
        chapter_data["related_elements"] = [
            r for r in result["related_elements"]
            if r["name"] is not None
        ]
        return chapter_data
    
    def get_concept(self, concept_name: str) -> dict[str, Any] | None:
        """Get concept/theme information.
        
        Args:
            concept_name: Name of the concept.
            
        Returns:
            Concept data with manifestations.
        """
        query = """
        MATCH (con:Concept)
        WHERE toLower(con.name) CONTAINS toLower($name)
           OR toLower(con.id) CONTAINS toLower($name)
        OPTIONAL MATCH (con)-[r]-(related)
        RETURN con,
               collect(DISTINCT {
                   type: type(r),
                   name: coalesce(related.name, related.title, related.id),
                   labels: labels(related),
                   context: r.context,
                   weight: r.weight
               }) as manifestations
        LIMIT 1
        """
        
        results = self.client.execute_query(query, {"name": concept_name})
        if not results:
            return None
        
        result = results[0]
        concept_data = dict(result["con"])
        concept_data["type"] = "Concept"
        concept_data["manifestations"] = [
            m for m in result["manifestations"]
            if m["name"] is not None
        ]
        return concept_data
    
    def get_symbol(self, symbol_name: str) -> dict[str, Any] | None:
        """Get symbol information.
        
        Args:
            symbol_name: Name of the symbol.
            
        Returns:
            Symbol data with associations.
        """
        query = """
        MATCH (sym:Symbol)
        WHERE toLower(sym.name) CONTAINS toLower($name)
           OR toLower(sym.id) CONTAINS toLower($name)
        OPTIONAL MATCH (sym)-[r]-(related)
        RETURN sym,
               collect(DISTINCT {
                   type: type(r),
                   name: coalesce(related.name, related.title, related.id),
                   labels: labels(related),
                   context: r.context
               }) as associations
        LIMIT 1
        """
        
        results = self.client.execute_query(query, {"name": symbol_name})
        if not results:
            return None
        
        result = results[0]
        symbol_data = dict(result["sym"])
        symbol_data["type"] = "Symbol"
        symbol_data["associations"] = [
            a for a in result["associations"]
            if a["name"] is not None
        ]
        return symbol_data
    
    def get_allusions(self, source: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Get allusions, optionally filtered by source work.
        
        Args:
            source: Optional source work filter (e.g., "Bible", "Homer").
            limit: Maximum results.
            
        Returns:
            List of allusions.
        """
        if source:
            query = """
            MATCH (a:Allusion)
            WHERE toLower(a.source_work) CONTAINS toLower($source)
            OPTIONAL MATCH (a)-[r]-(related)
            RETURN a,
                   collect(DISTINCT {
                       type: type(r),
                       name: coalesce(related.name, related.title, related.id),
                       labels: labels(related)
                   }) as connections
            LIMIT $limit
            """
            results = self.client.execute_query(query, {"source": source, "limit": limit})
        else:
            query = """
            MATCH (a:Allusion)
            OPTIONAL MATCH (a)-[r]-(related)
            RETURN a,
                   collect(DISTINCT {
                       type: type(r),
                       name: coalesce(related.name, related.title, related.id),
                       labels: labels(related)
                   }) as connections
            LIMIT $limit
            """
            results = self.client.execute_query(query, {"limit": limit})
        
        allusions = []
        for result in results:
            allusion = dict(result["a"])
            allusion["type"] = "Allusion"
            allusion["connections"] = [
                c for c in result["connections"]
                if c["name"] is not None
            ]
            allusions.append(allusion)
        
        return allusions
    
    def get_events_in_chapter(self, chapter_num: int) -> list[dict[str, Any]]:
        """Get events that occur in a specific chapter.
        
        Args:
            chapter_num: Chapter number.
            
        Returns:
            List of events with details.
        """
        query = """
        MATCH (e:Event)
        WHERE e.chapter_ref = $chapter_num 
           OR e.chapter = $chapter_num
           OR e.chapter_ref = toString($chapter_num)
        RETURN e
        ORDER BY e.order
        """
        
        results = self.client.execute_query(query, {"chapter_num": chapter_num})
        return [{"type": "Event", **dict(r["e"])} for r in results]
    
    def find_path_between(
        self,
        name1: str,
        name2: str,
        max_hops: int = 3
    ) -> list[dict[str, Any]]:
        """Find relationship paths between two nodes.
        
        Args:
            name1: First node name.
            name2: Second node name.
            max_hops: Maximum path length.
            
        Returns:
            List of paths with nodes and relationships.
        """
        query = f"""
        MATCH (a), (b)
        WHERE (toLower(a.name) CONTAINS toLower($name1) OR toLower(a.id) CONTAINS toLower($name1))
          AND (toLower(b.name) CONTAINS toLower($name2) OR toLower(b.id) CONTAINS toLower($name2))
        MATCH path = shortestPath((a)-[*1..{max_hops}]-(b))
        RETURN [n IN nodes(path) | {{
            name: coalesce(n.name, n.title, n.id),
            labels: labels(n)
        }}] as nodes,
        [r IN relationships(path) | {{
            type: type(r),
            context: r.context
        }}] as relationships
        LIMIT 5
        """
        
        return self.client.execute_query(query, {"name1": name1, "name2": name2})
    
    def get_by_layer(
        self,
        layer: NodeLayer,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get nodes from a specific layer.
        
        Args:
            layer: Layer to retrieve from.
            limit: Maximum results.
            
        Returns:
            List of nodes from the specified layer.
        """
        labels = self._get_labels_for_layer(layer)
        if not labels:
            return []
        
        label_filter = "|".join(labels)
        query = f"""
        MATCH (n:{label_filter})
        RETURN n, labels(n) as labels
        LIMIT $limit
        """
        
        results = self.client.execute_query(query, {"limit": limit})
        return self._format_nodes(results)
    
    def fulltext_search(
        self,
        query_text: str,
        layer: NodeLayer = NodeLayer.ALL,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Perform fulltext search across node properties.
        
        Args:
            query_text: Text to search for.
            layer: Layer filter.
            limit: Maximum results.
            
        Returns:
            List of matching nodes.
        """
        labels = self._get_labels_for_layer(layer)
        label_filter = "|".join(labels) if labels else ""
        
        # Extract meaningful words from query (remove stop words AND query noise words)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'until', 'while', 'about', 'against', 'what', 'which', 'who',
                      'whom', 'this', 'that', 'these', 'those', 'am', 'i', 'me',
                      'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'he',
                      'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
                      # Query noise words - common in questions but not useful for search
                      'explain', 'describe', 'tell', 'show', 'give', 'find',
                      'discuss', 'analyze', 'compare', 'contrast', 'list',
                      'concept', 'idea', 'example', 'examples', 'using', 'use',
                      'related', 'relationship', 'between', 'does', 'mean',
                      'significance', 'important', 'importance', 'role', 'book',
                      'novel', 'story', 'text', 'passage', 'chapter'}
        
        import re
        
        # First extract proper nouns (capitalized words) - these are likely entity names
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query_text)
        proper_nouns_lower = [w.lower() for w in proper_nouns]
        
        # Then extract all words
        words = re.findall(r'\b\w+\b', query_text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            keywords = words[:3]  # Fallback to first 3 words
        
        # Prioritize proper nouns (entity names) at the front
        prioritized_keywords = []
        for pn in proper_nouns_lower:
            if pn in keywords and pn not in prioritized_keywords:
                prioritized_keywords.append(pn)
        # Add remaining keywords
        for kw in keywords:
            if kw not in prioritized_keywords:
                prioritized_keywords.append(kw)
        
        keywords = prioritized_keywords[:8]  # Increased limit to 8 for richer queries
        
        logger.debug(f"Fulltext search keywords: {keywords}")
        
        # Build OR conditions for each keyword - also search id field for allusions etc.
        conditions = []
        for i, kw in enumerate(keywords):
            param = f"kw{i}"
            conditions.append(f"""
                (toLower(n.description) CONTAINS toLower(${param})
                 OR toLower(n.definition) CONTAINS toLower(${param})
                 OR toLower(n.backstory) CONTAINS toLower(${param})
                 OR toLower(n.synopsis) CONTAINS toLower(${param})
                 OR toLower(n.name) CONTAINS toLower(${param})
                 OR toLower(n.title) CONTAINS toLower(${param})
                 OR toLower(n.term) CONTAINS toLower(${param})
                 OR toLower(n.id) CONTAINS toLower(${param}))
            """)
        
        where_clause = " OR ".join(conditions)
        
        # Build relevance scoring - prioritize name/term/id matches and definition matches
        score_parts = []
        for i, kw in enumerate(keywords):
            param = f"kw{i}"
            score_parts.append(f"""
                CASE WHEN toLower(n.name) CONTAINS toLower(${param}) THEN 3 ELSE 0 END +
                CASE WHEN toLower(n.term) CONTAINS toLower(${param}) THEN 3 ELSE 0 END +
                CASE WHEN toLower(n.id) CONTAINS toLower(${param}) THEN 2 ELSE 0 END +
                CASE WHEN toLower(n.definition) CONTAINS toLower(${param}) THEN 2 ELSE 0 END +
                CASE WHEN toLower(n.description) CONTAINS toLower(${param}) THEN 1 ELSE 0 END
            """)
        relevance_score = " + ".join(score_parts)
        
        # Retrieve more results initially, then limit after sorting
        query = f"""
        MATCH (n{':' + label_filter if label_filter else ''})
        WHERE {where_clause}
        WITH n, labels(n) as labels, ({relevance_score}) as relevance
        ORDER BY relevance DESC
        LIMIT $limit
        RETURN n, labels
        """
        
        params = {"limit": limit * 3}  # Fetch more to allow for better sorting
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = kw
        
        results = self.client.execute_query(query, params)
        return self._format_nodes(results)[:limit]  # Apply final limit
    
    def search_glossary(self, term: str) -> list[dict[str, Any]]:
        """Search glossary entries by term or definition."""
        query = """
        MATCH (g:Glossary)
        WHERE toLower(g.term) CONTAINS toLower($term)
           OR toLower(g.definition) CONTAINS toLower($term)
        RETURN g.id as id, 'Glossary' as type, g.term as name,
               g.definition as description, g.category as category,
               1 as layer, 1.0 as weight
        LIMIT 10
        """
        return self.neo4j.execute_query(query, {"term": term})

    def search_all(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search across all node types including glossary."""
        search_query = """
        // Characters
        MATCH (n:Character)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.description) CONTAINS toLower($query)
        RETURN n.id as id, 'Character' as type, n.name as name,
               n.description as description, 1 as layer, 
               CASE WHEN toLower(n.name) CONTAINS toLower($query) THEN 1.0 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Locations
        MATCH (n:Location)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.description) CONTAINS toLower($query)
        RETURN n.id as id, 'Location' as type, n.name as name,
               n.description as description, 1 as layer,
               CASE WHEN toLower(n.name) CONTAINS toLower($query) THEN 1.0 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Concepts
        MATCH (n:Concept)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.definition) CONTAINS toLower($query)
        RETURN n.id as id, 'Concept' as type, n.name as name,
               n.definition as description, 2 as layer,
               CASE WHEN toLower(n.name) CONTAINS toLower($query) THEN 1.0 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Symbols
        MATCH (n:Symbol)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.meaning) CONTAINS toLower($query)
        RETURN n.id as id, 'Symbol' as type, n.name as name,
               n.meaning as description, 2 as layer,
               CASE WHEN toLower(n.name) CONTAINS toLower($query) THEN 1.0 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Themes
        MATCH (n:Theme)
        WHERE toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.description) CONTAINS toLower($query)
        RETURN n.id as id, 'Theme' as type, n.name as name,
               n.description as description, 2 as layer,
               CASE WHEN toLower(n.name) CONTAINS toLower($query) THEN 1.0 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Allusions - search description field since many have name=null
        MATCH (n:Allusion)
        WHERE toLower(n.description) CONTAINS toLower($query)
           OR toLower(n.name) CONTAINS toLower($query)
           OR toLower(n.id) CONTAINS toLower($query)
           OR toLower(n.source_text) CONTAINS toLower($query)
           OR toLower(n.reference) CONTAINS toLower($query)
        RETURN n.id as id, 'Allusion' as type, 
               coalesce(n.name, n.id) as name,
               n.description as description, 2 as layer,
               CASE WHEN toLower(n.description) CONTAINS toLower($query) THEN 0.9 ELSE 0.7 END as relevance
        
        UNION ALL
        
        // Glossary - NEW
        MATCH (n:Glossary)
        WHERE toLower(n.term) CONTAINS toLower($query)
           OR toLower(n.definition) CONTAINS toLower($query)
        RETURN n.id as id, 'Glossary' as type, n.term as name,
               n.definition as description, 1 as layer,
               CASE WHEN toLower(n.term) CONTAINS toLower($query) THEN 1.0 ELSE 0.8 END as relevance
        
        UNION ALL
        
        // Chapters
        MATCH (n:Chapter)
        WHERE toLower(n.title) CONTAINS toLower($query)
           OR toLower(n.summary) CONTAINS toLower($query)
        RETURN n.id as id, 'Chapter' as type, n.title as name,
               n.summary as description, 1 as layer,
               CASE WHEN toLower(n.title) CONTAINS toLower($query) THEN 1.0 ELSE 0.6 END as relevance
        """
        
        results = self.neo4j.execute_query(search_query, {"query": query})
        # Sort by relevance and limit
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return results[:limit]

    def get_allusions_for_character(self, character_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get allusions that mention a specific character in their description.
        
        This is important because many allusions have name=null but contain
        valuable character analysis in their descriptions.
        
        Args:
            character_name: Character name to search for.
            limit: Maximum results.
            
        Returns:
            List of allusions mentioning the character.
        """
        query = """
        MATCH (a:Allusion)
        WHERE toLower(a.description) CONTAINS toLower($name)
        RETURN a as n, labels(a) as labels
        ORDER BY size(a.description) DESC
        LIMIT $limit
        """
        
        results = self.client.execute_query(query, {"name": character_name, "limit": limit})
        return self._format_nodes(results)


# Singleton instance
_graph_retriever: GraphRetriever | None = None


def get_graph_retriever() -> GraphRetriever:
    """Get or create the singleton GraphRetriever instance."""
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever
