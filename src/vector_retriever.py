"""Vector-based semantic retriever using Gemini embeddings and Neo4j."""

from typing import Any
import logging

from .neo4j_client import Neo4jClient, get_neo4j_client
from .gemini_client import GeminiClient, get_gemini_client
from .graph_retriever import NodeLayer

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Semantic search using Gemini embeddings with Neo4j vector index."""
    
    # Properties to embed for each node type
    EMBEDDING_PROPERTIES = {
        "Character": ["description", "backstory", "role"],
        "Event": ["synopsis", "title"],
        "Location": ["description", "role"],
        "Object": ["description", "function"],
        "Chapter": ["title"],
        "Concept": ["definition", "name"],
        "Symbol": ["description", "name"],
        "Allusion": ["description", "target_text", "source_work", "id"],
        "Commentary": ["summary", "analysis"],
        "Glossary": ["term", "definition", "description"]
    }
    
    VECTOR_INDEX_NAME = "moby_dick_embeddings"
    EMBEDDING_DIMENSION = 768  # Gemini text-embedding-004 dimension
    
    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        gemini_client: GeminiClient | None = None
    ):
        """Initialize the vector retriever.
        
        Args:
            neo4j_client: Neo4j client instance.
            gemini_client: Gemini client for embeddings.
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        self.gemini = gemini_client or get_gemini_client()
        self._index_verified = False
    
    def ensure_vector_index(self) -> bool:
        """Ensure the vector index exists in Neo4j.
        
        Returns:
            True if index exists or was created successfully.
        """
        if self._index_verified:
            return True
        
        # Check if index exists
        check_query = """
        SHOW INDEXES
        WHERE name = $index_name
        RETURN count(*) as count
        """
        
        result = self.neo4j.execute_query(
            check_query, 
            {"index_name": self.VECTOR_INDEX_NAME}
        )
        
        if result and result[0]["count"] > 0:
            logger.info(f"Vector index '{self.VECTOR_INDEX_NAME}' already exists")
            self._index_verified = True
            return True
        
        # Create vector index - using a general approach
        # Neo4j 5.x+ supports vector indexes
        try:
            create_query = f"""
            CREATE VECTOR INDEX {self.VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n:EmbeddedNode)
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.EMBEDDING_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            self.neo4j.execute_write(create_query)
            logger.info(f"Created vector index '{self.VECTOR_INDEX_NAME}'")
            self._index_verified = True
            return True
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}. Will use fallback similarity search.")
            return False
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.
        
        Args:
            query: Search query text.
            
        Returns:
            Embedding vector.
        """
        return self.gemini.embed(query, task_type="retrieval_query")
    
    def embed_document(self, text: str) -> list[float]:
        """Generate embedding for a document.
        
        Args:
            text: Document text.
            
        Returns:
            Embedding vector.
        """
        return self.gemini.embed(text, task_type="retrieval_document")
    
    def semantic_search(
        self,
        query: str,
        layer: NodeLayer = NodeLayer.ALL,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Perform semantic search using embeddings.
        
        Args:
            query: Natural language search query.
            layer: Layer to search in.
            limit: Maximum results to return.
            similarity_threshold: Minimum similarity score (0-1).
            
        Returns:
            List of semantically similar nodes with scores.
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Try vector index first, fallback to computed similarity
        if self._index_verified:
            return self._search_with_index(
                query_embedding, layer, limit, similarity_threshold
            )
        else:
            return self._search_with_fallback(
                query, query_embedding, layer, limit
            )
    
    def _search_with_index(
        self,
        query_embedding: list[float],
        layer: NodeLayer,
        limit: int,
        threshold: float
    ) -> list[dict[str, Any]]:
        """Search using Neo4j vector index."""
        label_filter = self._get_label_filter(layer)
        
        query = f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $limit,
            $embedding
        ) YIELD node, score
        WHERE score >= $threshold
        {f"AND any(label IN labels(node) WHERE label IN $labels)" if label_filter else ""}
        RETURN node, score, labels(node) as labels
        ORDER BY score DESC
        """
        
        params = {
            "index_name": self.VECTOR_INDEX_NAME,
            "limit": limit,
            "embedding": query_embedding,
            "threshold": threshold
        }
        if label_filter:
            params["labels"] = label_filter
        
        results = self.neo4j.execute_query(query, params)
        
        return [
            {
                **dict(r["node"]),
                "type": r["labels"][0] if r["labels"] else "Unknown",
                "similarity_score": r["score"]
            }
            for r in results
        ]
    
    def _search_with_fallback(
        self,
        query_text: str,
        query_embedding: list[float],
        layer: NodeLayer,
        limit: int
    ) -> list[dict[str, Any]]:
        """Fallback search using text matching and local similarity computation.
        
        When vector index is not available, we:
        1. Do a broad text search to find candidates
        2. Get their text content
        3. Embed and compute similarity locally
        """
        logger.info("Using fallback semantic search (no vector index)")
        
        labels = self._get_label_filter(layer)
        label_str = "|".join(labels) if labels else ""
        
        # Get candidate nodes with their text content
        query = f"""
        MATCH (n{':' + label_str if label_str else ''})
        WHERE n.description IS NOT NULL 
           OR n.definition IS NOT NULL 
           OR n.synopsis IS NOT NULL
           OR n.backstory IS NOT NULL
        RETURN n, labels(n) as labels,
               coalesce(n.description, '') + ' ' + 
               coalesce(n.definition, '') + ' ' + 
               coalesce(n.synopsis, '') + ' ' +
               coalesce(n.backstory, '') as text_content
        LIMIT 50
        """
        
        candidates = self.neo4j.execute_query(query, {})
        
        if not candidates:
            return []
        
        # Compute similarities
        results_with_scores = []
        for candidate in candidates:
            text = candidate["text_content"].strip()
            if not text:
                continue
            
            # Compute cosine similarity
            doc_embedding = self.embed_document(text)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            results_with_scores.append({
                **dict(candidate["n"]),
                "type": candidate["labels"][0] if candidate["labels"] else "Unknown",
                "similarity_score": similarity
            })
        
        # Sort by similarity and return top results
        results_with_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results_with_scores[:limit]
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_label_filter(self, layer: NodeLayer) -> list[str]:
        """Get labels to filter by layer."""
        if layer == NodeLayer.FACTS:
            return ["Character", "Event", "Location", "Object", "Chapter", "Glossary"]
        elif layer == NodeLayer.ANALYSIS:
            return ["Concept", "Symbol", "Allusion", "Commentary"]
        else:
            return []
    
    def embed_and_store_nodes(self, batch_size: int = 10) -> int:
        """Embed all nodes and store embeddings in Neo4j.
        
        This is a one-time operation to prepare the database for vector search.
        
        Args:
            batch_size: Number of nodes to process at once.
            
        Returns:
            Number of nodes embedded.
        """
        total_embedded = 0
        
        for label, properties in self.EMBEDDING_PROPERTIES.items():
            logger.info(f"Embedding {label} nodes...")
            
            # Get nodes that need embedding
            query = f"""
            MATCH (n:{label})
            WHERE n.embedding IS NULL
            RETURN n, id(n) as node_id
            """
            
            nodes = self.neo4j.execute_query(query, {})
            
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                for node_data in batch:
                    node = node_data["n"]
                    node_id = node_data["node_id"]
                    
                    # Build text from relevant properties
                    text_parts = []
                    for prop in properties:
                        if prop in node and node[prop]:
                            text_parts.append(str(node[prop]))
                    
                    if not text_parts:
                        continue
                    
                    text = " ".join(text_parts)
                    
                    # Generate embedding
                    embedding = self.embed_document(text)
                    
                    # Store embedding
                    update_query = """
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.embedding = $embedding,
                        n:EmbeddedNode
                    """
                    
                    self.neo4j.execute_write(
                        update_query,
                        {"node_id": node_id, "embedding": embedding}
                    )
                    total_embedded += 1
                
                logger.info(f"Embedded {min(i + batch_size, len(nodes))}/{len(nodes)} {label} nodes")
        
        logger.info(f"Total nodes embedded: {total_embedded}")
        return total_embedded
    
    def get_similar_nodes(
        self,
        node_name: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find nodes similar to a given node.
        
        Args:
            node_name: Name of the reference node.
            limit: Maximum similar nodes to return.
            
        Returns:
            List of similar nodes with scores.
        """
        # Get the reference node's embedding
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name)
           OR toLower(n.id) CONTAINS toLower($name)
        RETURN n.embedding as embedding, n.name as name
        LIMIT 1
        """
        
        result = self.neo4j.execute_query(query, {"name": node_name})
        
        if not result or not result[0].get("embedding"):
            logger.warning(f"Node '{node_name}' not found or has no embedding")
            return []
        
        reference_embedding = result[0]["embedding"]
        reference_name = result[0]["name"]
        
        # Find similar nodes
        if self._index_verified:
            similar_query = """
            CALL db.index.vector.queryNodes($index_name, $limit + 1, $embedding)
            YIELD node, score
            WHERE node.name <> $exclude_name
            RETURN node, score, labels(node) as labels
            ORDER BY score DESC
            LIMIT $limit
            """
            
            results = self.neo4j.execute_query(similar_query, {
                "index_name": self.VECTOR_INDEX_NAME,
                "limit": limit,
                "embedding": reference_embedding,
                "exclude_name": reference_name
            })
            
            return [
                {
                    **dict(r["node"]),
                    "type": r["labels"][0] if r["labels"] else "Unknown",
                    "similarity_score": r["score"]
                }
                for r in results
            ]
        else:
            # Fallback: Build text from node and search
            return self.semantic_search(reference_name, limit=limit)


# Singleton instance
_retriever: VectorRetriever | None = None


def get_vector_retriever() -> VectorRetriever:
    """Get or create the singleton vector retriever."""
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
    return _retriever
