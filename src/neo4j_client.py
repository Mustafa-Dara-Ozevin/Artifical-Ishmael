"""Neo4j Aura client for Moby-Dick GraphRAG."""

from contextlib import contextmanager
from typing import Any, Generator
import logging

from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import ServiceUnavailable, AuthError

from .config import Neo4jConfig, get_config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j database client with connection pooling."""
    
    def __init__(self, config: Neo4jConfig | None = None):
        """Initialize Neo4j client.
        
        Args:
            config: Neo4j configuration. If None, loads from environment.
        """
        self.config = config or get_config().neo4j
        self._driver: Driver | None = None
    
    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
        return self._driver
    
    def close(self) -> None:
        """Close the database connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j Aura.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j Aura")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            return False
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            return False
    
    @contextmanager
    def session(self, database: str = "neo4j") -> Generator[Session, None, None]:
        """Get a database session.
        
        Args:
            database: Database name (default: neo4j).
            
        Yields:
            Neo4j session.
        """
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j"
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string.
            parameters: Query parameters.
            database: Database name.
            
        Returns:
            List of result records as dictionaries.
        """
        with self.session(database) as session:
            result: Result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j"
    ) -> list[dict[str, Any]]:
        """Execute a write transaction.
        
        Args:
            query: Cypher query string.
            parameters: Query parameters.
            database: Database name.
            
        Returns:
            List of result records as dictionaries.
        """
        with self.session(database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, parameters or {}))
            )
            return [record.data() for record in result]
    
    def get_node_labels(self) -> list[str]:
        """Get all node labels in the database.
        
        Returns:
            List of node label names.
        """
        result = self.execute_query("CALL db.labels()")
        return [r["label"] for r in result]
    
    def get_relationship_types(self) -> list[str]:
        """Get all relationship types in the database.
        
        Returns:
            List of relationship type names.
        """
        result = self.execute_query("CALL db.relationshipTypes()")
        return [r["relationshipType"] for r in result]
    
    def get_schema_summary(self) -> dict[str, Any]:
        """Get a summary of the database schema.
        
        Returns:
            Dictionary with labels, relationship types, and counts.
        """
        labels = self.get_node_labels()
        rel_types = self.get_relationship_types()
        
        # Get node counts per label
        node_counts = {}
        for label in labels:
            result = self.execute_query(f"MATCH (n:{label}) RETURN count(n) as count")
            node_counts[label] = result[0]["count"] if result else 0
        
        # Get total relationship count
        rel_count_result = self.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
        total_relationships = rel_count_result[0]["count"] if rel_count_result else 0
        
        return {
            "labels": labels,
            "relationship_types": rel_types,
            "node_counts": node_counts,
            "total_relationships": total_relationships
        }
    
    def __enter__(self) -> "Neo4jClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Singleton instance for convenience
_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create the singleton Neo4j client."""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
