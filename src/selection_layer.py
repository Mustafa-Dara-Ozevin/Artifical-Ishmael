"""Selection layer for rhetorical filtering between retrieval and generation.

This module implements a two-phase selection process:
1. Grounding check: Ensure at least one node with chapter_ref is retained
2. Rhetorical scoring: Prioritize nodes with rich relationships for synthesis
"""

from dataclasses import dataclass, field
from typing import Any
import logging
import re

from .hybrid_retriever import HybridResults, RetrievalResult
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for the selection layer."""
    min_grounded: int = 1  # Minimum nodes with chapter_ref to guarantee
    min_facts: int = 2  # Minimum facts to guarantee
    min_analysis: int = 1  # Minimum analysis nodes to guarantee
    relationship_weight: float = 0.3  # Weight per relationship
    cross_layer_bonus: float = 0.2  # Bonus for cross-layer connections
    grounded_bonus: float = 0.15  # Bonus for having chapter_ref
    max_relationship_score: float = 0.6  # Cap on relationship contribution


@dataclass
class ScoredResult:
    """A retrieval result with rhetorical scoring."""
    result: RetrievalResult
    rhetorical_score: float
    is_grounded: bool
    relationship_count: int
    has_cross_layer_link: bool
    
    @property
    def combined_score(self) -> float:
        """Combine retrieval score with rhetorical score."""
        # Handle cases where retrieval score might be None (e.g. raw graph hits)
        score = self.result.score if self.result.score is not None else 0.0
        return (score * 0.5) + (self.rhetorical_score * 0.5)


class SelectionLayer:
    """Filters retrieved content for rhetorical effectiveness.
    
    Implements a two-phase selection:
    1. Grounding phase: Tag nodes with chapter references as grounded sources
    2. Scoring phase: Rank by relationship richness and cross-layer connections
    
    The layer guarantees that at least `min_grounded` grounded sources are
    retained, even if they score lower on rhetorical metrics.
    """
    
    def __init__(self, config: SelectionConfig | None = None):
        """Initialize the selection layer.
        
        Args:
            config: Selection configuration. Uses defaults if not provided.
        """
        self.config = config or SelectionConfig()
    
    def filter(
        self,
        results: HybridResults,
        query: str,
        max_facts: int = 10,
        max_analysis: int = 5
    ) -> HybridResults:
        """Filter and re-rank results for rhetorical effectiveness.
        
        Args:
            results: Raw hybrid retrieval results.
            query: Original user query (for context-aware scoring).
            max_facts: Maximum facts to return after filtering.
            max_analysis: Maximum analysis nodes to return.
            
        Returns:
            Filtered HybridResults with rhetorically effective content.
        """
        logger.info(f"Selection layer filtering {len(results.facts)} facts, {len(results.analysis)} analysis")
        
        # Phase 1: Score all results
        scored_facts = [self._score_result(r, results.analysis) for r in results.facts]
        scored_analysis = [self._score_result(r, results.facts) for r in results.analysis]
        
        # Phase 2: Select with grounding guarantee
        selected_facts = self._select_with_grounding(
            scored_facts,
            max_results=max_facts,
            min_grounded=self.config.min_grounded,
            min_total=self.config.min_facts
        )
        
        selected_analysis = self._select_with_grounding(
            scored_analysis,
            max_results=max_analysis,
            min_grounded=0,  # Analysis nodes don't require grounding
            min_total=self.config.min_analysis
        )
        
        logger.info(f"Selection layer output: {len(selected_facts)} facts, {len(selected_analysis)} analysis")
        
        return HybridResults(
            facts=selected_facts,
            analysis=selected_analysis,
            query=results.query
        )
    
    def _score_result(
        self,
        result: RetrievalResult,
        other_layer: list[RetrievalResult]
    ) -> ScoredResult:
        """Compute rhetorical score for a single result.
        
        Scoring factors:
        - Relationship count (more relationships = better for synthesis)
        - Cross-layer connections (links to other layer = richer context)
        - Grounding (chapter_ref = authoritative evidence)
        
        Args:
            result: The retrieval result to score.
            other_layer: Results from the other layer for cross-link detection.
            
        Returns:
            Scored result with rhetorical metrics.
        """
        node = result.node
        
        # Count relationships
        relationships = node.get("relationships", [])
        if isinstance(relationships, str):
            # Handle comma-separated string format
            relationships = [r.strip() for r in relationships.split(",") if r.strip()]
        relationship_count = len(relationships) if isinstance(relationships, list) else 0
        
        # Check for grounding (chapter reference)
        is_grounded = self._has_chapter_ref(node)
        
        # Check for cross-layer connections
        has_cross_layer = self._has_cross_layer_link(node, other_layer)
        
        # Compute rhetorical score
        rhetorical_score = 0.0
        
        # Relationship contribution (capped)
        rel_contribution = min(
            relationship_count * self.config.relationship_weight,
            self.config.max_relationship_score
        )
        rhetorical_score += rel_contribution
        
        # Cross-layer bonus
        if has_cross_layer:
            rhetorical_score += self.config.cross_layer_bonus
        
        # Grounding bonus
        if is_grounded:
            rhetorical_score += self.config.grounded_bonus
        
        # Normalize to 0-1 range
        max_possible = (
            self.config.max_relationship_score +
            self.config.cross_layer_bonus +
            self.config.grounded_bonus
        )
        rhetorical_score = min(rhetorical_score / max_possible, 1.0)
        
        return ScoredResult(
            result=result,
            rhetorical_score=rhetorical_score,
            is_grounded=is_grounded,
            relationship_count=relationship_count,
            has_cross_layer_link=has_cross_layer
        )
    
    def _has_chapter_ref(self, node: dict[str, Any]) -> bool:
        """Check if a node has a chapter reference (grounding evidence).
        
        Args:
            node: Node data dictionary.
            
        Returns:
            True if node has chapter reference.
        """
        # Check various chapter reference fields
        if node.get("chapter_ref"):
            return True
        if node.get("chapter_refs"):
            return True
        if node.get("chapter"):
            return True
        
        # Check for chapter in description (e.g., "Chapter 42")
        description = node.get("description", "")
        if re.search(r"chapter\s+\d+", description, re.IGNORECASE):
            return True
        
        return False
    
    def _has_cross_layer_link(
        self,
        node: dict[str, Any],
        other_layer: list[RetrievalResult]
    ) -> bool:
        """Check if node connects to nodes in the other layer.
        
        Args:
            node: Node data dictionary.
            other_layer: Results from the other layer.
            
        Returns:
            True if cross-layer connection exists.
        """
        if not other_layer:
            return False
        
        node_name = node.get("name", "").lower()
        node_id = node.get("id", "")
        
        # Get names/ids from other layer
        other_names = set()
        other_ids = set()
        for r in other_layer:
            other_names.add(r.node.get("name", "").lower())
            other_ids.add(r.node.get("id", ""))
        
        # Check relationships for connections
        relationships = node.get("relationships", [])
        if isinstance(relationships, str):
            relationships = [r.strip() for r in relationships.split(",") if r.strip()]
        
        for rel in relationships:
            if isinstance(rel, dict):
                target = rel.get("target", "").lower()
                if target in other_names:
                    return True
            elif isinstance(rel, str):
                if rel.lower() in other_names:
                    return True
        
        # Check if node name appears in other layer descriptions
        for r in other_layer:
            other_desc = r.node.get("description", "").lower()
            if node_name and node_name in other_desc:
                return True
        
        return False
    
    def _select_with_grounding(
        self,
        scored_results: list[ScoredResult],
        max_results: int,
        min_grounded: int,
        min_total: int
    ) -> list[RetrievalResult]:
        """Select top results while guaranteeing grounded sources.
        
        Strategy:
        1. Separate grounded and ungrounded results
        2. Take top `min_grounded` grounded results first
        3. Fill remaining slots with highest-scoring results
        4. Ensure at least `min_total` results if available
        
        Args:
            scored_results: Results with rhetorical scores.
            max_results: Maximum results to return.
            min_grounded: Minimum grounded results to guarantee.
            min_total: Minimum total results to guarantee.
            
        Returns:
            Selected retrieval results.
        """
        if not scored_results:
            return []
        
        # Separate grounded and ungrounded
        grounded = [s for s in scored_results if s.is_grounded]
        ungrounded = [s for s in scored_results if not s.is_grounded]
        
        # Sort each by combined score
        grounded.sort(key=lambda x: x.combined_score, reverse=True)
        ungrounded.sort(key=lambda x: x.combined_score, reverse=True)
        
        selected: list[RetrievalResult] = []
        
        # Phase 1: Guarantee grounded sources
        grounded_to_take = min(min_grounded, len(grounded))
        for scored in grounded[:grounded_to_take]:
            selected.append(scored.result)
        
        # Phase 2: Fill with top-scoring results (both grounded and ungrounded)
        remaining_grounded = grounded[grounded_to_take:]
        all_remaining = remaining_grounded + ungrounded
        all_remaining.sort(key=lambda x: x.combined_score, reverse=True)
        
        slots_left = max_results - len(selected)
        for scored in all_remaining[:slots_left]:
            selected.append(scored.result)
        
        # Phase 3: Ensure minimum if adaptive relaxation needed
        if len(selected) < min_total and len(scored_results) >= min_total:
            # Add more from original list by score
            selected_ids = {r.node_id for r in selected}
            all_by_score = sorted(scored_results, key=lambda x: x.combined_score, reverse=True)
            
            for scored in all_by_score:
                if len(selected) >= min_total:
                    break
                if scored.result.node_id not in selected_ids:
                    selected.append(scored.result)
                    selected_ids.add(scored.result.node_id)
        
        return selected


def get_selection_layer(config: SelectionConfig | None = None) -> SelectionLayer:
    """Factory function for SelectionLayer.
    
    Args:
        config: Optional configuration override.
        
    Returns:
        Configured SelectionLayer instance.
    """
    return SelectionLayer(config)
