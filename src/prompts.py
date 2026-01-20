"""Layer-aware prompt templates for Moby-Dick GraphRAG."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedContext:
    """Container for retrieved graph context."""
    facts: list[dict[str, Any]]  # Layer 1: Characters, Events, Locations, Objects
    analysis: list[dict[str, Any]]  # Layer 2: Concepts, Symbols, Allusions, Commentary
    query: str
    
    def has_facts(self) -> bool:
        return len(self.facts) > 0
    
    def has_analysis(self) -> bool:
        return len(self.analysis) > 0
    
    def is_empty(self) -> bool:
        return not self.has_facts() and not self.has_analysis()


SYSTEM_INSTRUCTION = """You are an expert encyclopedia for Herman Melville's novel "Moby-Dick; or, The Whale" (1851).

Your role is to provide accurate, insightful, and well-sourced answers about:
- Characters, their relationships, and development
- Plot events and narrative structure
- Locations and settings (ships, ports, the sea)
- Objects and their significance
- Themes, symbols, and literary analysis
- Allusions to mythology, religion, and other works
- Historical and cultural context

Guidelines:
1. Base your answers primarily on the retrieved context from the knowledge graph
2. Clearly distinguish between FACTS (what happens in the text) and ANALYSIS (interpretation)
3. Cite chapter references when available
4. Acknowledge uncertainty if the retrieved context is insufficient
5. Use precise literary terminology when discussing themes and techniques
6. Be engaging but scholarly in tone

You have access to a two-layer knowledge graph:
- Layer 1 (Facts): Characters, Events, Locations, Objects, Chapters
- Layer 2 (Analysis): Concepts, Symbols, Allusions, Commentary"""


def format_facts_context(facts: list[dict[str, Any]]) -> str:
    """Format Layer 1 (facts) context for prompt injection.
    
    Args:
        facts: List of fact nodes/relationships from the graph.
        
    Returns:
        Formatted string for prompt injection.
    """
    if not facts:
        return "No factual context retrieved."
    
    lines = ["## FACTS (Layer 1 - Narrative Elements)"]
    
    for i, fact in enumerate(facts, 1):
        node_type = fact.get("type", fact.get("label", "Unknown"))
        lines.append(f"\n### {node_type} #{i}")
        
        # Handle different node types
        if "name" in fact:
            lines.append(f"**Name:** {fact['name']}")
        if "title" in fact:
            lines.append(f"**Title:** {fact['title']}")
        
        # Glossary-specific fields
        if "term" in fact:
            lines.append(f"**Term:** {fact['term']}")
        if "definition" in fact:
            lines.append(f"**Definition:** {fact['definition']}")
        if "category" in fact:
            lines.append(f"**Category:** {fact['category']}")
        
        if "description" in fact:
            lines.append(f"**Description:** {fact['description']}")
        if "role" in fact:
            lines.append(f"**Role:** {fact['role']}")
        if "backstory" in fact:
            lines.append(f"**Backstory:** {fact['backstory']}")
        if "synopsis" in fact:
            lines.append(f"**Synopsis:** {fact['synopsis']}")
        if "chapter_ref" in fact:
            lines.append(f"**Chapter:** {fact['chapter_ref']}")
        
        # Relationship context
        if "relationship" in fact:
            rel = fact["relationship"]
            lines.append(f"**Relationship:** {rel.get('type', 'RELATED_TO')}")
            if "context" in rel:
                lines.append(f"**Context:** {rel['context']}")
            if "weight" in rel:
                lines.append(f"**Relevance:** {rel['weight']:.2f}")
        
        # Connected nodes
        if "connected_to" in fact:
            lines.append(f"**Connected to:** {fact['connected_to']}")
    
    return "\n".join(lines)


def format_analysis_context(analysis: list[dict[str, Any]]) -> str:
    """Format Layer 2 (analysis) context for prompt injection.
    
    Args:
        analysis: List of analysis nodes/relationships from the graph.
        
    Returns:
        Formatted string for prompt injection.
    """
    if not analysis:
        return "No analytical context retrieved."
    
    lines = ["## ANALYSIS (Layer 2 - Interpretive Elements)"]
    
    for i, item in enumerate(analysis, 1):
        node_type = item.get("type", item.get("label", "Unknown"))
        lines.append(f"\n### {node_type} #{i}")
        
        if "name" in item:
            lines.append(f"**Name:** {item['name']}")
        if "definition" in item:
            lines.append(f"**Definition:** {item['definition']}")
        if "description" in item:
            lines.append(f"**Description:** {item['description']}")
        if "source_work" in item:
            lines.append(f"**Source:** {item['source_work']}")
        if "target_text" in item:
            lines.append(f"**In Text:** {item['target_text']}")
        if "motif_group" in item:
            lines.append(f"**Motif Group:** {item['motif_group']}")
        if "summary" in item:
            lines.append(f"**Summary:** {item['summary']}")
        if "analysis" in item:
            lines.append(f"**Analysis:** {item['analysis']}")
        if "chapter_refs" in item:
            refs = item["chapter_refs"]
            if isinstance(refs, list):
                refs = ", ".join(str(r) for r in refs)
            lines.append(f"**Chapters:** {refs}")
        
        # Relationship context
        if "relationship" in item:
            rel = item["relationship"]
            lines.append(f"**Relationship:** {rel.get('type', 'RELATED_TO')}")
            if "context" in rel:
                lines.append(f"**Context:** {rel['context']}")
    
    return "\n".join(lines)


def build_query_prompt(context: RetrievedContext) -> str:
    """Build the complete prompt with retrieved context.
    
    Args:
        context: Retrieved context from the knowledge graph.
        
    Returns:
        Complete prompt string for Gemini.
    """
    parts = []
    
    # Add context sections
    if context.has_facts():
        parts.append(format_facts_context(context.facts))
    
    if context.has_analysis():
        parts.append(format_analysis_context(context.analysis))
    
    if context.is_empty():
        parts.append("No relevant context was retrieved from the knowledge graph. Answer based on your general knowledge of Moby-Dick, but note the limitation.")
    
    # Add the user query
    parts.append(f"\n---\n\n## USER QUESTION\n{context.query}")
    
    # Add response instructions
    parts.append("""
---

## RESPONSE INSTRUCTIONS
1. Answer the question using the retrieved context above
2. If citing facts, indicate they come from the narrative (Layer 1)
3. If discussing interpretation, indicate it's analysis (Layer 2)
4. Include chapter references where available
5. If the context is insufficient, acknowledge this clearly""")
    
    return "\n\n".join(parts)


def build_character_prompt(character_name: str, context: RetrievedContext) -> str:
    """Build a prompt specifically for character queries.
    
    Args:
        character_name: Name of the character being queried.
        context: Retrieved context from the knowledge graph.
        
    Returns:
        Complete prompt string for Gemini.
    """
    base_prompt = build_query_prompt(context)
    
    character_instructions = f"""
## CHARACTER FOCUS: {character_name}

When answering about this character, address:
1. **Identity**: Who are they? What is their role in the story?
2. **Relationships**: Key connections to other characters
3. **Development**: How do they change throughout the narrative?
4. **Significance**: What themes or ideas do they embody?
5. **Key Scenes**: Notable moments involving this character"""
    
    return base_prompt + "\n\n" + character_instructions


def build_theme_prompt(theme: str, context: RetrievedContext) -> str:
    """Build a prompt specifically for thematic queries.
    
    Args:
        theme: Theme or concept being queried.
        context: Retrieved context from the knowledge graph.
        
    Returns:
        Complete prompt string for Gemini.
    """
    base_prompt = build_query_prompt(context)
    
    theme_instructions = f"""
## THEMATIC FOCUS: {theme}

When discussing this theme, address:
1. **Definition**: How is this theme understood in the context of the novel?
2. **Manifestations**: Where and how does it appear in the text?
3. **Characters**: Which characters embody or explore this theme?
4. **Symbols**: What symbols or motifs relate to this theme?
5. **Interpretation**: What is the significance of this theme in Melville's work?"""
    
    return base_prompt + "\n\n" + theme_instructions


def build_chapter_prompt(chapter_num: int, context: RetrievedContext) -> str:
    """Build a prompt specifically for chapter queries.
    
    Args:
        chapter_num: Chapter number being queried.
        context: Retrieved context from the knowledge graph.
        
    Returns:
        Complete prompt string for Gemini.
    """
    base_prompt = build_query_prompt(context)
    
    chapter_instructions = f"""
## CHAPTER FOCUS: Chapter {chapter_num}

When discussing this chapter, address:
1. **Summary**: What happens in this chapter?
2. **Characters**: Who appears and what do they do?
3. **Setting**: Where does the action take place?
4. **Themes**: What themes or ideas are explored?
5. **Significance**: How does this chapter fit into the larger narrative?
6. **Notable Elements**: Any important symbols, allusions, or literary techniques?"""
    
    return base_prompt + "\n\n" + chapter_instructions
