"""Layer-aware prompt templates for Moby-Dick GraphRAG."""

from dataclasses import dataclass, field
from typing import Any
import re
import logging

logger = logging.getLogger(__name__)


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


@dataclass
class QuoteBudget:
    """Configuration for limiting quotations in responses.
    
    Prevents quote-heavy outputs by tracking and limiting direct quotations
    from source material.
    """
    max_quotes: int = 3
    max_quote_length: int = 150
    enabled: bool = True
    quotes_found: list[str] = field(default_factory=list)
    
    # Regex patterns for detecting quotations
    # Matches "..." and "..." (straight and curly quotes)
    QUOTE_PATTERNS: list[str] = field(default_factory=lambda: [
        r'"([^"]{10,})"',  # Straight double quotes, min 10 chars
        r'"([^"]{10,})"',  # Curly double quotes
        r"'([^']{10,})'",  # Single quotes (for nested quotes)
    ])
    
    def detect_quotes(self, text: str) -> list[str]:
        """Detect quotations in text using regex patterns.
        
        Args:
            text: Text to scan for quotations.
            
        Returns:
            List of unique detected quotation strings.
        """
        quotes = []
        seen = set()
        for pattern in self.QUOTE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                # Deduplicate by normalized content
                normalized = match.strip().lower()
                if normalized not in seen:
                    quotes.append(match)
                    seen.add(normalized)
        return quotes
    
    def process_context(self, context: RetrievedContext) -> tuple[RetrievedContext, int]:
        """Process context to track and optionally truncate quotes.
        
        Args:
            context: Retrieved context with facts and analysis.
            
        Returns:
            Tuple of (processed context, quote count).
        """
        if not self.enabled:
            return context, 0
        
        all_quotes = []
        
        # Scan facts for quotes
        for fact in context.facts:
            for fld in ["description", "synopsis", "target_text", "backstory"]:
                if fld in fact and fact[fld]:
                    quotes = self.detect_quotes(str(fact[fld]))
                    all_quotes.extend(quotes)
        
        # Scan analysis for quotes
        for item in context.analysis:
            for fld in ["description", "analysis", "target_text", "summary"]:
                if fld in item and item[fld]:
                    quotes = self.detect_quotes(str(item[fld]))
                    all_quotes.extend(quotes)
        
        # Store found quotes and truncate long ones
        self.quotes_found = []
        for quote in all_quotes:
            if len(quote) > self.max_quote_length:
                truncated = quote[:self.max_quote_length] + "..."
                self.quotes_found.append(truncated)
            else:
                self.quotes_found.append(quote)
        
        return context, len(all_quotes)
    
    def get_budget_instruction(self, quote_count: int) -> str:
        """Generate instruction for the LLM about quote usage.
        
        Args:
            quote_count: Number of quotes found in context.
            
        Returns:
            Instruction string for the prompt.
        """
        if not self.enabled:
            return ""
        
        if quote_count == 0:
            return "Note: No direct quotations were found in the context. Feel free to paraphrase the source material."
        
        if quote_count <= self.max_quotes:
            return f"Quote Budget: You may use up to {self.max_quotes} direct quotations. Use them strategically for emphasis."
        
        return f"""Quote Budget: The context contains {quote_count} quotations, but limit your response to at most {self.max_quotes} direct quotes.
- Paraphrase when possible instead of quoting directly
- Reserve quotations for particularly striking or essential passages
- When quoting, keep quotes under {self.max_quote_length} characters"""


# Synthesis-focused instructions for generating connections
SYNTHESIS_INSTRUCTIONS = """
## SYNTHESIS GUIDELINES
Your response should prioritize generating CONNECTIONS and INSIGHTS over presenting facts:

1. **Explain relationships**: When presenting information, explain WHY and HOW elements connect, not just WHAT they are
2. **Cross-layer synthesis**: Generate at least one insight that links Layer 1 facts to Layer 2 analysis
3. **Causal reasoning**: Use phrases like "this leads to", "because of this", "which reveals" to show logical flow
4. **Avoid enumeration**: Do not use bullet-point lists unless specifically asked; prefer flowing prose
5. **Novel connections**: Identify relationships between elements that may not be explicitly stated in the context
6. **Interpretive synthesis**: Don't just report what the text says—explain what it means and why it matters"""


SYSTEM_INSTRUCTION = """You are an expert encyclopedia for Herman Melville's novel "Moby-Dick; or, The Whale" (1851).

Your role is to provide accurate, insightful, and well-sourced answers that SYNTHESIZE knowledge rather than merely report facts.

Topics you can address:
- Characters, their relationships, and development
- Plot events and narrative structure  
- Locations and settings (ships, ports, the sea)
- Objects and their significance
- Themes, symbols, and literary analysis
- Allusions to mythology, religion, and other works
- Historical and cultural context

## Response Structure (Inverted Pyramid Style)

**For simple factual or Yes/No questions:**
Start with a **Direct Answer** in bold—one clear sentence that directly answers the question. Then provide the Detailed Analysis below with supporting evidence and context.

Example format:
> **Direct Answer:** Yes, Queequeg is a harpooner aboard the Pequod.
>
> *Detailed Analysis:* Queequeg serves as one of three harpooners...

**For complex analytical questions:**
Lead with your key insight or thesis, then develop it with synthesized evidence from both layers.

Core Guidelines:
1. SYNTHESIZE, don't enumerate: Weave facts and analysis into cohesive insights
2. Explain CONNECTIONS: Show how elements relate to each other and why those relationships matter
3. Distinguish layers: Indicate when you're drawing from FACTS (Layer 1) vs ANALYSIS (Layer 2)
4. Ground in text: Cite chapter references when available to anchor interpretations
5. Generate insight: Go beyond what's explicitly stated to reveal deeper patterns and meanings
6. Use flowing prose: Avoid bullet-point lists unless specifically requested
7. Be judicious with quotes: Paraphrase when possible; quote only for rhetorical effect

You have access to a two-layer knowledge graph:
- Layer 1 (Facts): Characters, Events, Locations, Objects, Chapters
- Layer 2 (Analysis): Concepts, Symbols, Allusions, Commentary

Your goal is to help readers understand not just WHAT happens in Moby-Dick, but WHY it matters."""


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
            if "weight" in rel and rel["weight"] is not None:
                lines.append(f"**Relevance:** {rel['weight']:.2f}")
            elif "weight" in rel:
                lines.append("**Relevance:** 0.00")
        
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
    
    # Add response instructions with synthesis focus
    parts.append("""
---

## RESPONSE INSTRUCTIONS
1. SYNTHESIZE the retrieved context into a cohesive response—don't just list facts
2. CONNECT elements: Explain how facts from Layer 1 relate to interpretations in Layer 2
3. Generate at least one INSIGHT that links narrative elements to thematic meaning
4. Use FLOWING PROSE rather than bullet points or numbered lists
5. GROUND your response with chapter references where available
6. Be JUDICIOUS with quotations—paraphrase when the meaning can be conveyed without direct quotes
7. If context is insufficient, acknowledge this while offering what synthesis is possible""")
    
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


def build_synthesis_prompt(
    context: RetrievedContext,
    quote_budget: QuoteBudget | None = None,
    include_synthesis_guidelines: bool = True
) -> str:
    """Build a prompt optimized for synthesis and connection-making.
    
    This is the preferred prompt builder when using the selection layer,
    as it includes quote budget awareness and synthesis guidelines.
    
    Args:
        context: Retrieved context from the knowledge graph.
        quote_budget: Optional quote budget configuration.
        include_synthesis_guidelines: Whether to include detailed synthesis instructions.
        
    Returns:
        Complete prompt string optimized for synthesized responses.
    """
    parts = []
    
    # Process context through quote budget if provided
    quote_count = 0
    if quote_budget:
        context, quote_count = quote_budget.process_context(context)
    
    # Add context sections
    if context.has_facts():
        parts.append(format_facts_context(context.facts))
    
    if context.has_analysis():
        parts.append(format_analysis_context(context.analysis))
    
    if context.is_empty():
        parts.append("No relevant context was retrieved from the knowledge graph. Synthesize an answer based on your general knowledge of Moby-Dick, acknowledging the limitation.")
    
    # Add synthesis guidelines if enabled
    if include_synthesis_guidelines:
        parts.append(SYNTHESIS_INSTRUCTIONS)
    
    # Add quote budget instruction if applicable
    if quote_budget and quote_budget.enabled:
        budget_instruction = quote_budget.get_budget_instruction(quote_count)
        if budget_instruction:
            parts.append(f"\n{budget_instruction}")
    
    # Add the user query
    parts.append(f"\n---\n\n## USER QUESTION\n{context.query}")
    
    # Add enhanced response instructions
    parts.append("""
---

## RESPONSE INSTRUCTIONS
1. SYNTHESIZE: Weave the retrieved facts and analysis into a unified, insightful response
2. CONNECT LAYERS: Explicitly link narrative facts (Layer 1) to interpretive analysis (Layer 2)
3. GENERATE INSIGHT: Offer at least one original observation about how elements interrelate
4. FLOW: Write in connected prose paragraphs, not disconnected bullet points
5. GROUND: Anchor interpretations with specific chapter or scene references
6. QUOTE WISELY: Use direct quotations sparingly and only for rhetorical impact
7. ACKNOWLEDGE GAPS: If context is limited, say so while synthesizing what's available""")
    
    return "\n\n".join(parts)


def build_comparison_synthesis_prompt(
    entity1: str,
    entity2: str,
    context: RetrievedContext,
    quote_budget: QuoteBudget | None = None
) -> str:
    """Build a synthesis-focused prompt for comparing two entities.
    
    Args:
        entity1: First entity being compared.
        entity2: Second entity being compared.
        context: Retrieved context from the knowledge graph.
        quote_budget: Optional quote budget configuration.
        
    Returns:
        Complete prompt string for comparative synthesis.
    """
    base_prompt = build_synthesis_prompt(context, quote_budget, include_synthesis_guidelines=True)
    
    comparison_instructions = f"""
## COMPARISON FOCUS: {entity1} vs {entity2}

Generate a synthesized comparison that:
1. **Identifies the nature** of each element (character, theme, symbol, etc.)
2. **Reveals connections**: How do these elements relate to each other in the text?
3. **Explores contrasts**: What meaningful differences illuminate Melville's intentions?
4. **Traces development**: How does their relationship or contrast evolve?
5. **Synthesizes meaning**: What does comparing these elements reveal about the novel's themes?

Avoid simple side-by-side listing—instead, weave the comparison into a flowing analysis."""
    
    return base_prompt + "\n\n" + comparison_instructions
