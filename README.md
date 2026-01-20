# Moby-Dick GraphRAG Encyclopedia 🐋

A Gemini-powered knowledge base for Herman Melville's *Moby-Dick*, built on a two-layer Neo4j knowledge graph.

## Features

- **Two-Layer Knowledge Graph**: Facts (characters, events, locations) + Analysis (concepts, symbols, allusions)
- **Hybrid Retrieval**: Combines graph traversal with semantic vector search
- **Natural Language Queries**: Ask questions in plain English
- **Gemini-Powered Responses**: Grounded answers using Google's Gemini API
- **Rich CLI Interface**: Interactive encyclopedia with streaming responses

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` with your credentials:

```env
# Neo4j Aura (already configured)
NEO4J_URI=neo4j+s://da49a084.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Google AI Studio
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the Encyclopedia

```bash
# Interactive mode
python main.py interactive

# Ask a single question
python main.py ask "Who is Captain Ahab?"

# Character lookup
python main.py character Queequeg

# Chapter summary
python main.py chapter 1

# Theme exploration
python main.py theme obsession

# Compare entities
python main.py compare Ishmael Ahab

# View knowledge graph schema
python main.py schema
```

## Project Structure

```
src/
├── config.py           # Configuration management
├── neo4j_client.py     # Neo4j Aura connection
├── gemini_client.py    # Gemini API wrapper
├── graph_retriever.py  # Cypher-based retrieval
├── vector_retriever.py # Semantic search
├── hybrid_retriever.py # Combined retrieval + ranking
├── prompts.py          # Layer-aware prompt templates
├── query_engine.py     # Orchestration layer
└── cli.py              # Typer CLI interface
```

## Knowledge Graph Layers

### Layer 1: Facts (Narrative Elements)
- **Character**: People in the story (Ishmael, Ahab, Queequeg, etc.)
- **Event**: Plot points and actions
- **Location**: Places (The Pequod, Nantucket, the sea)
- **Object**: Significant items (harpoons, the ivory leg)
- **Chapter**: Structural divisions

### Layer 2: Analysis (Interpretive Elements)
- **Concept**: Themes and ideas (obsession, fate, democracy)
- **Symbol**: Symbolic meanings (the whale, whiteness)
- **Allusion**: References to other works (Bible, Shakespeare)
- **Commentary**: Critical analysis

## Example Queries

```bash
# Character questions
python main.py ask "What is Ishmael's role in the story?"
python main.py ask "Describe the relationship between Ishmael and Queequeg"

# Thematic questions
python main.py ask "What does the white whale symbolize?"
python main.py ask "Explain the theme of obsession in Moby-Dick"

# Plot questions
python main.py ask "What happens in Chapter 10?"
python main.py ask "How does the Pequod meet its fate?"

# Comparative questions
python main.py compare "Ahab" "Starbuck"
```

## Architecture

```
┌─────────────────┐      ┌──────────────────┐
│   Gemini API    │◄────►│  Query Engine    │
│  (Generation)   │      │  (Orchestrator)  │
└─────────────────┘      └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌───────────┐ ┌─────────────┐ ┌──────────────┐
            │   Graph   │ │   Vector    │ │   Prompt     │
            │ Retriever │ │ Retriever   │ │  Templates   │
            └─────┬─────┘ └──────┬──────┘ └──────────────┘
                  │              │
                  └──────┬───────┘
                         ▼
                ┌────────────────┐
                │ Hybrid Ranker  │
                └────────┬───────┘
                         ▼
                ┌────────────────┐
                │  Neo4j Aura    │
                │ (Knowledge DB) │
                └────────────────┘
```

## License

MIT
