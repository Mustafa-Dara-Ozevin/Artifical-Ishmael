# Moby-Dick GraphRAG Encyclopedia 🐋

An Evren SSB API-powered knowledge base for Herman Melville's *Moby-Dick*, built on a two-layer Neo4j knowledge graph.

## Features

- **Two-Layer Knowledge Graph**: Facts (characters, events, locations) + Analysis (concepts, symbols, allusions)
- **Hybrid Retrieval**: Combines graph traversal with semantic vector search
- **Natural Language Queries**: Ask questions in plain English
- **Evren SSB API-Powered Responses**: Grounded answers using Cumhurbaşkanlığı Savunma Sanayii Başkanlığı's (SSB) EVREN sovereign AI platform
- **Rich CLI Interface & Streamlit UI**: Interactive encyclopedia with streaming responses and knowledge graph visualizer

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` (for local development) with your credentials:

```env
# Neo4j Aura
NEO4J_URI=neo4j+s://526fc1bc.databases.neo4j.io
NEO4J_USER=526fc1bc
NEO4J_PASSWORD=your_password

# Choose your provider: 'evren', 'groq', or 'gemini'
LLM_PROVIDER=evren

# SSB EVREN API (https://evren.ssyz.org.tr)
EVREN_API_KEY=your_evren_api_key_here
EVREN_BASE_URL=https://evren-llmapi.ssyz.org.tr/v1
EVREN_MODEL=deepseek-v4-flash
```

Get your EVREN API key from [evren.ssyz.org.tr](https://evren.ssyz.org.tr) via e-Devlet authentication under **"LLM Çıkarımı"**.

### Streamlit Cloud Deployment

When deploying to Streamlit Cloud, add your environment variables under **App settings > Secrets**:

```toml
NEO4J_URI = "neo4j+s://526fc1bc.databases.neo4j.io"
NEO4J_USER = "526fc1bc"
NEO4J_PASSWORD = "your_password"

LLM_PROVIDER = "evren"
EVREN_API_KEY = "your_evren_api_key"
EVREN_BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"
EVREN_MODEL = "deepseek-v4-flash"

# Optional LLM fallbacks
GROQ_API_KEY = "your_groq_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

> ⚠️ **AuraDB Free Note:** If you see `neo4j.exceptions.ServiceUnavailable: Failed to DNS resolve address`, your Neo4j AuraDB Free instance may be **Paused** due to inactivity. Go to [console.neo4j.io](https://console.neo4j.io) and click **Resume**. When an instance is paused, its DNS address is temporarily deactivated.


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
├── config.py           # Configuration management (Evren, Groq, Gemini, Neo4j)
├── neo4j_client.py     # Neo4j Aura connection
├── evren_client.py     # Evren SSB API wrapper (OpenAI-compatible)
├── gemini_client.py    # Gemini API wrapper (fallback/embeddings)
├── groq_client.py      # Groq API wrapper
├── graph_retriever.py  # Cypher-based retrieval
├── vector_retriever.py # Semantic search
├── hybrid_retriever.py # Combined retrieval + ranking
├── selection_layer.py  # Rhetorical filtering layer
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
┌──────────────────┐      ┌──────────────────┐
│  Evren SSB API   │◄────►│  Query Engine    │
│  (Generation)    │      │  (Orchestrator)  │
└──────────────────┘      └────────┬─────────┘
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
