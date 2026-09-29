import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import logging
import pandas as pd

from src.query_engine import QueryEngine, QueryResult, get_query_engine
from src.config import get_config, validate_config
from src.neo4j_client import get_neo4j_client
from src.evren_client import EvrenClient, get_evren_client
from src.prompts import SYSTEM_INSTRUCTION

# Page configuration
st.set_page_config(
    page_title="🐋 Artifical Ishmael",
    page_icon="🐋",
    layout="wide"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sidebar Configuration ---

with st.sidebar:
    st.header("Settings")
    cfg = get_config()
    cfg.llm_provider = "evren"
    
    st.caption("🤖 **LLM Provider:** `EVREN SSB API` 🇹🇷")
    evren_models = [
        "deepseek-v4-flash",
        "deepseek-v4.1-flash",
        "glm-5.3",
        "gemma-4-31b",
        "qwen3.8-flash-next",
        "mimo-v2.6-pro"
    ]
    default_idx = evren_models.index(cfg.evren.model) if cfg.evren.model in evren_models else 0
    selected_model = st.selectbox(
        "Evren Model",
        options=evren_models,
        index=default_idx,
        help="Select LLM model hosted on Turkey SSB EVREN AI Platform"
    )
    use_stream = st.checkbox("Stream Responses", value=True)
    show_sources = st.checkbox("Show Sources", value=True)
    st.divider()
    st.info("""
    **Legend:**
    * 🔵 **Facts**: Characters, Chapters, Locations
    * 🟠 **Analysis**: Themes, Symbols, Allusions
    """)

# --- Initialization ---

@st.cache_resource
def init_engine(model_name: str = "deepseek-v4-flash"):
    config = get_config()
    config.llm_provider = "evren"
    config.evren.model = model_name
    errors = validate_config(config)
    if errors:
        for error in errors:
            st.error(error)
        st.info(
            "💡 **EVREN API Configuration:**\n"
            "- Ensure `EVREN_API_KEY` is configured in Streamlit Cloud under **App settings > Secrets** or in `.env`.\n"
            "- EVREN API endpoint: `https://evren-llmapi.ssyz.org.tr/v1`\n"
            "- Default model: `deepseek-v4-flash`"
        )
        st.stop()
    evren_client = EvrenClient(config=config.evren)
    return QueryEngine(llm=evren_client)

@st.cache_resource
def init_neo4j():
    return get_neo4j_client()

engine = init_engine(selected_model)
neo4j = init_neo4j()

@st.cache_data(ttl=300)
def get_cached_schema():
    return neo4j.get_schema_summary()

# --- Shared Functions ---

def get_color_for_labels(labels):
    """Determine color based on node labels."""
    fact_types = {"Character", "Event", "Location", "Object", "Chapter", "Glossary"}
    analysis_types = {"Concept", "Symbol", "Allusion", "Commentary"}
    
    # Check if any of the labels match our types
    if any(l in fact_types for l in labels):
        return "#3498db" # Blue for Facts
    if any(l in analysis_types for l in labels):
        return "#e67e22" # Orange for Analysis
    return "#9b59b6" # Purple for others/unknown

def render_graph(nodes, edges, height=600):
    config = Config(
        width=None, # Auto width
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False
    )
    return agraph(nodes=nodes, edges=edges, config=config)

def get_graph_data_from_results(query_result: QueryResult):
    """Extract nodes and relationships for visualization from RAG results."""
    nodes = []
    edges = []
    retrieved_nodes = query_result.context.facts + query_result.context.analysis
    node_ids = []
    
    for r in retrieved_nodes:
        n = r.node
        node_id = n.get("id", n.get("name", "Unknown"))
        node_ids.append(node_id)
        color = "#3498db" if r.layer == 1 else "#e67e22"
        nodes.append(Node(
            id=node_id,
            label=n.get("name", node_id),
            size=25 if r.layer == 1 else 20,
            color=color,
            title=f"Type: {r.node_type}\n{n.get('description', '')[:100]}..."
        ))

    if node_ids:
        query = "MATCH (a)-[r]->(b) WHERE a.id IN $ids AND b.id IN $ids RETURN a.id as start_id, b.id as end_id, type(r) as rel_type"
        try:
            rel_results = neo4j.execute_query(query, {"ids": node_ids})
            for rel in rel_results:
                edges.append(Edge(source=rel["start_id"], target=rel["end_id"], label=rel["rel_type"], color="#7f8c8d"))
        except Exception as e:
            logger.warning(f"Could not load graph relationships: {e}")
            
    return nodes, edges

# --- UI Layout ---

st.title("🐋 Moby-Dick GraphRAG Encyclopedia")

tab1, tab2 = st.tabs(["💬 Chat & Context", "🕸️ Graph Explorer"])

# --- TAB 1: Chat & Context ---
with tab1:
    st.markdown("Ask anything about Melville's masterpiece and see the knowledge graph in action.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Ahab, Moby Dick, or the themes of the book..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                if use_stream:
                    with st.spinner("Searching knowledge graph & encyclopedia..."):
                        query_type = engine._classify_query(prompt)
                        context = engine._retrieve_context(prompt, query_type)
                        if engine.selection_layer:
                            context = engine.selection_layer.filter(
                                context, query=prompt
                            )
                        llm_prompt = engine._build_prompt(prompt, query_type, context)
                        sources = engine._extract_sources(context)

                    def stream_generator():
                        for chunk in engine.llm.generate_stream(
                            llm_prompt,
                            system_instruction=SYSTEM_INSTRUCTION
                        ):
                            yield chunk

                    streamed_answer = response_placeholder.write_stream(stream_generator())
                    result = QueryResult(
                        query=prompt,
                        query_type=query_type,
                        answer=streamed_answer,
                        context=context,
                        sources=sources
                    )
                    st.session_state.messages.append({"role": "assistant", "content": streamed_answer})
                    st.session_state.last_result = result
                else:
                    with st.spinner("Searching the encyclopedia..."):
                        result = engine.query(prompt)
                        response_placeholder.markdown(result.answer)
                        st.session_state.messages.append({"role": "assistant", "content": result.answer})
                        st.session_state.last_result = result
            except Exception as e:
                response_placeholder.error(f"Error querying encyclopedia: {e}")

    if "last_result" in st.session_state:
        st.divider()
        col1, col2 = st.columns([2, 1])
        result = st.session_state.last_result
        with col1:
            st.subheader("🕸️ Knowledge Graph Context")
            nodes, edges = get_graph_data_from_results(result)
            clicked_node = render_graph(nodes, edges)
        with col2:
            st.subheader("📄 Node Details")
            if clicked_node:
                all_retrieved = result.context.facts + result.context.analysis
                node_data = next((r.node for r in all_retrieved if r.node.get("id") == clicked_node or r.node.get("name") == clicked_node), None)
                if node_data:
                    st.success(f"**{node_data.get('name', 'N/A')}**")
                    st.write(f"**Type:** {node_data.get('type', 'N/A')}")
                    if "description" in node_data: 
                        st.markdown(f"**Description:**  \n{node_data['description']}")
                    if "analysis" in node_data: 
                        st.markdown(f"**Analysis:**  \n{node_data['analysis']}")
                    with st.expander("Raw Properties"): 
                        st.json(node_data)
                else: 
                    st.info("Node details not in current context.")
            else: 
                st.info("Click a node in the graph to see its details.")
            
            if show_sources:
                st.divider()
                st.subheader("📚 Top Sources")
                for i, source in enumerate(result.sources[:5]):
                    st.markdown(f"{i+1}. **{source['name']}**  \n`{source['type']}` | Score: {source['score']}")

# --- TAB 2: Graph Explorer & Cypher Workbench ---
with tab2:
    st.header("🔍 Global Graph Explorer")
    st.markdown("Explore the entire knowledge graph or run custom Cypher queries.")
    
    col_q, col_s = st.columns([3, 1])
    
    with col_s:
        st.subheader("📊 Schema Summary")
        if st.button("Refresh Schema"):
            get_cached_schema.clear()
            st.rerun()
        
        try:
            schema = get_cached_schema()
            st.write("**Node Labels:**")
            for label in schema.get('labels', []):
                st.caption(f"- {label}")
            st.write("**Relationship Types:**")
            for rel in schema.get('relationships', []):
                st.caption(f"- {rel}")
        except Exception as e:
            st.error(f"⚠️ Could not load schema from Neo4j: {e}")
            st.info(
                "💡 **Troubleshooting Tips:**\n"
                "- **Is AuraDB Paused?** Neo4j AuraDB Free tier automatically pauses after 3 days of inactivity. Go to [console.neo4j.io](https://console.neo4j.io) and check if your instance is Paused. If so, click **Resume**.\n"
                "- **Check Streamlit Cloud Secrets:** Ensure `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` are configured in Streamlit Cloud under **App settings > Secrets**.\n"
                "- **Check URI Format:** Your URI should look like `neo4j+s://<instance-id>.databases.neo4j.io` without extra quotes or trailing slashes."
            )

    with col_q:
        st.subheader("⌨️ Cypher Workbench")
        default_query = "MATCH (n)-[r]->(m) RETURN n, type(r) as rel, m LIMIT 25"
        cypher_query = st.text_area("Enter Cypher Query:", value=default_query, height=100)
        
        if st.button("Execute Query"):
            try:
                with st.spinner("Executing..."):
                    raw_results = neo4j.execute_query(cypher_query)
                
                if not raw_results:
                    st.warning("No results found.")
                else:
                    st.success(f"Found {len(raw_results)} records.")
                    
                    viz_nodes = {}
                    viz_edges = []
                    sanitized_results = []
                    
                    for record in raw_results:
                        sanitized_record = {}
                        # 1. Extract Nodes and Sanitize Data
                        for key, value in record.items():
                            if isinstance(value, dict) and ("id" in value or "name" in value or "title" in value):
                                n_id = value.get("id", value.get("name", value.get("title")))
                                if n_id and n_id not in viz_nodes:
                                    # Fix: Extract label from dict if present or use 'type'
                                    labels = [value.get("type", "")]
                                    viz_nodes[n_id] = Node(
                                        id=n_id,
                                        label=value.get("name", value.get("title", n_id)),
                                        size=20,
                                        color=get_color_for_labels(labels)
                                    )
                                sanitized_record[key] = n_id # Replace dict with ID for DataFrame
                            else:
                                sanitized_record[key] = str(value) # Stringify everything else (relationships, list)
                        
                        sanitized_results.append(sanitized_record)
                        
                        # 2. Extract Relationships (handle standard n, r, m return)
                        if 'n' in record and 'm' in record:
                            n_id = record['n'].get('id', record['n'].get('name', record['n'].get('title')))
                            m_id = record['m'].get('id', record['m'].get('name', record['m'].get('title')))
                            
                            rel_type = "RELATED_TO"
                            if 'r' in record:
                                if isinstance(record['r'], dict):
                                    rel_type = record['r'].get('type', "RELATED_TO")
                                else:
                                    rel_type = str(record['r'])
                            elif 'rel' in record:
                                rel_type = str(record['rel'])
                            
                            if n_id and m_id:
                                viz_edges.append(Edge(source=n_id, target=m_id, label=rel_type, color="#7f8c8d"))
                    
                    # Convert to lists
                    nodes_list = list(viz_nodes.values())
                    
                    st.divider()
                    v_col1, v_col2 = st.columns([2, 1])
                    
                    with v_col1:
                        st.subheader("Graph View")
                        if nodes_list:
                            render_graph(nodes_list, viz_edges, height=400)
                        else:
                            st.info("Query didn't return visualizable nodes.")
                    
                    with v_col2:
                        st.subheader("Raw Data Table")
                        st.dataframe(pd.DataFrame(sanitized_results))
            
            except Exception as e:
                st.error(f"Cypher Error: {e}")

    st.divider()
    st.subheader("🌐 Knowledge Map (Sample)")
    if st.button("Load Sample Graph"):
        with st.spinner("Fetching map..."):
            try:
                # Load a diverse sample of the graph
                sample_query = """
                MATCH (n)-[r]->(m)
                RETURN n, type(r) as rel_type, m
                LIMIT 50
                """
                sample_data = neo4j.execute_query(sample_query)
                
                s_nodes = {}
                s_edges = []
                
                for rec in sample_data:
                    n, m = rec['n'], rec['m']
                    n_id, m_id = n.get('id', n.get('name')), m.get('id', m.get('name'))
                    
                    if n_id and n_id not in s_nodes:
                        labels = [n.get("type", "")]
                        s_nodes[n_id] = Node(id=n_id, label=n.get('name', n_id), size=20, color=get_color_for_labels(labels))
                    if m_id and m_id not in s_nodes:
                        labels = [m.get("type", "")]
                        s_nodes[m_id] = Node(id=m_id, label=m.get('name', m_id), size=20, color=get_color_for_labels(labels))
                    
                    if n_id and m_id:
                        s_edges.append(Edge(source=n_id, target=m_id, label=rec.get('rel_type', ""), color="#7f8c8d"))
                
                render_graph(list(s_nodes.values()), s_edges, height=700)
            except Exception as e:
                st.error(f"Error loading sample graph: {e}")
