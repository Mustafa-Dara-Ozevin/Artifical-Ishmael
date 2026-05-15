import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import logging
import pandas as pd

from src.query_engine import get_query_engine, QueryResult
from src.config import get_config, validate_config
from src.neo4j_client import get_neo4j_client

# Page configuration
st.set_page_config(
    page_title="🐋 Artifical Ishmael",
    page_icon="🐋",
    layout="wide"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---

@st.cache_resource
def init_engine():
    config = get_config()
    errors = validate_config(config)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    return get_query_engine()

@st.cache_resource
def init_neo4j():
    return get_neo4j_client()

engine = init_engine()
neo4j = init_neo4j()

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
        rel_results = neo4j.execute_query(query, {"ids": node_ids})
        for rel in rel_results:
            edges.append(Edge(source=rel["start_id"], target=rel["end_id"], label=rel["rel_type"], color="#7f8c8d"))
            
    return nodes, edges

# --- UI Layout ---

st.title("🐋 Moby-Dick GraphRAG Encyclopedia")

tab1, tab2 = st.tabs(["💬 Chat & Context", "🕸️ Graph Explorer"])

# --- TAB 1: Chat & Context ---
with tab1:
    st.markdown("Ask anything about Melville's masterpiece and see the knowledge graph in action.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        use_stream = st.checkbox("Stream Responses", value=True)
        show_sources = st.checkbox("Show Sources", value=True)
        st.divider()
        st.info("""
        **Legend:**
        * 🔵 **Facts**: Characters, Chapters, Locations
        * 🟠 **Analysis**: Themes, Symbols, Allusions
        """)

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
            with st.spinner("Searching the encyclopedia..."):
                result = engine.query(prompt)
            response_placeholder.markdown(result.answer)
            st.session_state.messages.append({"role": "assistant", "content": result.answer})
            st.session_state.last_result = result

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
            st.cache_data.clear()
        
        schema = neo4j.get_schema_summary()
        st.write("**Node Labels:**")
        for label in schema.get('labels', []):
            st.caption(f"- {label}")
        st.write("**Relationship Types:**")
        for rel in schema.get('relationships', []):
            st.caption(f"- {rel}")

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
