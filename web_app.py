"""Streamlit Web Interface for AI Research Assistant."""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import components
from src.web.state import initialize_session_state, add_message, clear_chat, get_messages
from src.web.components import (
    render_message, render_document_card, render_search_result,
    render_stats_card, render_sidebar, show_success, show_error, show_info
)

# Import backend components
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.ingestion.uploader import PDFUploader
from src.ingestion.storage import PDFStorage
from src.ingestion.validator import PDFValidator
from src.orchestration.workflow import DocumentProcessor
from src.vector.store import VectorStore
from src.vector.embedder import EmbeddingGenerator, EmbeddingConfig
from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.rag.assistant import ResearchAssistant
from src.graph.builder import KnowledgeGraphBuilder
from src.graph.queries import GraphQueryEngine

# Setup logger
logger = logging.getLogger(__name__)


# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS with Dark Theme
st.markdown("""
<style>
    /* Global dark theme override */
    .main {
        background-color: #0E1117;
        color: #F1F3F6;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0E1117;
    }
    
    /* Header styling - Dark theme compatible */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .app-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Dark card styling */
    .dark-card {
        background: #1C1F26;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid #2D3139;
        margin-bottom: 1rem;
        color: #F1F3F6;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #1C1F26 !important;
        border: 1px solid #2D3139;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #667eea;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #764ba2;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.75rem;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling - Dark theme */
    div[data-testid="stExpander"] {
        background-color: #1C1F26 !important;
        border-radius: 0.75rem;
        border: 1px solid #2D3139;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stExpander"] summary {
        color: #F1F3F6 !important;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1C1F26;
        border-right: 1px solid #2D3139;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
        background-color: #1C1F26;
    }
    
    /* Metric card styling - Dark theme */
    .metric-card {
        background: #1C1F26;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid #2D3139;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #AAB0B6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    /* File upload area - Dark theme */
    .uploadedFile {
        background-color: #1C1F26 !important;
        border-radius: 0.75rem;
        border: 2px dashed #667eea !important;
        padding: 1.5rem;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #1C1F26;
        border-radius: 0.75rem;
        padding: 1rem;
        border: 2px dashed #2D3139;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background-color: #252830;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input styling - Dark theme */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #1C1F26;
        color: #F1F3F6;
        border: 1px solid #2D3139;
        border-radius: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Select box styling */
    .stSelectbox>div>div {
        background-color: #1C1F26;
        color: #F1F3F6;
        border: 1px solid #2D3139;
        border-radius: 0.75rem;
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background-color: #2D3139;
    }
    
    .stSlider>div>div>div>div {
        background-color: #667eea;
    }
    
    /* Success/Error/Warning/Info boxes - Dark theme */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
        color: #F1F3F6;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border-left: 4px solid #dc3545;
        color: #F1F3F6;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        color: #F1F3F6;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        color: #F1F3F6;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #F1F3F6;
        font-size: 1.5rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #AAB0B6;
    }
    
    /* Code block styling */
    code {
        background-color: #1C1F26;
        color: #667eea;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        border: 1px solid #2D3139;
    }
    
    pre {
        background-color: #1C1F26;
        border: 1px solid #2D3139;
        border-radius: 0.75rem;
        padding: 1rem;
    }
    
    /* Divider styling */
    hr {
        border-color: #2D3139;
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Override any remaining white backgrounds */
    .element-container {
        color: #F1F3F6;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #F1F3F6;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #F1F3F6;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #F1F3F6;
    }
    
    /* Ensure all text is readable */
    p, span, div, label {
        color: #F1F3F6;
    }
    
    /* Secondary text */
    .secondary-text {
        color: #AAB0B6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_components():
    """Initialize all backend components (cached)."""
    
    with st.spinner("🔄 Initializing AI Research Assistant..."):
        try:
            config = Config().config
            setup_logging(
                log_level=config["logging"]["level"],
                log_file=config["logging"]["file"]
            )
            
            # Show initialization progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize vector store
            status_text.text("📊 Loading vector database...")
            progress_bar.progress(20)
            vector_store = VectorStore(
                persist_directory=config["vector"]["persist_directory"]
            )
            
            # Initialize embedding generator
            status_text.text("🧠 Loading embedding model...")
            progress_bar.progress(40)
            embedding_generator = EmbeddingGenerator(
                config=EmbeddingConfig(
                    model_name=config["vector"]["embedding_model"],
                    batch_size=config["vector"]["batch_size"]
                )
            )
            
            # Initialize search engine
            status_text.text("🔍 Setting up search engine...")
            progress_bar.progress(60)
            query_processor = QueryProcessor(embedding_generator)
            search_engine = SemanticSearchEngine(vector_store, query_processor)
            
            # Initialize RAG components
            status_text.text("🤖 Initializing AI assistant...")
            progress_bar.progress(80)
            retriever = RAGRetriever(
                search_engine=search_engine,
                max_context_tokens=config["rag"]["max_context_tokens"]
            )
            
            # Get API key
            provider = config["rag"]["llm_provider"]
            if provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
            
            llm_client = LLMClient(
                provider=provider,
                model=config["rag"]["llm_model"],
                api_key=api_key
            )
            
            assistant = ResearchAssistant(
                retriever=retriever,
                llm_client=llm_client
            )
            
            # Initialize uploader and storage
            status_text.text("📁 Setting up document storage...")
            progress_bar.progress(90)
            storage = PDFStorage(config["storage"]["pdf_directory"])
            validator = PDFValidator(config["storage"]["max_file_size_mb"])
            uploader = PDFUploader(validator, storage)
            
            # Initialize workflow
            workflow = DocumentProcessor(config)
            
            # Initialize graph components (if Neo4j is configured)
            graph_builder = None
            graph_query_engine = None
            if config.get('neo4j'):
                try:
                    status_text.text("🕸️ Connecting to knowledge graph...")
                    graph_builder = KnowledgeGraphBuilder(
                        uri=config['neo4j']['uri'],
                        user=config['neo4j']['user'],
                        password=config['neo4j']['password'],
                        database=config['neo4j']['database']
                    )
                    graph_query_engine = GraphQueryEngine(
                        driver=graph_builder.driver,
                        database=config['neo4j']['database']
                    )
                    logger.info("Knowledge graph connected")
                except Exception as e:
                    logger.warning(f"Knowledge graph not available: {e}")
            
            progress_bar.progress(100)
            status_text.text("✅ Initialization complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            return {
                'config': config,
                'assistant': assistant,
                'search_engine': search_engine,
                'uploader': uploader,
                'workflow': workflow,
                'storage': storage,
                'graph_builder': graph_builder,
                'graph_query_engine': graph_query_engine
            }
        
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            return None


def chat_page(components):
    """Render chat page with modern UI."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">💬 AI Research Assistant</h1>
        <p class="app-subtitle">Ask questions about your research papers and get AI-powered answers with citations</p>
    </div>
    """, unsafe_allow_html=True)
    
    assistant = components['assistant']
    config = components['config']
    
    # Default top_k value
    top_k = config['rag']['top_k_retrieval']
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Chat Settings")
        
        # Model info
        with st.expander("🤖 Model Information", expanded=False):
            st.markdown(f"**Provider:** {config['rag']['llm_provider']}")
            st.markdown(f"**Model:** {config['rag']['llm_model']}")
            st.markdown(f"**Temperature:** {config['rag']['temperature']}")
        
        # Retrieval settings
        with st.expander("🔍 Retrieval Settings", expanded=False):
            top_k = st.slider(
                "Number of sources",
                min_value=1,
                max_value=10,
                value=config['rag']['top_k_retrieval'],
                help="Number of relevant chunks to retrieve"
            )
            st.markdown(f"**Max context:** {config['rag']['max_context_tokens']} tokens")
        
        st.markdown("---")
        
        # Conversation stats
        st.markdown("### 📊 Conversation Stats")
        conv_length = assistant.get_conversation_length()
        msg_count = len(get_messages())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Turns", conv_length)
        with col2:
            st.metric("Messages", msg_count)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            clear_chat()
            assistant.clear_conversation()
            st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        if not get_messages():
            st.info("� Welcome! Ask me anything about your research papers.")
        
        for message in get_messages():
            render_message(message)
    
    # Chat input
    if prompt := st.chat_input("💭 Ask a question about your papers...", key="chat_input"):
        # Add user message
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = assistant.ask_question(prompt, top_k=top_k)
                    
                    # Display answer
                    st.markdown(response.answer)
                    
                    # Add to history
                    add_message(
                        "assistant",
                        response.answer,
                        response.sources,
                        response.citations
                    )
                    
                    # Show sources
                    if response.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(response.sources, 1):
                                metadata = source.get('metadata', {})
                                text = source.get('text', '')
                                score = source.get('score', 0)
                                
                                # Extract metadata with fallbacks
                                title = metadata.get('title', metadata.get('document_id', 'Unknown Document'))
                                authors = metadata.get('authors', ['Unknown'])
                                section = metadata.get('section_heading', 'N/A')
                                doc_id = metadata.get('document_id', 'unknown')
                                
                                # Format authors
                                if isinstance(authors, list):
                                    authors_str = ", ".join(authors[:2])
                                    if len(authors) > 2:
                                        authors_str += " et al."
                                else:
                                    authors_str = str(authors)
                                
                                st.markdown(f"**[{i}] {title}**")
                                st.markdown(f"📝 *Authors:* {authors_str}")
                                st.markdown(f"📂 *Section:* {section}")
                                st.markdown(f"🎯 *Relevance:* {score:.3f}")
                                st.markdown(f"🆔 *ID:* `{doc_id[:20]}...`")
                                
                                # Show preview
                                if text:
                                    preview = text[:300] + "..." if len(text) > 300 else text
                                    st.markdown(f"**Preview:**")
                                    st.markdown(f"> {preview}")
                                
                                if i < len(response.sources):
                                    st.divider()
                    
                    # Show citations
                    if response.citations.get('total_citations', 0) > 0:
                        total = response.citations['total_citations']
                        accuracy = response.citations.get('citation_accuracy', 0)
                        with st.expander(f"📝 Citations ({total} total, {accuracy:.0%} accuracy)", expanded=False):
                            citation_list = response.citations.get('citations', [])
                            if citation_list:
                                for idx, citation in enumerate(citation_list[:10], 1):
                                    status = "✅" if citation.get('valid', False) else "❌"
                                    citation_text = citation.get('citation', 'Unknown')
                                    st.markdown(f"{idx}. {status} {citation_text}")
                            else:
                                st.markdown("*No citations found*")
                
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")


def upload_page(components):
    """Render upload page with modern UI."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📤 Upload Research Papers</h1>
        <p class="app-subtitle">Add PDF files to your research library for AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploader = components['uploader']
    workflow = components['workflow']
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### � Select PDF File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a research paper in PDF format (max 200MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Show file info in a nice card
            st.markdown(f"""
            <div style="background: #1C1F26; padding: 1.5rem; border-radius: 1rem; border-left: 4px solid #667eea; border: 1px solid #2D3139;">
                <h4 style="margin: 0; color: #667eea;">📄 {uploaded_file.name}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #AAB0B6;">Size: {uploaded_file.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Phase 1: Upload
                        status_text.markdown("**Phase 1/4:** 📤 Uploading and validating PDF...")
                        progress_bar.progress(25)
                        time.sleep(0.3)
                        doc_id = uploader.upload(uploaded_file, uploaded_file.name)
                        
                        # Phase 2: Parsing and chunking
                        status_text.markdown("**Phase 2/4:** 📝 Parsing document and creating chunks...")
                        progress_bar.progress(50)
                        time.sleep(0.3)
                        
                        # Phase 3: Knowledge graph
                        status_text.markdown("**Phase 3/4:** 🕸️ Building knowledge graph...")
                        progress_bar.progress(75)
                        time.sleep(0.3)
                        
                        # Phase 4: Embeddings
                        status_text.markdown("**Phase 4/4:** 🧠 Generating embeddings...")
                        progress_bar.progress(90)
                        
                        workflow.process_document(doc_id)
                        
                        # Complete
                        progress_bar.progress(100)
                        status_text.markdown("**✅ Processing complete!**")
                        time.sleep(0.5)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ Document processed successfully!\n\n**Document ID:** `{doc_id}`")
                        
                        # Clear cache to reload components
                        st.cache_resource.clear()
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Error processing document: {e}")
    
    with col2:
        st.markdown("### 📋 Upload Guidelines")
        
        st.markdown("""
        <div style="background: #1C1F26; padding: 1.5rem; border-radius: 1rem; border: 1px solid #2D3139;">
            <h4 style="margin-top: 0; color: #F1F3F6;">✅ Supported</h4>
            <ul style="margin-bottom: 1.5rem; color: #AAB0B6;">
                <li>PDF format only</li>
                <li>Research papers</li>
                <li>Academic articles</li>
                <li>Technical documents</li>
            </ul>
            
            <h4 style="color: #F1F3F6;">⚙️ Processing Steps</h4>
            <ol style="margin-bottom: 0; color: #AAB0B6;">
                <li>PDF validation</li>
                <li>Text extraction</li>
                <li>Semantic chunking</li>
                <li>Knowledge graph</li>
                <li>Vector embeddings</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


def library_page(components):
    """Render library page with modern UI."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">📚 Document Library</h1>
        <p class="app-subtitle">Browse and manage your uploaded research papers</p>
    </div>
    """, unsafe_allow_html=True)
    
    storage = components['storage']
    
    try:
        # Get all documents
        parsed_dir = Path("data/parsed")
        if not parsed_dir.exists():
            st.info("📭 No documents found. Upload some papers to get started!")
            return
        
        doc_files = list(parsed_dir.glob("*.json"))
        
        if not doc_files:
            st.info("📭 No documents found. Upload some papers to get started!")
            return
        
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Documents</div>
                <div class="metric-value">{len(doc_files)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pdf_dir = Path(components['config']["storage"]["pdf_directory"])
            pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.exists() else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PDF Files</div>
                <div class="metric-value">{pdf_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            try:
                vector_store = components['search_engine'].vector_store
                collection = vector_store.collection
                vector_count = collection.count()
            except:
                vector_count = "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Vector Embeddings</div>
                <div class="metric-value">{vector_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Storage Used</div>
                <div class="metric-value">{sum(f.stat().st_size for f in doc_files) / 1024 / 1024:.1f} MB</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Search/filter
        search_term = st.text_input("🔍 Search documents", placeholder="Search by title, author, or keyword...")
        
        st.markdown("### 📄 Documents")
        
        # Display documents in a grid
        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                
                # Filter by search term
                if search_term:
                    title = doc.get('title', '').lower()
                    authors = str(doc.get('authors', [])).lower()
                    if search_term.lower() not in title and search_term.lower() not in authors:
                        continue
                
                render_document_card(doc)
            except Exception as e:
                st.error(f"❌ Error loading {doc_file.name}: {e}")
    
    except Exception as e:
        st.error(f"❌ Error loading library: {e}")


def search_page(components):
    """Render search page with modern UI."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🔍 Semantic Search</h1>
        <p class="app-subtitle">Search across all your research papers using AI-powered semantic understanding</p>
    </div>
    """, unsafe_allow_html=True)
    
    search_engine = components['search_engine']
    
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g., machine learning applications in healthcare",
            label_visibility="collapsed"
        )
    
    with col2:
        top_k = st.selectbox(
            "Results",
            options=[5, 10, 15, 20],
            index=0,
            label_visibility="collapsed"
        )
    
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("🔎 Searching..."):
            try:
                start_time = time.time()
                results = search_engine.search(query, top_k=top_k)
                search_time = time.time() - start_time
                
                if results:
                    st.success(f"✅ Found {len(results)} results in {search_time:.2f}s")
                    
                    st.markdown("---")
                    st.markdown("### 📊 Search Results")
                    
                    for i, result in enumerate(results, 1):
                        render_search_result(result, i)
                else:
                    st.info("🔍 No results found. Try a different query.")
            
            except Exception as e:
                st.error(f"❌ Search error: {e}")
    
    # Search tips
    with st.expander("💡 Search Tips", expanded=False):
        st.markdown("""
        **How semantic search works:**
        - Uses AI to understand the meaning of your query
        - Finds relevant content even if exact words don't match
        - Ranks results by semantic similarity
        
        **Tips for better results:**
        - Use natural language questions
        - Be specific about what you're looking for
        - Try different phrasings if you don't find what you need
        - Use technical terms when appropriate
        """)


def knowledge_graph_page(components):
    """Render knowledge graph page with visualization."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🕸️ Knowledge Graph</h1>
        <p class="app-subtitle">Explore relationships between papers and concepts</p>
    </div>
    """, unsafe_allow_html=True)
    
    graph_query_engine = components.get('graph_query_engine')
    
    if not graph_query_engine:
        st.warning("⚠️ Knowledge Graph is not available. Neo4j must be running and configured.")
        st.info("""
        **To enable the Knowledge Graph:**
        1. Install and start Neo4j
        2. Configure connection in `config/default.yaml`
        3. Restart the application
        
        See `INSTALL_NEO4J_WINDOWS.md` for setup instructions.
        """)
        return
    
    try:
        # Get graph statistics
        stats = graph_query_engine.get_graph_statistics()
        
        # Display statistics
        st.markdown("### 📊 Graph Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📄 Papers</div>
                <div class="metric-value">{stats.get('papers', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🏷️ Concepts</div>
                <div class="metric-value">{stats.get('concepts', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🔗 Mentions</div>
                <div class="metric-value">{stats.get('mentions', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🌐 Relationships</div>
                <div class="metric-value">{stats.get('relationships', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Two-column layout for exploration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📄 Papers in Graph")
            
            # Get all papers
            papers = graph_query_engine.get_all_papers(limit=20)
            
            if papers:
                for paper in papers:
                    with st.expander(f"📄 {paper.get('title', 'Unknown')}", expanded=False):
                        st.markdown(f"**ID:** `{paper.get('id', 'N/A')[:30]}...`")
                        if paper.get('year'):
                            st.markdown(f"**Year:** {paper['year']}")
                        if paper.get('page_count'):
                            st.markdown(f"**Pages:** {paper['page_count']}")
                        
                        # Find related papers
                        if st.button(f"Find Related Papers", key=f"related_{paper.get('id')}"):
                            related = graph_query_engine.find_related_papers(paper.get('id'), limit=5)
                            if related:
                                st.markdown("**Related Papers:**")
                                for rel in related:
                                    st.markdown(f"- {rel.get('title')} ({rel.get('shared_concepts')} shared concepts)")
                            else:
                                st.info("No related papers found")
            else:
                st.info("📭 No papers in graph yet. Upload and process some papers!")
        
        with col2:
            st.markdown("### 🏷️ Top Concepts")
            
            # Get top concepts
            concepts = graph_query_engine.get_all_concepts(limit=20)
            
            if concepts:
                for i, concept in enumerate(concepts, 1):
                    concept_name = concept.get('name', 'Unknown')
                    concept_type = concept.get('type', 'N/A')
                    frequency = concept.get('frequency', 0)
                    
                    # Color code by frequency
                    if frequency >= 10:
                        color = "#28a745"  # Green
                    elif frequency >= 5:
                        color = "#667eea"  # Blue
                    else:
                        color = "#6c757d"  # Gray
                    
                    st.markdown(f"""
                    <div style="background: #1C1F26; padding: 1rem; border-radius: 0.75rem; border-left: 4px solid {color}; margin-bottom: 0.5rem; border: 1px solid #2D3139;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #F1F3F6;">{i}. {concept_name}</strong>
                                <span style="color: #AAB0B6; font-size: 0.85rem; margin-left: 0.5rem;">({concept_type})</span>
                            </div>
                            <div style="background: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.85rem; font-weight: 600;">
                                {frequency} mentions
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No concepts in graph yet")
        
        st.markdown("---")
        
        # Concept exploration
        st.markdown("### 🔍 Explore Concepts")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            concept_search = st.text_input(
                "Search for a concept",
                placeholder="e.g., COVID-19, machine learning, neural networks",
                key="concept_search"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and concept_search:
            with st.spinner("🔎 Searching knowledge graph..."):
                # Find papers mentioning this concept
                papers_with_concept = graph_query_engine.find_papers_by_concept(concept_search)
                
                # Find related concepts
                related_concepts = graph_query_engine.find_related_concepts(concept_search, limit=10)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"#### 📄 Papers mentioning '{concept_search}'")
                    if papers_with_concept:
                        for paper in papers_with_concept:
                            mentions = paper.get('mentions', 0)
                            confidence = paper.get('confidence', 0)
                            st.markdown(f"""
                            <div style="background: #1C1F26; padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.5rem; border: 1px solid #2D3139;">
                                <strong style="color: #F1F3F6;">{paper.get('title', 'Unknown')}</strong><br>
                                <span style="color: #AAB0B6; font-size: 0.85rem;">
                                    Year: {paper.get('year', 'N/A')} | 
                                    Mentions: {mentions} | 
                                    Confidence: {confidence:.2f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"No papers found mentioning '{concept_search}'")
                
                with col2:
                    st.markdown(f"#### 🔗 Related Concepts")
                    if related_concepts:
                        for concept in related_concepts:
                            name = concept.get('name', 'Unknown')
                            strength = concept.get('relationship_strength', 0)
                            papers_count = concept.get('papers_count', 0)
                            
                            st.markdown(f"""
                            <div style="background: #1C1F26; padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.5rem; border: 1px solid #2D3139;">
                                <strong style="color: #F1F3F6;">{name}</strong><br>
                                <span style="color: #AAB0B6; font-size: 0.85rem;">
                                    Strength: {strength:.2f} | 
                                    Co-occurs in {papers_count} papers
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"No related concepts found for '{concept_search}'")
        
        st.markdown("---")
        
        # Neo4j Browser link
        st.markdown("### 🌐 Advanced Visualization")
        st.info("""
        **For advanced graph visualization:**
        
        Open Neo4j Browser at [http://localhost:7474](http://localhost:7474)
        
        **Example Cypher Queries:**
        ```cypher
        // View all papers and concepts
        MATCH (p:Paper)-[r:MENTIONS]->(c:Concept)
        RETURN p, r, c LIMIT 50
        
        // Find most connected concepts
        MATCH (c:Concept)<-[r:MENTIONS]-()
        RETURN c.name, count(r) as mentions
        ORDER BY mentions DESC LIMIT 20
        
        // Find papers related through concepts
        MATCH (p1:Paper)-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Paper)
        WHERE p1 <> p2
        RETURN p1.title, p2.title, collect(c.name) as shared_concepts
        LIMIT 10
        ```
        """)
    
    except Exception as e:
        st.error(f"❌ Error loading knowledge graph: {e}")
        st.info("Make sure Neo4j is running and the connection is configured correctly.")


def settings_page(components):
    """Render settings page with modern UI."""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">⚙️ Settings & System Status</h1>
        <p class="app-subtitle">Monitor system performance and manage configuration</p>
    </div>
    """, unsafe_allow_html=True)
    
    config = components['config']
    
    # System stats
    st.markdown("### 📊 System Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count documents
        parsed_dir = Path("data/parsed")
        doc_count = len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📄 Documents</div>
            <div class="metric-value">{doc_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Count PDFs
        pdf_dir = Path(config["storage"]["pdf_directory"])
        pdf_count = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.exists() else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📁 PDF Files</div>
            <div class="metric-value">{pdf_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Vector store stats
        try:
            vector_store = components['search_engine'].vector_store
            collection = vector_store.collection
            vector_count = collection.count()
        except:
            vector_count = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔢 Vectors</div>
            <div class="metric-value">{vector_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Chat history
        msg_count = len(get_messages())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💬 Messages</div>
            <div class="metric-value">{msg_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration in two columns
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🤖 LLM Configuration", expanded=True):
            st.markdown(f"**Provider:** `{config['rag']['llm_provider']}`")
            st.markdown(f"**Model:** `{config['rag']['llm_model']}`")
            st.markdown(f"**Temperature:** `{config['rag']['temperature']}`")
            st.markdown(f"**Max Context Tokens:** `{config['rag']['max_context_tokens']}`")
            st.markdown(f"**Top-K Retrieval:** `{config['rag']['top_k_retrieval']}`")
        
        with st.expander("� Storage Configuration", expanded=False):
            st.markdown(f"**PDF Directory:** `{config['storage']['pdf_directory']}`")
            st.markdown(f"**Max File Size:** `{config['storage']['max_file_size_mb']} MB`")
            st.markdown(f"**Parsed Directory:** `data/parsed`")
    
    with col2:
        with st.expander("🧠 Vector Configuration", expanded=True):
            st.markdown(f"**Embedding Model:** `{config['vector']['embedding_model']}`")
            st.markdown(f"**Batch Size:** `{config['vector']['batch_size']}`")
            st.markdown(f"**Persist Directory:** `{config['vector']['persist_directory']}`")
        
        with st.expander("🔗 Neo4j Configuration", expanded=False):
            if config.get('neo4j'):
                st.markdown(f"**URI:** `{config['neo4j']['uri']}`")
                st.markdown(f"**Database:** `{config['neo4j']['database']}`")
                st.markdown(f"**User:** `{config['neo4j']['user']}`")
                st.markdown("**Status:** ✅ Configured")
            else:
                st.markdown("**Status:** ⚠️ Not configured")
    
    st.markdown("---")
    
    # Actions
    st.markdown("### 🔧 System Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reload Components", use_container_width=True):
            with st.spinner("Reloading..."):
                st.cache_resource.clear()
                time.sleep(0.5)
            st.success("✅ Components reloaded!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
    
    with col3:
        if st.button("💬 Clear Chat History", use_container_width=True):
            clear_chat()
            st.success("✅ Chat history cleared!")
            st.rerun()
    
    with col4:
        if st.button("📊 View Logs", use_container_width=True):
            log_file = Path(config['logging']['file'])
            if log_file.exists():
                with st.expander("📋 Recent Logs", expanded=True):
                    with open(log_file, 'r') as f:
                        logs = f.readlines()[-100:]  # Last 100 lines
                    st.code("".join(logs), language="log")
            else:
                st.info("📭 No log file found")


def main():
    """Main application with modern UI."""
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="color: #667eea; margin: 0; font-size: 1.8rem;">🔬 Research AI</h1>
            <p style="color: #6c757d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Intelligent Paper Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select a page",
            ["💬 Chat", "📤 Upload", "📚 Library", "🔍 Search", "🕸️ Knowledge Graph", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        parsed_dir = Path("data/parsed")
        doc_count = len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
        msg_count = len(get_messages())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", doc_count, delta=None)
        with col2:
            st.metric("Chats", msg_count, delta=None)
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **AI Research Assistant** helps you:
            
            - 📤 Upload research papers
            - 💬 Ask questions with AI
            - 🔍 Search semantically
            - 📚 Manage your library
            - 📊 Track citations
            
            Built with Streamlit, LangChain, and Google Gemini.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
            <p style="margin: 0;">Phase 6: Web Interface</p>
            <p style="margin: 0.25rem 0 0 0;">v1.0.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize components with loading state
    components = initialize_components()
    
    if components is None:
        st.error("❌ Failed to initialize. Please check your configuration.")
        st.stop()
    
    # Route to appropriate page
    if page == "💬 Chat":
        chat_page(components)
    elif page == "📤 Upload":
        upload_page(components)
    elif page == "📚 Library":
        library_page(components)
    elif page == "🔍 Search":
        search_page(components)
    elif page == "🕸️ Knowledge Graph":
        knowledge_graph_page(components)
    elif page == "⚙️ Settings":
        settings_page(components)


if __name__ == "__main__":
    main()
