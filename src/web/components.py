"""Reusable UI components for Streamlit app."""

import streamlit as st
from typing import List, Dict, Any


def render_message(message: Dict[str, Any]):
    """Render a chat message with sources and citations."""
    
    role = message['role']
    content = message['content']
    sources = message.get('sources', [])
    citations = message.get('citations', {})
    
    # Display message
    with st.chat_message(role):
        st.markdown(content)
        
        # Display sources if available
        if sources and role == 'assistant':
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    metadata = source.get('metadata', {})
                    text = source.get('text', '')
                    score = source.get('score', 0)
                    
                    # Extract metadata fields with fallbacks
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
                    
                    # Display source card
                    st.markdown(f"**[{i}] {title}**")
                    st.markdown(f"📝 *Authors:* {authors_str}")
                    st.markdown(f"📂 *Section:* {section}")
                    st.markdown(f"🎯 *Relevance Score:* {score:.3f}")
                    st.markdown(f"🆔 *Document ID:* `{doc_id[:20]}...`")
                    
                    # Show text preview
                    if text:
                        preview = text[:300] + "..." if len(text) > 300 else text
                        st.markdown(f"**Preview:**")
                        st.markdown(f"> {preview}")
                    
                    if i < len(sources):
                        st.divider()
        
        # Display citations if available
        if citations and role == 'assistant':
            total = citations.get('total_citations', 0)
            accuracy = citations.get('citation_accuracy', 0)
            
            if total > 0:
                with st.expander(f"📝 Citations ({total} total, {accuracy:.0%} accuracy)", expanded=False):
                    citation_list = citations.get('citations', [])
                    if citation_list:
                        for idx, citation in enumerate(citation_list[:10], 1):
                            status = "✅" if citation.get('valid', False) else "❌"
                            citation_text = citation.get('citation', 'Unknown citation')
                            st.markdown(f"{idx}. {status} {citation_text}")
                    else:
                        st.markdown("*No citations found in response*")


def render_document_card(doc: Dict[str, Any]):
    """Render a document card with dark theme styling."""
    
    title = doc.get('title', 'Untitled')
    authors = doc.get('authors', ['Unknown'])
    year = doc.get('year', 'N/A')
    doc_id = doc.get('document_id', 'unknown')
    abstract = doc.get('abstract', '')
    
    if isinstance(authors, list):
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += f" +{len(authors) - 3} more"
    else:
        authors_str = str(authors)
    
    # Create a dark theme card
    st.markdown(f"""
    <div style="
        background: #1C1F26;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        border: 1px solid #2D3139;
    ">
        <h3 style="margin: 0 0 0.5rem 0; color: #F1F3F6;">📄 {title}</h3>
        <p style="margin: 0.25rem 0; color: #AAB0B6;"><strong>Authors:</strong> {authors_str}</p>
        <p style="margin: 0.25rem 0; color: #AAB0B6;"><strong>Year:</strong> {year}</p>
        <p style="margin: 0.25rem 0; color: #AAB0B6; font-family: monospace; font-size: 0.85rem;"><strong>ID:</strong> {doc_id[:30]}...</p>
    </div>
    """, unsafe_allow_html=True)
    
    if abstract:
        with st.expander("📖 Abstract"):
            st.markdown(abstract[:500] + ("..." if len(abstract) > 500 else ""))


def render_search_result(result: Dict[str, Any], index: int):
    """Render a search result with dark theme styling."""
    
    text = result.get('text', '')
    metadata = result.get('metadata', {})
    score = result.get('score', 0)
    
    title = metadata.get('title', 'Unknown')
    doc_id = metadata.get('document_id', 'unknown')
    section = metadata.get('section_heading', 'N/A')
    
    # Determine score color
    if score >= 0.8:
        score_color = "#28a745"
    elif score >= 0.6:
        score_color = "#ffc107"
    else:
        score_color = "#dc3545"
    
    # Create dark theme result card
    st.markdown(f"""
    <div style="
        background: #1C1F26;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        border-left: 4px solid {score_color};
        border: 1px solid #2D3139;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #F1F3F6;">{index}. {title}</h4>
            <span style="
                background: {score_color};
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                font-size: 0.85rem;
                font-weight: 600;
            ">{score:.3f}</span>
        </div>
        <p style="margin: 0.5rem 0; color: #AAB0B6; font-size: 0.9rem;">
            <strong>Section:</strong> {section} | <strong>Doc ID:</strong> <code style="background: #0E1117; padding: 0.2rem 0.4rem; border-radius: 0.25rem;">{doc_id[:20]}...</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show text preview
    preview = text[:400] + "..." if len(text) > 400 else text
    with st.expander("📄 View Content"):
        st.markdown(f"> {preview}")


def render_stats_card(title: str, value: Any, icon: str = "📊"):
    """Render a statistics card."""
    
    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            text-align: center;
        ">
            <h3 style="margin: 0; color: #262730;">{icon} {title}</h3>
            <h1 style="margin: 0.5rem 0 0 0; color: #ff4b4b;">{value}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with navigation and info."""
    
    with st.sidebar:
        st.title("🤖 AI Research Assistant")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Go to:",
            ["💬 Chat", "📤 Upload", "📚 Library", "🔍 Search", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Info
        st.markdown("### About")
        st.markdown("""
        This AI Research Assistant helps you:
        - Upload and process research papers
        - Ask questions about your papers
        - Search semantically across documents
        - Get AI-generated answers with citations
        """)
        
        st.markdown("---")
        st.markdown("**Phase 6: Web Interface**")
        st.markdown("Built with Streamlit")
        
        return page


def show_success(message: str):
    """Show success message."""
    st.success(f"✅ {message}")


def show_error(message: str):
    """Show error message."""
    st.error(f"❌ {message}")


def show_warning(message: str):
    """Show warning message."""
    st.warning(f"⚠️ {message}")


def show_info(message: str):
    """Show info message."""
    st.info(f"ℹ️ {message}")
