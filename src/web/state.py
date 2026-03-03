"""Session state management for Streamlit app."""

import streamlit as st
from typing import List, Dict, Any


def initialize_session_state():
    """Initialize session state variables."""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Processing status
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Current document
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None
    
    # Search results
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Components initialized flag
    if 'components_initialized' not in st.session_state:
        st.session_state.components_initialized = False


def add_message(role: str, content: str, sources: List[Dict] = None, citations: Dict = None):
    """Add a message to chat history."""
    message = {
        'role': role,
        'content': content,
        'sources': sources or [],
        'citations': citations or {}
    }
    st.session_state.messages.append(message)


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []


def get_messages() -> List[Dict[str, Any]]:
    """Get all messages."""
    return st.session_state.messages


def set_processing(status: bool):
    """Set processing status."""
    st.session_state.processing = status


def is_processing() -> bool:
    """Check if currently processing."""
    return st.session_state.processing
