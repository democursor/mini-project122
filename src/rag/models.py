"""Data models for RAG system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGContext:
    """Context retrieved for RAG generation."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    total_chunks_found: int
    retrieval_time: float
    context_length: int


@dataclass
class ConversationEntry:
    """Single conversation turn."""
    question: str
    response: str
    context: RAGContext
    citations: Dict[str, Any]
    timestamp: float


@dataclass
class RAGResponse:
    """Complete RAG response with metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    citations: Dict[str, Any]
    retrieval_stats: Dict[str, Any]
    conversation_id: Optional[str] = None
