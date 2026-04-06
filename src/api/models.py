"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Document Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    pages: Optional[int] = None
    upload_date: Optional[str] = None
    status: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int

# Search Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filters: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    document_id: str
    title: str
    authors: Optional[List[str]] = None
    excerpt: str
    score: float
    chunk_id: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int

# Graph Models
class GraphStats(BaseModel):
    total_papers: int
    total_concepts: int
    total_mentions: int
    total_relationships: int

class PaperNode(BaseModel):
    id: str
    title: str
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    abstract: Optional[str] = None

class ConceptNode(BaseModel):
    name: str
    frequency: int
    type: Optional[str] = None

class RelatedPaper(BaseModel):
    paper: PaperNode
    shared_concepts: int
    similarity_score: float

class ConceptSearchRequest(BaseModel):
    concept_name: str = Field(..., description="Concept to search for")
    limit: int = Field(default=10, ge=1, le=50)

class ConceptSearchResponse(BaseModel):
    concept: str
    papers: List[PaperNode]
    related_concepts: List[ConceptNode]

# Chat Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    conversation_history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = None

class Citation(BaseModel):
    document_id: str
    title: str
    excerpt: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    sources_count: int
    session_id: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str

class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    total: int

class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations: list = []
    created_at: str

class ChatMessageListResponse(BaseModel):
    messages: List[ChatMessageResponse]
    session_id: str
