"""Data models for vector storage"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import numpy as np


@dataclass
class ChunkEmbedding:
    """Represents a chunk with its embedding"""
    chunk_id: str
    document_id: str
    text: str
    embedding: np.ndarray
    embedding_model: str
    section_heading: Optional[str] = None
    page_numbers: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'embedding': self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            'embedding_model': self.embedding_model,
            'section_heading': self.section_heading,
            'page_numbers': self.page_numbers,
            'created_at': self.created_at
        }


@dataclass
class SearchResult:
    """Represents a search result"""
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float
    metadata: dict
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'similarity_score': self.similarity_score,
            'metadata': self.metadata
        }
