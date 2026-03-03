"""RAG (Retrieval-Augmented Generation) module for AI Research Assistant."""

from src.rag.retriever import RAGRetriever, RAGContext
from src.rag.llm_client import LLMClient
from src.rag.prompt_template import RAGPromptTemplate
from src.rag.citation_extractor import CitationExtractor
from src.rag.assistant import ResearchAssistant

__all__ = [
    "RAGRetriever",
    "RAGContext",
    "LLMClient",
    "RAGPromptTemplate",
    "CitationExtractor",
    "ResearchAssistant"
]
