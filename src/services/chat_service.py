"""
Chat service - handles AI assistant (RAG) operations
"""
import logging
from typing import List, Optional

from src.rag.assistant import ResearchAssistant
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.vector.embedder import EmbeddingGenerator, EmbeddingConfig
from src.vector.store import VectorStore
from src.api.models import ChatMessage, ChatResponse, Citation
from src.utils.config import Config

logger = logging.getLogger(__name__)

class ChatService:
    """Service for AI chat assistant"""
    
    def __init__(self):
        self.config = Config()
        
        # Initialize components
        embedding_config = EmbeddingConfig(
            model_name=self.config.get('vector.embedding_model', 'all-MiniLM-L6-v2'),
            device='cpu'
        )
        embedding_generator = EmbeddingGenerator(embedding_config)
        
        vector_store = VectorStore(
            persist_directory=self.config.get('vector.persist_directory', './data/chroma')
        )
        
        query_processor = QueryProcessor(embedding_generator)
        search_engine = SemanticSearchEngine(vector_store, query_processor)
        
        retriever = RAGRetriever(search_engine)
        
        llm_client = LLMClient(
            provider=self.config.get('rag.llm_provider', 'google'),
            model=self.config.get('rag.llm_model', 'gemini-pro'),
            api_key=None  # Will be loaded from environment
        )
        
        self.assistant = ResearchAssistant(retriever, llm_client)
    
    async def answer_question(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        """
        Answer a question using RAG
        
        Args:
            question: User question
            conversation_history: Previous conversation
            
        Returns:
            Chat response with answer and citations
        """
        try:
            # Convert conversation history
            history = []
            if conversation_history:
                for msg in conversation_history:
                    history.append({
                        'role': msg.role,
                        'content': msg.content
                    })
            
            # Get answer from assistant
            result = self.assistant.ask_question(question)
            
            # Extract citations
            citations = []
            for source in result.sources:
                citations.append(Citation(
                    document_id=source.get('document_id', ''),
                    title=source.get('title', 'Untitled'),
                    excerpt=source.get('text', '')[:300]
                ))
            
            return ChatResponse(
                answer=result.answer,
                citations=citations,
                sources_count=len(citations)
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
