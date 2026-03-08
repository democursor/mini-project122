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
            # Check if there are any documents in the vector store
            total_chunks = self.assistant.retriever.search_engine.vector_store.collection.count()
            
            if total_chunks == 0:
                return ChatResponse(
                    answer="I don't have any documents to answer your question. Please upload some research papers first.",
                    citations=[],
                    sources_count=0
                )
            
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
            
            # Check if any relevant sources were found
            if len(result.sources) == 0:
                return ChatResponse(
                    answer="I couldn't find any relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents.",
                    citations=[],
                    sources_count=0
                )
            
            # Extract citations
            citations = []
            for source in result.sources:
                # Get document metadata to include title
                doc_id = source.get('metadata', {}).get('document_id', source.get('document_id', ''))
                
                # Try to get document title from metadata file
                title = 'Untitled'
                try:
                    import json
                    from pathlib import Path
                    metadata_file = Path('./data/documents_metadata.json')
                    if metadata_file.exists():
                        metadata = json.loads(metadata_file.read_text())
                        if doc_id in metadata:
                            title = metadata[doc_id].get('filename', 'Untitled')
                except:
                    pass
                
                citations.append(Citation(
                    document_id=doc_id,
                    title=title,
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
