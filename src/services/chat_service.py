"""
Chat service - handles AI assistant (RAG) operations
"""
import logging
from typing import List, Optional

from src.rag.assistant import ResearchAssistant
from src.api.models import ChatMessage, ChatResponse, Citation
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class ChatService:
    """Service for AI chat assistant"""
    
    def __init__(self):
        self.config = load_config()
        self.assistant = ResearchAssistant(self.config)
    
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
            result = self.assistant.answer_question(question, history)
            
            # Extract citations
            citations = []
            for source in result.get('sources', []):
                citations.append(Citation(
                    document_id=source.get('document_id', ''),
                    title=source.get('title', 'Untitled'),
                    excerpt=source.get('text', '')[:300]
                ))
            
            return ChatResponse(
                answer=result.get('answer', ''),
                citations=citations,
                sources_count=len(citations)
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
