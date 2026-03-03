"""
AI Assistant (RAG) endpoints
"""
from fastapi import APIRouter, HTTPException
import logging

from src.api.models import ChatRequest, ChatResponse
from src.services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
chat_service = ChatService()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question to the AI research assistant
    
    Uses RAG (Retrieval-Augmented Generation) to provide answers
    grounded in your document collection
    """
    try:
        response = await chat_service.answer_question(
            question=request.question,
            conversation_history=request.conversation_history
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
