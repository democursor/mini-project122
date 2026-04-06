"""
AI Assistant (RAG) endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import Optional

from src.api.models import (
    ChatRequest, ChatResponse,
    ChatSessionResponse, ChatSessionListResponse,
    ChatMessageListResponse
)
from src.services.chat_service import ChatService
from src.auth.dependencies import get_current_user
from src.auth.supabase_db import ChatDB

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
chat_service = ChatService()

def get_chat_db():
    return ChatDB()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ask a question to the AI research assistant
    
    Uses RAG (Retrieval-Augmented Generation) to provide answers
    grounded in your document collection
    """
    try:
        chat_db = get_chat_db()
        user_id = current_user["user_id"]
        
        # Create session if none provided
        session_id = request.session_id
        if not session_id:
            session = chat_db.create_session(user_id, "New Conversation")
            session_id = session["id"]
        
        # Get answer from chat service
        try:
            response = await chat_service.answer_question(
                question=request.question,
                conversation_history=request.conversation_history
            )
        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Chat service error: {str(e)}"
            )
        
        # Save user message
        chat_db.save_message(session_id, "user", request.question)
        
        # Convert Pydantic Citation objects to plain dicts
        citations_data = []
        if response.citations:
            for c in response.citations:
                if hasattr(c, 'dict'):
                    citations_data.append(c.dict())
                elif hasattr(c, 'model_dump'):
                    citations_data.append(c.model_dump())
                else:
                    citations_data.append(c)
        
        # Save assistant message with citations
        chat_db.save_message(
            session_id,
            "assistant",
            response.answer,
            citations_data
        )
        
        # Return response with session_id
        return ChatResponse(
            answer=response.answer,
            citations=response.citations,
            sources_count=len(response.citations),
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """
    List all chat sessions for the current user
    """
    try:
        chat_db = get_chat_db()
        sessions = chat_db.list_sessions(current_user["user_id"])
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all messages for a specific session
    """
    try:
        chat_db = get_chat_db()
        messages = chat_db.get_session_messages(session_id, current_user["user_id"])
        return {"messages": messages, "session_id": session_id}
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    title: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Update a session's title
    """
    try:
        chat_db = get_chat_db()
        session = chat_db.update_session_title(session_id, current_user["user_id"], title)
        return session
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a chat session and all its messages
    """
    try:
        chat_db = get_chat_db()
        chat_db.delete_session(session_id, current_user["user_id"])
        return {"message": "Session deleted", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
