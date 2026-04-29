from supabase import create_client, Client
from typing import List, Optional
import os
import logging
import threading
from functools import wraps

logger = logging.getLogger(__name__)

# Thread-safe singleton for Supabase client
_supabase_client: Optional[Client] = None
_supabase_lock = threading.Lock()


def get_db() -> Client:
    """
    Get Supabase client using thread-safe singleton pattern with 8-second timeout.
    Client is created once and reused across all calls.
    """
    global _supabase_client
    
    # Fast path: client already exists
    if _supabase_client is not None:
        logger.debug("Reusing existing Supabase client (singleton)")
        return _supabase_client
    
    # Slow path: need to create client (thread-safe)
    with _supabase_lock:
        # Double-check pattern: another thread might have created it
        if _supabase_client is not None:
            logger.debug("Reusing Supabase client created by another thread")
            return _supabase_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables must be set. "
                "Please check your .env file."
            )
        
        logger.info(f"Creating Supabase client with URL: {url[:30]}...")
        
        try:
            # Create client with timeout configuration
            import httpx
            _supabase_client = create_client(
                url, 
                key,
                options={
                    "postgrest_client_timeout": 8,
                    "storage_client_timeout": 8,
                }
            )
            logger.info("Supabase client created successfully (singleton initialized)")
            return _supabase_client
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            raise RuntimeError(f"Failed to create Supabase client: {e}")


def supabase_error_handler(func):
    """
    Decorator to handle Supabase pause/connection errors.
    Returns HTTP 503 instead of crashing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for connection/timeout errors
            if any(keyword in error_msg for keyword in [
                'timeout', 'connection', 'unreachable', 'refused', 'paused'
            ]):
                logger.error(f"Supabase connection error: {e}", exc_info=True)
                raise ConnectionError(
                    "Database temporarily unavailable, please try again in 30 seconds"
                )
            else:
                # Re-raise other errors
                logger.error(f"Supabase operation error: {e}", exc_info=True)
                raise
    
    return wrapper


def check_supabase_health() -> dict:
    """Check Supabase connection health"""
    try:
        client = get_db()
        # Simple query to test connection
        client.table("documents").select("id").limit(1).execute()
        return {"status": "connected", "timeout": "8s"}
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}


class DocumentDB:
    def __init__(self):
        self.db = get_db()

    @supabase_error_handler
    def upsert_document(self, doc_data: dict) -> dict:
        result = self.db.table("documents").upsert(doc_data).execute()
        return result.data[0] if result.data else {}

    @supabase_error_handler
    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        result = (
            self.db.table("documents")
            .select("*")
            .eq("id", document_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return result.data

    @supabase_error_handler
    def get_document_by_id(self, document_id: str) -> Optional[dict]:
        result = (
            self.db.table("documents")
            .select("*")
            .eq("id", document_id)
            .single()
            .execute()
        )
        return result.data

    @supabase_error_handler
    def list_documents(self, user_id: str) -> List[dict]:
        result = (
            self.db.table("documents")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        docs = result.data or []
        # Sort by upload_date in Python to avoid supabase-py order() bug
        return sorted(docs, key=lambda x: x.get("upload_date", ""), reverse=True)

    @supabase_error_handler
    def update_document_status(
        self, document_id: str, status: str, metadata: dict = None
    ) -> dict:
        update_data = {"status": status}
        if metadata:
            update_data["metadata"] = metadata
        result = (
            self.db.table("documents")
            .update(update_data)
            .eq("id", document_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    @supabase_error_handler
    def delete_document(self, document_id: str, user_id: str) -> bool:
        self.db.table("documents").delete().eq("id", document_id).eq(
            "user_id", user_id
        ).execute()
        return True


class ChatDB:
    def __init__(self):
        self.db = get_db()

    @supabase_error_handler
    def create_session(self, user_id: str, title: str = "New Conversation") -> dict:
        result = (
            self.db.table("chat_sessions")
            .insert({"user_id": user_id, "title": title})
            .execute()
        )
        return result.data[0] if result.data else {}

    @supabase_error_handler
    def list_sessions(self, user_id: str) -> List[dict]:
        result = (
            self.db.table("chat_sessions")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        sessions = result.data or []
        # Sort by updated_at in Python to avoid supabase-py order() bug
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    @supabase_error_handler
    def get_session_messages(self, session_id: str, user_id: str) -> List[dict]:
        # Verify session belongs to user
        session = (
            self.db.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not session.data:
            return []

        result = (
            self.db.table("chat_messages")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )
        messages = result.data or []
        # Sort by created_at in Python to avoid supabase-py order() bug
        return sorted(messages, key=lambda x: x.get("created_at", ""))

    @supabase_error_handler
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list = None,
    ) -> dict:
        # Convert Citation objects to dicts if needed
        citations_data = []
        if citations:
            for citation in citations:
                if hasattr(citation, 'model_dump'):
                    # Pydantic v2
                    citations_data.append(citation.model_dump())
                elif hasattr(citation, 'dict'):
                    # Pydantic v1
                    citations_data.append(citation.dict())
                elif isinstance(citation, dict):
                    citations_data.append(citation)
                else:
                    # Fallback: convert to dict manually
                    citations_data.append({
                        'document_id': getattr(citation, 'document_id', ''),
                        'title': getattr(citation, 'title', ''),
                        'excerpt': getattr(citation, 'excerpt', '')
                    })
        
        result = (
            self.db.table("chat_messages")
            .insert(
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "citations": citations_data,
                }
            )
            .execute()
        )
        # Touch updated_at on session
        self.db.table("chat_sessions").update(
            {"updated_at": "now()"}
        ).eq("id", session_id).execute()

        return result.data[0] if result.data else {}

    @supabase_error_handler
    def update_session_title(
        self, session_id: str, user_id: str, title: str
    ) -> dict:
        result = (
            self.db.table("chat_sessions")
            .update({"title": title})
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    @supabase_error_handler
    def delete_session(self, session_id: str, user_id: str) -> bool:
        self.db.table("chat_sessions").delete().eq("id", session_id).eq(
            "user_id", user_id
        ).execute()
        return True

    @supabase_error_handler
    def save_search(self, user_id: str, query: str, results_count: int):
        self.db.table("search_history").insert(
            {
                "user_id": user_id,
                "query": query,
                "results_count": results_count,
            }
        ).execute()