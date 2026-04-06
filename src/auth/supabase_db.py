from supabase import create_client, Client
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)


def get_db() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables must be set. "
            "Please check your .env file."
        )
    
    logger.info(f"Creating Supabase client with URL: {url[:30]}...")
    
    try:
        client = create_client(url, key)
        logger.info("Supabase client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        raise RuntimeError(f"Failed to create Supabase client: {e}")


class DocumentDB:
    def __init__(self):
        self.db = get_db()

    def upsert_document(self, doc_data: dict) -> dict:
        try:
            result = self.db.table("documents").upsert(doc_data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error upserting document: {e}")
            raise

    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        try:
            result = (
                self.db.table("documents")
                .select("*")
                .eq("id", document_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None

    def get_document_by_id(self, document_id: str) -> Optional[dict]:
        try:
            result = (
                self.db.table("documents")
                .select("*")
                .eq("id", document_id)
                .single()
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting document by id: {e}")
            return None

    def list_documents(self, user_id: str) -> List[dict]:
        try:
            result = (
                self.db.table("documents")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            docs = result.data or []
            # Sort by upload_date in Python to avoid supabase-py order() bug
            return sorted(docs, key=lambda x: x.get("upload_date", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def update_document_status(
        self, document_id: str, status: str, metadata: dict = None
    ) -> dict:
        try:
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
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            raise

    def delete_document(self, document_id: str, user_id: str) -> bool:
        try:
            self.db.table("documents").delete().eq("id", document_id).eq(
                "user_id", user_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False


class ChatDB:
    def __init__(self):
        self.db = get_db()

    def create_session(self, user_id: str, title: str = "New Conversation") -> dict:
        try:
            result = (
                self.db.table("chat_sessions")
                .insert({"user_id": user_id, "title": title})
                .execute()
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise

    def list_sessions(self, user_id: str) -> List[dict]:
        try:
            result = (
                self.db.table("chat_sessions")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )
            sessions = result.data or []
            # Sort by updated_at in Python to avoid supabase-py order() bug
            return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def get_session_messages(self, session_id: str, user_id: str) -> List[dict]:
        try:
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
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list = None,
    ) -> dict:
        try:
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
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise
            raise

    def update_session_title(
        self, session_id: str, user_id: str, title: str
    ) -> dict:
        try:
            result = (
                self.db.table("chat_sessions")
                .update({"title": title})
                .eq("id", session_id)
                .eq("user_id", user_id)
                .execute()
            )
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.error(f"Error updating session title: {e}")
            raise

    def delete_session(self, session_id: str, user_id: str) -> bool:
        try:
            self.db.table("chat_sessions").delete().eq("id", session_id).eq(
                "user_id", user_id
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def save_search(self, user_id: str, query: str, results_count: int):
        try:
            self.db.table("search_history").insert(
                {
                    "user_id": user_id,
                    "query": query,
                    "results_count": results_count,
                }
            ).execute()
        except Exception as e:
            logger.error(f"Error saving search history: {e}")