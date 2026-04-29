"""
ChromaDB HTTP client singleton for Render deployment
"""
import chromadb
from chromadb.config import Settings
import os
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Thread-safe singleton
_chroma_client: Optional[chromadb.HttpClient] = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """
    Get ChromaDB client using thread-safe singleton pattern.
    Connects to external ChromaDB server if CHROMA_HOST is set,
    otherwise falls back to local PersistentClient for development.
    """
    global _chroma_client
    
    # Fast path
    if _chroma_client is not None:
        logger.debug("Reusing existing ChromaDB client (singleton)")
        return _chroma_client
    
    # Slow path with lock
    with _chroma_lock:
        if _chroma_client is not None:
            return _chroma_client
        
        try:
            host = os.getenv("CHROMA_HOST")
            
            # If CHROMA_HOST is set and not localhost, use HTTP client
            if host and host != "localhost":
                port = int(os.getenv("CHROMA_PORT", "8000"))
                logger.info(f"Creating ChromaDB HTTP client: {host}:{port}")
                
                _chroma_client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
                
                # Test connection
                _chroma_client.heartbeat()
                logger.info("ChromaDB HTTP client created successfully (singleton)")
            else:
                # Fall back to local PersistentClient for development
                persist_dir = "./data/chroma"
                logger.info(f"Creating ChromaDB PersistentClient: {persist_dir}")
                
                _chroma_client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info("ChromaDB PersistentClient created successfully (singleton)")
            
            return _chroma_client
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB client: {e}")
            raise ConnectionError(f"ChromaDB unavailable: {e}")


def check_chroma_health() -> dict:
    """Check ChromaDB connection health"""
    try:
        client = get_chroma_client()
        # Determine client type by checking class name
        client_type = "http" if "HttpClient" in str(type(client)) else "persistent"
        if hasattr(client, 'heartbeat'):
            client.heartbeat()
        return {"status": "connected", "type": client_type}
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}
