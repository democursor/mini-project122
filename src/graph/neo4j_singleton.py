"""
Neo4j driver singleton with connection pooling and retry logic
"""
import logging
import os
import threading
from typing import Optional
from functools import wraps
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

logger = logging.getLogger(__name__)

# Thread-safe singleton
_neo4j_driver: Optional[Driver] = None
_neo4j_lock = threading.Lock()


def get_neo4j_driver() -> Driver:
    """
    Get Neo4j driver using thread-safe singleton pattern.
    Configured for Render free tier (512MB RAM).
    """
    global _neo4j_driver
    
    # Fast path
    if _neo4j_driver is not None:
        logger.debug("Reusing existing Neo4j driver (singleton)")
        return _neo4j_driver
    
    # Slow path with lock
    with _neo4j_lock:
        if _neo4j_driver is not None:
            return _neo4j_driver
        
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            logger.info(f"Creating Neo4j driver: {uri}")
            
            _neo4j_driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=5,        # Limit for free tier
                connection_timeout=10,              # Fail fast
                max_connection_lifetime=200,        # Recycle stale connections
                keep_alive=True                     # Keep connections alive
            )
            
            # Test connection
            _neo4j_driver.verify_connectivity()
            
            logger.info("Neo4j driver created successfully (singleton)")
            return _neo4j_driver
            
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            raise ConnectionError(f"Neo4j unavailable: {e}")


def neo4j_retry(func):
    """
    Decorator for Neo4j operations with automatic reconnection.
    Handles AuraDB Free pause/resume (72hr inactivity).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ServiceUnavailable, SessionExpired) as e:
            logger.warning(f"Neo4j connection stale/expired: {e}")
            logger.info("Attempting to reconnect to Neo4j...")
            
            # Force reconnection by clearing singleton
            global _neo4j_driver
            with _neo4j_lock:
                if _neo4j_driver:
                    try:
                        _neo4j_driver.close()
                    except:
                        pass
                    _neo4j_driver = None
            
            # Retry once with new connection
            try:
                get_neo4j_driver()
                logger.info("Neo4j reconnection successful, retrying operation")
                return func(*args, **kwargs)
            except Exception as retry_error:
                logger.error(f"Neo4j reconnection failed: {retry_error}")
                raise ConnectionError("Graph database temporarily unavailable")
        except Exception as e:
            logger.error(f"Neo4j operation failed: {e}")
            raise
    
    return wrapper


def check_neo4j_health() -> dict:
    """Check Neo4j connection health"""
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return {"status": "connected", "pool_size": 5}
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}
