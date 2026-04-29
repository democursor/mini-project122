"""
Neo4j client wrapper using singleton driver
"""
import logging
from src.graph.neo4j_singleton import get_neo4j_driver, neo4j_retry

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client using singleton driver"""
    
    def __init__(self):
        """Initialize using singleton driver"""
        try:
            self.driver = get_neo4j_driver()
            self.database = "neo4j"  # Default database
            logger.debug("Neo4jClient initialized with singleton driver")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j client: {e}")
            self.driver = None
            self.database = None
    
    @neo4j_retry
    def is_connected(self) -> bool:
        """Check if Neo4j is connected"""
        try:
            if self.driver is None:
                self.driver = get_neo4j_driver()
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j connectivity check failed: {e}")
            return False
    
    def close(self):
        """Close is handled by singleton - no-op here"""
        logger.debug("Neo4j close called (singleton manages lifecycle)")
