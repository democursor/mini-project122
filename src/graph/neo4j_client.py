"""
Neo4j client wrapper for connection management
"""
import logging
from neo4j import GraphDatabase
from src.utils.config import Config

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Simple Neo4j client for connection checking"""
    
    def __init__(self):
        """Initialize Neo4j connection"""
        try:
            config = Config()
            uri = config.get('neo4j.uri', 'bolt://localhost:7687')
            user = config.get('neo4j.user', 'neo4j')
            password = config.get('neo4j.password', 'password')
            database = config.get('neo4j.database', 'neo4j')
            
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.database = database
            logger.debug("Neo4jClient initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j client: {e}")
            self.driver = None
            self.database = None
    
    def is_connected(self) -> bool:
        """Check if Neo4j is connected and accessible"""
        try:
            if self.driver is None:
                return False
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j connectivity check failed: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            try:
                self.driver.close()
                logger.debug("Neo4j connection closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j connection: {e}")
