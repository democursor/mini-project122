"""Vector storage using ChromaDB"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import numpy as np

from .models import ChunkEmbedding


class VectorStore:
    """Manages vector storage using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the main document collection"""
        collection_name = "research_papers"
        
        try:
            collection = self.client.get_collection(collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Research paper chunks with embeddings",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "created_at": datetime.now().isoformat()
                }
            )
            self.logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def add_embeddings(self, chunk_embeddings: List[ChunkEmbedding]):
        """Add chunk embeddings to vector store"""
        if not chunk_embeddings:
            return
        
        ids = [ce.chunk_id for ce in chunk_embeddings]
        embeddings = [ce.embedding.tolist() if isinstance(ce.embedding, np.ndarray) 
                     else ce.embedding for ce in chunk_embeddings]
        
        metadatas = []
        documents = []
        
        for ce in chunk_embeddings:
            metadata = {
                "document_id": ce.document_id,
                "chunk_id": ce.chunk_id,
                "embedding_model": ce.embedding_model,
                "created_at": ce.created_at
            }
            
            if ce.section_heading:
                metadata["section_heading"] = ce.section_heading
            if ce.page_numbers:
                metadata["page_numbers"] = ce.page_numbers
            
            metadatas.append(metadata)
            documents.append(ce.text)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        self.logger.info(f"Added {len(chunk_embeddings)} embeddings to vector store")
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # ChromaDB doesn't accept empty filters
        query_params = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filters and len(filters) > 0:
            query_params["where"] = filters
        
        results = self.collection.query(**query_params)
        
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i],
                "distance": results["distances"][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        count = self.collection.count()
        
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }
    
    def delete_document_chunks(self, document_id: str):
        """Delete all chunks for a specific document"""
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            self.logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
