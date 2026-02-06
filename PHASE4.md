# Phase 4: Vector Storage and Semantic Search

## Overview

Phase 4 implements the semantic search capabilities that enable users to find relevant research papers using natural language queries. This phase creates vector embeddings for all document chunks and builds a fast similarity search system using ChromaDB.

**Learning Objectives:**
- Understand vector embeddings and semantic similarity
- Learn approximate nearest neighbor (ANN) search algorithms
- Master ChromaDB for vector storage and retrieval
- Implement hybrid search combining multiple signals
- Optimize search performance and relevance

**Key Concepts:**
- Dense vector representations of text
- Cosine similarity and distance metrics
- HNSW (Hierarchical Navigable Small World) indexing
- Approximate vs exact nearest neighbor search
- Embedding normalization and dimensionality
- Search result ranking and filtering

---

## Table of Contents

1. [Vector Storage Module](#vector-storage-module)
2. [Semantic Search Engine](#semantic-search-engine)
3. [Search Optimization](#search-optimization)
4. [Hybrid Search Strategies](#hybrid-search-strategies)
5. [Learning Outcomes](#learning-outcomes)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Success Criteria](#success-criteria)

---

## Vector Storage Module

### Purpose

The Vector Storage Module manages the creation, storage, and retrieval of semantic embeddings for document chunks. It provides the foundation for semantic similarity search by converting text into high-dimensional vectors that capture semantic meaning.

**Why Vector Search Matters:**
- **Semantic Understanding:** Finds conceptually similar content, not just keyword matches
- **Query Flexibility:** Works with natural language queries
- **Cross-Language Potential:** Embeddings can bridge language barriers
- **Scalability:** Efficient search across millions of documents

### 14.1 Embedding Generation

#### Sentence-BERT Implementation

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    max_seq_length: int = 512

class EmbeddingGenerator:
    """Generates semantic embeddings for text chunks"""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding generator.
        
        Model choices:
        - all-MiniLM-L6-v2: Fast, 384 dims, good quality
        - all-mpnet-base-v2: Best quality, 768 dims, slower
        - sentence-t5-base: Good for scientific text
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
        """
        self.config = config
        self.model = self._load_model()
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self) -> SentenceTransformer:
        """Load and configure sentence transformer model"""
        model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device
        )
        
        # Set maximum sequence length
        model.max_seq_length = self.config.max_seq_length
        
        # Optimize for inference
        if self.config.device == "cuda":
            model.half()  # Use FP16 for faster inference
        
        return model
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        self.logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()
```

---

#### Batch Processing for Efficiency

```python
class BatchEmbeddingProcessor:
    """Processes embeddings in memory-efficient batches"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator,
                 max_memory_mb: int = 1000):
        self.embedding_generator = embedding_generator
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
    
    def process_chunks_in_batches(self, chunks: List[Chunk]) -> List[ChunkEmbedding]:
        """
        Process chunks in memory-efficient batches.
        
        Strategy:
        1. Estimate memory usage per chunk
        2. Create batches that fit in memory limit
        3. Process each batch and yield results
        4. Clear memory between batches
        """
        chunk_embeddings = []
        current_batch = []
        current_memory = 0
        
        for chunk in chunks:
            # Estimate memory usage (rough approximation)
            chunk_memory = len(chunk.text) * 0.001  # MB
            
            if current_memory + chunk_memory > self.max_memory_mb and current_batch:
                # Process current batch
                batch_embeddings = self._process_batch(current_batch)
                chunk_embeddings.extend(batch_embeddings)
                
                # Reset for next batch
                current_batch = []
                current_memory = 0
            
            current_batch.append(chunk)
            current_memory += chunk_memory
        
        # Process final batch
        if current_batch:
            batch_embeddings = self._process_batch(current_batch)
            chunk_embeddings.extend(batch_embeddings)
        
        return chunk_embeddings
    
    def _process_batch(self, chunks: List[Chunk]) -> List[ChunkEmbedding]:
        """Process a single batch of chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        chunk_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_embedding = ChunkEmbedding(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                embedding=embedding,
                embedding_model=self.embedding_generator.config.model_name,
                created_at=datetime.now().isoformat()
            )
            chunk_embeddings.append(chunk_embedding)
        
        self.logger.info(f"Processed batch of {len(chunks)} chunks")
        return chunk_embeddings
```

---

### 14.2 Embedding Storage with ChromaDB

#### ChromaDB Setup and Configuration

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Any, Optional

class VectorStore:
    """Manages vector storage using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
        self.logger = logging.getLogger(__name__)
    
    def _get_or_create_collection(self):
        """Get or create the main document collection"""
        collection_name = "research_papers"
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            # Create new collection
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
        """
        Add chunk embeddings to vector store.
        
        Args:
            chunk_embeddings: List of chunk embeddings to store
        """
        if not chunk_embeddings:
            return
        
        # Prepare data for ChromaDB
        ids = [ce.chunk_id for ce in chunk_embeddings]
        embeddings = [ce.embedding.tolist() for ce in chunk_embeddings]
        
        # Prepare metadata
        metadatas = []
        documents = []
        
        for ce in chunk_embeddings:
            metadata = {
                "document_id": ce.document_id,
                "chunk_id": ce.chunk_id,
                "embedding_model": ce.embedding_model,
                "created_at": ce.created_at
            }
            
            # Add additional metadata if available
            if hasattr(ce, 'section_heading') and ce.section_heading:
                metadata["section_heading"] = ce.section_heading
            if hasattr(ce, 'page_numbers') and ce.page_numbers:
                metadata["page_numbers"] = ce.page_numbers
            
            metadatas.append(metadata)
            documents.append(ce.text if hasattr(ce, 'text') else "")
        
        # Add to collection
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
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar chunks with metadata
        """
        # Convert numpy array to list
        query_vector = query_embedding.tolist()
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
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
        # Get chunks for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        
        if results["ids"]:
            # Delete the chunks
            self.collection.delete(ids=results["ids"])
            self.logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
```

---

### 14.3 Vector Normalization

#### L2 Normalization Implementation

```python
import numpy as np
from sklearn.preprocessing import normalize

class EmbeddingNormalizer:
    """Handles embedding normalization for consistent similarity computation"""
    
    @staticmethod
    def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to embeddings.
        
        Why L2 normalization:
        - Makes cosine similarity equivalent to dot product
        - Ensures all vectors have unit length
        - Improves similarity computation stability
        - Required for some distance metrics
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            
        Returns:
            L2 normalized embeddings
        """
        if embeddings.ndim == 1:
            # Single embedding
            norm = np.linalg.norm(embeddings)
            if norm == 0:
                return embeddings
            return embeddings / norm
        else:
            # Multiple embeddings
            return normalize(embeddings, norm='l2', axis=1)
    
    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, 
                                 embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        For L2 normalized vectors: cos(θ) = A · B
        """
        # Ensure embeddings are normalized
        emb1_norm = EmbeddingNormalizer.l2_normalize(embedding1)
        emb2_norm = EmbeddingNormalizer.l2_normalize(embedding2)
        
        # Compute dot product (equivalent to cosine similarity for normalized vectors)
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for embeddings.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        # Normalize embeddings
        normalized = EmbeddingNormalizer.l2_normalize(embeddings)
        
        # Compute similarity matrix using matrix multiplication
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
```

---

### 14.4 Batch Processing

#### Efficient Batch Insertion

```python
class BatchVectorProcessor:
    """Handles batch processing for vector operations"""
    
    def __init__(self, vector_store: VectorStore, batch_size: int = 1000):
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def batch_insert_embeddings(self, chunk_embeddings: List[ChunkEmbedding]):
        """
        Insert embeddings in batches for memory efficiency.
        
        Strategy:
        1. Split embeddings into batches
        2. Insert each batch separately
        3. Handle errors gracefully
        4. Provide progress feedback
        """
        total_embeddings = len(chunk_embeddings)
        successful_inserts = 0
        failed_inserts = 0
        
        for i in range(0, total_embeddings, self.batch_size):
            batch = chunk_embeddings[i:i + self.batch_size]
            
            try:
                self.vector_store.add_embeddings(batch)
                successful_inserts += len(batch)
                
                # Progress logging
                progress = (i + len(batch)) / total_embeddings * 100
                self.logger.info(f"Batch insert progress: {progress:.1f}% ({successful_inserts}/{total_embeddings})")
                
            except Exception as e:
                self.logger.error(f"Failed to insert batch {i//self.batch_size + 1}: {e}")
                failed_inserts += len(batch)
                
                # Continue with next batch
                continue
        
        self.logger.info(f"Batch insert completed: {successful_inserts} successful, {failed_inserts} failed")
        
        return {
            "successful": successful_inserts,
            "failed": failed_inserts,
            "total": total_embeddings
        }
    
    def batch_search(self, queries: List[str], 
                    embedding_generator: EmbeddingGenerator,
                    top_k: int = 10) -> List[List[Dict]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of query strings
            embedding_generator: Generator for query embeddings
            top_k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        # Generate embeddings for all queries
        query_embeddings = embedding_generator.generate_embeddings(queries)
        
        # Perform searches
        all_results = []
        for i, (query, query_embedding) in enumerate(zip(queries, query_embeddings)):
            try:
                results = self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
                all_results.append(results)
                
            except Exception as e:
                self.logger.error(f"Search failed for query {i}: {e}")
                all_results.append([])  # Empty results for failed query
        
        return all_results
```

---

### 14.5 HNSW Indexing

#### Understanding HNSW Algorithm

```python
class HNSWIndexManager:
    """Manages HNSW indexing for fast approximate nearest neighbor search"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    def explain_hnsw(self) -> str:
        """
        Explain HNSW algorithm for educational purposes.
        
        HNSW (Hierarchical Navigable Small World):
        - Builds a multi-layer graph structure
        - Higher layers have fewer, more connected nodes
        - Lower layers have more nodes with local connections
        - Search starts at top layer and moves down
        - Provides sub-linear search complexity O(log n)
        
        Trade-offs:
        - Speed: Much faster than exact search
        - Accuracy: ~95-99% recall vs exact search
        - Memory: Additional memory for graph structure
        - Build time: Longer index construction
        """
        return """
        HNSW (Hierarchical Navigable Small World) Algorithm:
        
        Structure:
        - Multi-layer graph (typically 3-5 layers)
        - Layer 0: All vectors (densely connected locally)
        - Layer 1: Subset of vectors (medium connections)
        - Layer 2+: Fewer vectors (long-range connections)
        
        Search Process:
        1. Start at top layer with random entry point
        2. Greedily navigate to closest node in current layer
        3. Move down to next layer
        4. Repeat until reaching layer 0
        5. Perform local search in layer 0
        
        Performance:
        - Time Complexity: O(log n) average case
        - Space Complexity: O(n × M) where M is max connections
        - Recall: 95-99% vs exact search
        - Speed: 10-100x faster than brute force
        
        Parameters:
        - M: Maximum connections per node (16-64 typical)
        - efConstruction: Size of candidate set during construction
        - efSearch: Size of candidate set during search
        """
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the HNSW index"""
        collection_stats = self.vector_store.get_collection_stats()
        
        # ChromaDB uses HNSW internally, but doesn't expose detailed stats
        # In a production system, you might use a library like hnswlib directly
        
        return {
            "total_vectors": collection_stats["total_chunks"],
            "index_type": "HNSW (via ChromaDB)",
            "approximate_search": True,
            "expected_recall": "95-99%",
            "search_complexity": "O(log n)"
        }
```
---

## Semantic Search Engine

### Purpose

The Semantic Search Engine provides natural language search capabilities over the research paper collection. It combines vector similarity search with metadata filtering and result ranking to deliver relevant, contextual search results.

### 15.1 Query Processing

#### Query Embedding and Preprocessing

```python
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class SearchQuery:
    """Represents a processed search query"""
    original_query: str
    processed_query: str
    embedding: np.ndarray
    filters: Dict[str, Any]
    top_k: int = 10

class QueryProcessor:
    """Processes and optimizes search queries"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, query: str, filters: Optional[Dict] = None,
                     top_k: int = 10) -> SearchQuery:
        """
        Process raw query into structured search query.
        
        Steps:
        1. Clean and normalize query text
        2. Extract filters from query (if any)
        3. Generate query embedding
        4. Return structured query object
        """
        # Step 1: Clean query
        processed_query = self._clean_query(query)
        
        # Step 2: Extract implicit filters
        extracted_filters = self._extract_filters(processed_query)
        
        # Combine with explicit filters
        combined_filters = filters or {}
        combined_filters.update(extracted_filters)
        
        # Step 3: Generate embedding
        query_embedding = self.embedding_generator.generate_single_embedding(processed_query)
        
        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            embedding=query_embedding,
            filters=combined_filters,
            top_k=top_k
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that don't add semantic value
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', cleaned)
        
        # Convert to lowercase for consistency
        cleaned = cleaned.lower()
        
        return cleaned
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filters from query text.
        
        Examples:
        - "transformers after 2020" → {"year": {"$gte": 2020}}
        - "papers by Hinton" → {"author": "hinton"}
        - "NeurIPS papers" → {"venue": "neurips"}
        """
        filters = {}
        
        # Year filters
        year_match = re.search(r'after (\d{4})', query)
        if year_match:
            filters["year"] = {"$gte": int(year_match.group(1))}
        
        year_match = re.search(r'before (\d{4})', query)
        if year_match:
            filters["year"] = {"$lte": int(year_match.group(1))}
        
        year_match = re.search(r'in (\d{4})', query)
        if year_match:
            filters["year"] = int(year_match.group(1))
        
        # Author filters
        author_match = re.search(r'by ([a-zA-Z\s]+)', query)
        if author_match:
            author_name = author_match.group(1).strip().lower()
            filters["author"] = {"$contains": author_name}
        
        # Venue filters
        venues = ["neurips", "icml", "iclr", "aaai", "ijcai", "acl", "emnlp"]
        for venue in venues:
            if venue in query.lower():
                filters["venue"] = {"$contains": venue}
                break
        
        return filters
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        This can improve recall by finding documents that use
        different terminology for the same concepts.
        """
        # Simple synonym expansion (in production, use WordNet or custom thesaurus)
        synonyms = {
            "neural network": ["neural net", "deep network", "artificial neural network"],
            "machine learning": ["ml", "artificial intelligence", "ai"],
            "natural language processing": ["nlp", "computational linguistics"],
            "computer vision": ["cv", "image processing", "visual recognition"],
            "deep learning": ["dl", "deep neural network"]
        }
        
        expanded_queries = [query]
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns:
                    expanded_query = query.lower().replace(term, syn)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
```

---

### 15.2 Retrieval and Ranking

#### Multi-Stage Retrieval Pipeline

```python
class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self, vector_store: VectorStore, 
                 query_processor: QueryProcessor):
        self.vector_store = vector_store
        self.query_processor = query_processor
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, top_k: int = 10,
              filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search with multi-stage retrieval.
        
        Pipeline:
        1. Process query (clean, embed, extract filters)
        2. Vector similarity search (retrieve candidates)
        3. Re-rank results using multiple signals
        4. Apply post-processing filters
        5. Return final ranked results
        """
        # Stage 1: Process query
        search_query = self.query_processor.process_query(
            query, filters, top_k * 3  # Retrieve more for re-ranking
        )
        
        # Stage 2: Vector similarity search
        candidates = self.vector_store.search_similar(
            query_embedding=search_query.embedding,
            top_k=search_query.top_k,
            filters=search_query.filters
        )
        
        # Stage 3: Re-rank results
        ranked_results = self._rerank_results(search_query, candidates)
        
        # Stage 4: Apply final filters and limit
        final_results = ranked_results[:top_k]
        
        # Stage 5: Enhance results with additional metadata
        enhanced_results = self._enhance_results(final_results)
        
        self.logger.info(f"Search completed: {len(final_results)} results for '{query}'")
        return enhanced_results
    
    def _rerank_results(self, query: SearchQuery, 
                       candidates: List[Dict]) -> List[Dict]:
        """
        Re-rank search results using multiple signals.
        
        Ranking factors:
        1. Vector similarity score (primary)
        2. Query term frequency in document
        3. Document recency
        4. Document popularity (citation count)
        5. Section relevance (abstract > introduction > conclusion)
        """
        for result in candidates:
            # Start with vector similarity score
            base_score = result["similarity_score"]
            
            # Factor 1: Query term frequency boost
            term_boost = self._calculate_term_frequency_boost(
                query.processed_query, result["document"]
            )
            
            # Factor 2: Recency boost
            recency_boost = self._calculate_recency_boost(result["metadata"])
            
            # Factor 3: Section relevance boost
            section_boost = self._calculate_section_boost(result["metadata"])
            
            # Factor 4: Document popularity boost
            popularity_boost = self._calculate_popularity_boost(result["metadata"])
            
            # Combine scores (weighted average)
            final_score = (
                base_score * 0.6 +           # Vector similarity (primary)
                term_boost * 0.15 +          # Term frequency
                recency_boost * 0.1 +        # Recency
                section_boost * 0.1 +        # Section relevance
                popularity_boost * 0.05      # Popularity
            )
            
            result["final_score"] = final_score
            result["ranking_factors"] = {
                "similarity": base_score,
                "term_frequency": term_boost,
                "recency": recency_boost,
                "section": section_boost,
                "popularity": popularity_boost
            }
        
        # Sort by final score
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    
    def _calculate_term_frequency_boost(self, query: str, document: str) -> float:
        """Calculate boost based on query term frequency in document"""
        query_terms = query.lower().split()
        document_lower = document.lower()
        
        term_matches = sum(1 for term in query_terms if term in document_lower)
        
        if not query_terms:
            return 0.0
        
        return term_matches / len(query_terms)
    
    def _calculate_recency_boost(self, metadata: Dict) -> float:
        """Calculate boost based on document recency"""
        current_year = 2024  # Would be dynamic in real implementation
        
        if "year" in metadata and metadata["year"]:
            doc_year = int(metadata["year"])
            years_old = current_year - doc_year
            
            # Boost recent papers (exponential decay)
            if years_old <= 0:
                return 1.0
            elif years_old <= 2:
                return 0.8
            elif years_old <= 5:
                return 0.6
            else:
                return 0.4
        
        return 0.5  # Default for unknown year
    
    def _calculate_section_boost(self, metadata: Dict) -> float:
        """Calculate boost based on document section"""
        section_weights = {
            "abstract": 1.0,
            "introduction": 0.8,
            "conclusion": 0.7,
            "results": 0.6,
            "methods": 0.5,
            "references": 0.2
        }
        
        section = metadata.get("section_heading", "").lower()
        
        for section_name, weight in section_weights.items():
            if section_name in section:
                return weight
        
        return 0.5  # Default weight
    
    def _calculate_popularity_boost(self, metadata: Dict) -> float:
        """Calculate boost based on document popularity"""
        # In a real system, this would use citation counts
        # For now, use a placeholder
        citation_count = metadata.get("citation_count", 0)
        
        if citation_count > 1000:
            return 1.0
        elif citation_count > 100:
            return 0.8
        elif citation_count > 10:
            return 0.6
        else:
            return 0.4
    
    def _enhance_results(self, results: List[Dict]) -> List[Dict]:
        """Enhance results with additional metadata and formatting"""
        enhanced = []
        
        for result in results:
            enhanced_result = {
                "chunk_id": result["chunk_id"],
                "document_id": result["metadata"]["document_id"],
                "text": result["document"],
                "similarity_score": result["similarity_score"],
                "final_score": result["final_score"],
                "metadata": result["metadata"],
                "ranking_factors": result.get("ranking_factors", {}),
                
                # Add highlighting (simple version)
                "highlighted_text": self._highlight_text(
                    result["document"], 
                    result.get("query_terms", [])
                )
            }
            
            enhanced.append(enhanced_result)
        
        return enhanced
    
    def _highlight_text(self, text: str, query_terms: List[str]) -> str:
        """Add simple highlighting to matching terms"""
        highlighted = text
        
        for term in query_terms:
            # Simple highlighting with **bold** markers
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term}**", highlighted)
        
        return highlighted
```

---

### 15.3 Result Formatting

#### Search Result Aggregation

```python
class SearchResultAggregator:
    """Aggregates and formats search results"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def aggregate_by_document(self, chunk_results: List[Dict]) -> List[Dict]:
        """
        Aggregate chunk results by document.
        
        Strategy:
        1. Group chunks by document_id
        2. Calculate document-level relevance score
        3. Select best chunks per document
        4. Format document-level results
        """
        # Group by document
        doc_groups = {}
        for result in chunk_results:
            doc_id = result["metadata"]["document_id"]
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Aggregate each document
        aggregated_results = []
        for doc_id, chunks in doc_groups.items():
            doc_result = self._aggregate_document_chunks(doc_id, chunks)
            aggregated_results.append(doc_result)
        
        # Sort by document relevance
        aggregated_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return aggregated_results
    
    def _aggregate_document_chunks(self, doc_id: str, 
                                  chunks: List[Dict]) -> Dict:
        """Aggregate chunks for a single document"""
        # Calculate document relevance (max of chunk scores)
        relevance_score = max(chunk["final_score"] for chunk in chunks)
        
        # Select top chunks for this document
        top_chunks = sorted(chunks, key=lambda x: x["final_score"], reverse=True)[:3]
        
        # Extract document metadata from first chunk
        doc_metadata = chunks[0]["metadata"]
        
        return {
            "document_id": doc_id,
            "relevance_score": relevance_score,
            "title": doc_metadata.get("title", "Unknown Title"),
            "authors": doc_metadata.get("authors", []),
            "year": doc_metadata.get("year"),
            "venue": doc_metadata.get("venue"),
            "abstract": doc_metadata.get("abstract", ""),
            "top_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk["final_score"],
                    "section": chunk["metadata"].get("section_heading", "")
                }
                for chunk in top_chunks
            ],
            "total_matching_chunks": len(chunks)
        }
```

---

### 15.4 Metadata Filtering

#### Advanced Filtering System

```python
class MetadataFilter:
    """Handles complex metadata filtering for search results"""
    
    def __init__(self):
        self.supported_operators = {
            "$eq": self._equals,
            "$ne": self._not_equals,
            "$gt": self._greater_than,
            "$gte": self._greater_than_equal,
            "$lt": self._less_than,
            "$lte": self._less_than_equal,
            "$in": self._in_list,
            "$nin": self._not_in_list,
            "$contains": self._contains,
            "$regex": self._regex_match
        }
    
    def apply_filters(self, results: List[Dict], 
                     filters: Dict[str, Any]) -> List[Dict]:
        """
        Apply metadata filters to search results.
        
        Filter examples:
        - {"year": {"$gte": 2020}} - Papers from 2020 onwards
        - {"authors": {"$contains": "hinton"}} - Papers by Hinton
        - {"venue": {"$in": ["NeurIPS", "ICML"]}} - Specific venues
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            if self._matches_filters(metadata, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches all filters"""
        for field, condition in filters.items():
            if not self._matches_condition(metadata, field, condition):
                return False
        return True
    
    def _matches_condition(self, metadata: Dict, field: str, condition: Any) -> bool:
        """Check if a single condition matches"""
        field_value = metadata.get(field)
        
        if field_value is None:
            return False
        
        # Simple equality check
        if not isinstance(condition, dict):
            return field_value == condition
        
        # Complex condition with operators
        for operator, value in condition.items():
            if operator in self.supported_operators:
                if not self.supported_operators[operator](field_value, value):
                    return False
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        
        return True
    
    # Operator implementations
    def _equals(self, field_value: Any, filter_value: Any) -> bool:
        return field_value == filter_value
    
    def _not_equals(self, field_value: Any, filter_value: Any) -> bool:
        return field_value != filter_value
    
    def _greater_than(self, field_value: Any, filter_value: Any) -> bool:
        return field_value > filter_value
    
    def _greater_than_equal(self, field_value: Any, filter_value: Any) -> bool:
        return field_value >= filter_value
    
    def _less_than(self, field_value: Any, filter_value: Any) -> bool:
        return field_value < filter_value
    
    def _less_than_equal(self, field_value: Any, filter_value: Any) -> bool:
        return field_value <= filter_value
    
    def _in_list(self, field_value: Any, filter_value: List) -> bool:
        return field_value in filter_value
    
    def _not_in_list(self, field_value: Any, filter_value: List) -> bool:
        return field_value not in filter_value
    
    def _contains(self, field_value: Any, filter_value: str) -> bool:
        if isinstance(field_value, str):
            return filter_value.lower() in field_value.lower()
        elif isinstance(field_value, list):
            return any(filter_value.lower() in str(item).lower() for item in field_value)
        return False
    
    def _regex_match(self, field_value: Any, filter_value: str) -> bool:
        if isinstance(field_value, str):
            return bool(re.search(filter_value, field_value, re.IGNORECASE))
        return False
```

---

### 15.5 Search Performance Optimization

#### Caching and Performance Monitoring

```python
import time
from functools import lru_cache
import hashlib

class SearchPerformanceOptimizer:
    """Optimizes search performance through caching and monitoring"""
    
    def __init__(self, search_engine: SemanticSearchEngine):
        self.search_engine = search_engine
        self.query_cache = {}
        self.performance_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
            "slow_queries": []
        }
    
    def search_with_caching(self, query: str, top_k: int = 10,
                           filters: Optional[Dict] = None,
                           use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform search with caching and performance monitoring.
        
        Returns:
            Dictionary with results and performance metrics
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, top_k, filters)
        
        # Check cache
        if use_cache and cache_key in self.query_cache:
            results = self.query_cache[cache_key]
            response_time = time.time() - start_time
            
            self.performance_stats["cache_hits"] += 1
            self.performance_stats["total_queries"] += 1
            
            return {
                "results": results,
                "cached": True,
                "response_time": response_time,
                "total_results": len(results)
            }
        
        # Perform search
        results = self.search_engine.search(query, top_k, filters)
        response_time = time.time() - start_time
        
        # Cache results (if reasonable size)
        if use_cache and len(results) < 100:
            self.query_cache[cache_key] = results
        
        # Update performance stats
        self._update_performance_stats(query, response_time)
        
        return {
            "results": results,
            "cached": False,
            "response_time": response_time,
            "total_results": len(results)
        }
    
    def _generate_cache_key(self, query: str, top_k: int, 
                           filters: Optional[Dict]) -> str:
        """Generate cache key for query"""
        cache_data = {
            "query": query.lower().strip(),
            "top_k": top_k,
            "filters": filters or {}
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _update_performance_stats(self, query: str, response_time: float):
        """Update performance statistics"""
        self.performance_stats["total_queries"] += 1
        
        # Update average response time
        total = self.performance_stats["total_queries"]
        current_avg = self.performance_stats["average_response_time"]
        new_avg = (current_avg * (total - 1) + response_time) / total
        self.performance_stats["average_response_time"] = new_avg
        
        # Track slow queries (> 2 seconds)
        if response_time > 2.0:
            self.performance_stats["slow_queries"].append({
                "query": query,
                "response_time": response_time,
                "timestamp": time.time()
            })
            
            # Keep only last 100 slow queries
            if len(self.performance_stats["slow_queries"]) > 100:
                self.performance_stats["slow_queries"] = \
                    self.performance_stats["slow_queries"][-100:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics report"""
        total_queries = self.performance_stats["total_queries"]
        cache_hits = self.performance_stats["cache_hits"]
        
        cache_hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "average_response_time": f"{self.performance_stats['average_response_time']:.3f}s",
            "slow_queries_count": len(self.performance_stats["slow_queries"]),
            "cache_size": len(self.query_cache)
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
```

---

## Search Optimization

### Hybrid Search Strategies

#### Combining Vector and Keyword Search

```python
class HybridSearchEngine:
    """Combines vector similarity search with keyword search"""
    
    def __init__(self, semantic_engine: SemanticSearchEngine):
        self.semantic_engine = semantic_engine
        self.keyword_weights = {
            "vector_similarity": 0.7,
            "keyword_match": 0.3
        }
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword approaches.
        
        Strategy:
        1. Perform vector similarity search
        2. Perform keyword-based search (BM25-like)
        3. Combine and re-rank results
        4. Return unified results
        """
        # Vector similarity search
        vector_results = self.semantic_engine.search(query, top_k * 2)
        
        # Keyword search (simplified BM25)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_search_results(
            vector_results, keyword_results, query
        )
        
        return combined_results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Simplified keyword search using TF-IDF-like scoring.
        
        In production, this would use Elasticsearch or similar.
        """
        query_terms = query.lower().split()
        
        # This is a simplified implementation
        # In practice, you'd use a proper text search engine
        
        # For now, return empty results as this would require
        # a separate text index
        return []
    
    def _combine_search_results(self, vector_results: List[Dict],
                               keyword_results: List[Dict],
                               query: str) -> List[Dict]:
        """Combine and re-rank results from both search methods"""
        # Create unified result set
        all_results = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result["chunk_id"]
            all_results[chunk_id] = {
                **result,
                "vector_score": result["final_score"],
                "keyword_score": 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            if chunk_id in all_results:
                all_results[chunk_id]["keyword_score"] = result.get("score", 0.0)
            else:
                all_results[chunk_id] = {
                    **result,
                    "vector_score": 0.0,
                    "keyword_score": result.get("score", 0.0)
                }
        
        # Calculate hybrid scores
        for result in all_results.values():
            hybrid_score = (
                result["vector_score"] * self.keyword_weights["vector_similarity"] +
                result["keyword_score"] * self.keyword_weights["keyword_match"]
            )
            result["hybrid_score"] = hybrid_score
        
        # Sort by hybrid score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        return sorted_results
```

---

## Learning Outcomes

### Skills Learned in Phase 4

**1. Vector Embeddings**
- Understanding dense vector representations
- Sentence-BERT and transformer embeddings
- Embedding normalization and similarity metrics
- Batch processing for efficiency

**2. Vector Databases**
- ChromaDB setup and configuration
- Vector storage and retrieval
- HNSW indexing for fast search
- Metadata filtering and hybrid queries

**3. Semantic Search**
- Query processing and embedding
- Multi-stage retrieval pipelines
- Result ranking and re-ranking
- Performance optimization techniques

**4. Search Engineering**
- Caching strategies for performance
- Query expansion and preprocessing
- Result aggregation and formatting
- Hybrid search combining multiple signals

**5. Production Considerations**
- Memory management for large datasets
- Batch processing and error handling
- Performance monitoring and optimization
- Scalability and indexing strategies

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: "Slow embedding generation"**
- **Cause:** Large batch sizes, CPU inference
- **Solutions:**
  - Use GPU if available
  - Reduce batch size
  - Use smaller/faster models (MiniLM vs MPNet)

**Issue 2: "Poor search relevance"**
- **Cause:** Wrong embedding model, poor query processing
- **Solutions:**
  - Try domain-specific models
  - Improve query preprocessing
  - Adjust ranking factors
  - Use hybrid search

**Issue 3: "ChromaDB connection errors"**
- **Cause:** Persistence directory issues, version conflicts
- **Solutions:**
  - Check directory permissions
  - Update ChromaDB version
  - Clear and recreate collection

**Issue 4: "Out of memory during search"**
- **Cause:** Large result sets, memory leaks
- **Solutions:**
  - Limit result sizes
  - Implement result streaming
  - Clear caches regularly

**Issue 5: "Inconsistent similarity scores"**
- **Cause:** Non-normalized embeddings, different models
- **Solutions:**
  - Ensure L2 normalization
  - Use consistent embedding models
  - Validate embedding dimensions

---

## Success Criteria

Phase 4 is successful when:

✅ **Vector Storage**
- Embeddings are generated efficiently
- ChromaDB stores and retrieves vectors correctly
- HNSW indexing provides fast search

✅ **Search Quality**
- Semantic search returns relevant results
- Query processing handles natural language well
- Result ranking considers multiple factors

✅ **Performance**
- Search responds within 2 seconds
- Caching improves repeated query performance
- Memory usage is reasonable

✅ **Functionality**
- Metadata filtering works correctly
- Result aggregation groups chunks by document
- Hybrid search combines multiple signals

✅ **Scalability**
- System handles 10,000+ documents
- Batch processing manages large datasets
- Performance degrades gracefully with scale

---

## Next Steps

After completing Phase 4, you'll have:
- A fast semantic search engine
- Vector storage with efficient retrieval
- Understanding of embedding-based search systems

**Phase 5** will build on this foundation by:
- Implementing RAG (Retrieval-Augmented Generation)
- Creating an AI research assistant
- Combining search with language models
- Enabling conversational research queries

---

**Phase 4 demonstrates advanced search engineering skills including vector databases, semantic similarity, and search optimization - core competencies for roles in search, recommendation systems, and AI applications.**