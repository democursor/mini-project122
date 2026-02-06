# System Architecture: Autonomous Research Literature Intelligence & Discovery Platform

## Overview

This document explains the system architecture, component interactions, design patterns, and how the platform scales from a learning project to a production system. Understanding architecture is crucial for system design interviews.

---

## Table of Contents

1. [Architectural Principles](#architectural-principles)
2. [System Components](#system-components)
3. [Component Interactions](#component-interactions)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Scalability](#scalability)
7. [Interview Preparation](#interview-preparation)

---

## Architectural Principles

### 1. Separation of Concerns

**What it means**: Each component has a single, well-defined responsibility.

**Why it matters**:
- **Testability**: Test each component independently
- **Maintainability**: Change one component without breaking others
- **Extensibility**: Add new features by adding new components
- **Debugging**: Isolate problems to specific components

**Example**: PDF parsing is separate from concept extraction. If we want to change how we extract concepts, we don't touch the parser.

### 2. Modularity

**What it means**: Components are independent modules with clear interfaces.

**Why it matters**:
- **Reusability**: Use components in different contexts
- **Replaceability**: Swap implementations without changing interfaces
- **Parallel Development**: Different developers can work on different modules

**Example**: We can swap ChromaDB for Pinecone without changing the search engine, as long as the interface remains the same.

### 3. Loose Coupling

**What it means**: Components depend on interfaces, not concrete implementations.

**Why it matters**:
- **Flexibility**: Easy to change implementations
- **Testing**: Easy to mock dependencies
- **Evolution**: System can evolve without breaking changes

**Example**: The orchestrator depends on a "Parser" interface, not a specific PyMuPDF implementation.

### 4. High Cohesion

**What it means**: Related functionality is grouped together.

**Why it matters**:
- **Understandability**: Easy to understand what a component does
- **Maintainability**: Changes are localized
- **Reusability**: Components are self-contained

**Example**: All PDF-related operations (validation, storage, parsing) are in the PDF module.

### 5. Fail-Safe Design

**What it means**: System handles failures gracefully without cascading.

**Why it matters**:
- **Reliability**: One failure doesn't bring down the system
- **User Experience**: Clear error messages, not crashes
- **Debugging**: Failures are logged and traceable

**Example**: If concept extraction fails, we still store the parsed text and chunks.

---

## System Components

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Upload, Search, Chat)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│              (Workflow Management & State Tracking)             │
└─┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┘
  │          │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ PDF   │ │Parser │ │Chunker│ │Concept│ │ Graph │ │Vector │
│Ingest │ │       │ │       │ │Extract│ │Builder│ │Store  │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┴─────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   File   │  │  SQLite  │  │  Neo4j   │  │ ChromaDB │      │
│  │ Storage  │  │Metadata  │  │  Graph   │  │ Vectors  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query & Retrieval Layer                      │
│         ┌──────────────────┐    ┌──────────────────┐          │
│         │ Semantic Search  │    │  Graph Queries   │          │
│         │    (Vector)      │    │   (Cypher)       │          │
│         └──────────────────┘    └──────────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Research Assistant                        │
│                  (RAG with LLM Integration)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### 1. PDF Ingestion Module

**Responsibility**: Accept, validate, and store uploaded PDF files

**Inputs**: PDF file from user's local machine  
**Outputs**: Document ID, stored PDF file, metadata record

**Key Operations**:
- Validate file format and size
- Generate unique document ID (UUID)
- Store PDF in file system
- Create metadata record in SQLite
- Trigger processing workflow

**Error Handling**:
- Invalid format → Reject with clear message
- File too large → Reject with size limit
- Storage failure → Retry with exponential backoff

**Interface**:
```python
class PDFIngestionModule:
    def upload_pdf(self, file: BinaryIO, filename: str) -> DocumentID:
        """Upload and validate PDF file"""
        
    def validate_pdf(self, file: BinaryIO) -> bool:
        """Check if file is valid PDF"""
        
    def store_pdf(self, file: BinaryIO, doc_id: DocumentID) -> Path:
        """Store PDF in file system"""
```

**Why separate**: Ingestion concerns (validation, storage) are different from processing concerns (parsing, analysis).

#### 2. Parser Module

**Responsibility**: Extract text, metadata, and structure from PDFs

**Inputs**: PDF file path, document ID  
**Outputs**: Structured JSON with text, metadata, sections

**Key Operations**:
- Extract text using PyMuPDF
- Identify document structure (title, abstract, sections, references)
- Extract metadata (authors, date, venue)
- Handle multi-column layouts
- Preserve text positioning for structure

**Deep Learning**: None directly (rule-based extraction)

**Error Handling**:
- Corrupted PDF → Log error, mark document as failed
- Extraction failure → Retry with different method
- Missing metadata → Continue with partial data

**Output Format**:
```json
{
  "document_id": "uuid",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Vaswani et al."],
    "year": 2017,
    "venue": "NeurIPS"
  },
  "sections": [
    {"heading": "Abstract", "text": "...", "position": 0},
    {"heading": "Introduction", "text": "...", "position": 1}
  ]
}
```

**Interface**:
```python
class ParserModule:
    def parse_pdf(self, pdf_path: Path, doc_id: DocumentID) -> ParsedDocument:
        """Extract text and metadata from PDF"""
        
    def extract_metadata(self, pdf_path: Path) -> Metadata:
        """Extract title, authors, date, venue"""
        
    def extract_structure(self, pdf_path: Path) -> List[Section]:
        """Identify sections and hierarchy"""
```

**Why separate**: Parsing is distinct from semantic understanding. You might want to swap PDF parsers without changing downstream processing.

#### 3. Chunking Service

**Responsibility**: Segment documents into semantically coherent chunks

**Inputs**: Parsed document JSON  
**Outputs**: List of chunks with metadata

**Key Operations**:
- Use sentence embeddings to measure semantic similarity
- Identify natural boundaries (section breaks, topic shifts)
- Create chunks of 100-500 tokens
- Maintain references to source document and position

**Deep Learning**:
- **Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Purpose**: Compute sentence embeddings to detect semantic boundaries
- **How**: Compare consecutive sentence embeddings; large distance = boundary

**Algorithm**:
```
1. Split document into sentences
2. Compute embedding for each sentence
3. Calculate cosine similarity between consecutive sentences
4. When similarity drops below threshold, create chunk boundary
5. Respect section boundaries as hard boundaries
6. Ensure chunks are 100-500 tokens
```

**Why semantic chunking**:
- Fixed-size chunks can split concepts mid-thought
- Semantic chunks keep related ideas together
- Better retrieval: Get complete thoughts, not fragments

**Interface**:
```python
class ChunkingService:
    def chunk_document(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """Segment document into semantic chunks"""
        
    def compute_boundaries(self, sentences: List[str]) -> List[int]:
        """Identify semantic boundaries using embeddings"""
        
    def create_chunks(self, text: str, boundaries: List[int]) -> List[Chunk]:
        """Create chunks respecting boundaries and size constraints"""
```

#### 4. Concept Extraction Module

**Responsibility**: Identify key concepts, entities, and relationships

**Inputs**: Document chunks  
**Outputs**: Concepts, entities, relationships with confidence scores

**Key Operations**:
- Named Entity Recognition (NER) for researchers, methods, datasets
- Keyphrase extraction for important concepts
- Relationship extraction between entities
- Concept normalization (e.g., "BERT" and "Bidirectional Encoder" → same)

**Deep Learning**:
- **Model 1**: SpaCy en_core_sci_md (scientific NER)
  - Recognizes: PERSON, ORG, METHOD, DATASET, METRIC
- **Model 2**: KeyBERT (keyphrase extraction)
  - Uses BERT embeddings to find important phrases

**Example Output**:
```json
{
  "chunk_id": "uuid",
  "entities": [
    {"text": "transformer", "type": "METHOD", "confidence": 0.95},
    {"text": "ImageNet", "type": "DATASET", "confidence": 0.98}
  ],
  "keyphrases": [
    {"phrase": "self-attention mechanism", "score": 0.87}
  ],
  "relationships": [
    {"subject": "transformer", "predicate": "uses", "object": "self-attention"}
  ]
}
```

**Interface**:
```python
class ConceptExtractionModule:
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities using NER"""
        
    def extract_keyphrases(self, text: str) -> List[Keyphrase]:
        """Extract important phrases using KeyBERT"""
        
    def normalize_concepts(self, concepts: List[str]) -> List[str]:
        """Normalize concept names for consistency"""
```

#### 5. Knowledge Graph Builder

**Responsibility**: Construct and maintain the research knowledge graph

**Inputs**: Documents, concepts, entities, relationships  
**Outputs**: Graph database with nodes and edges

**Graph Schema**:
```
Nodes:
- Paper: {id, title, authors, year, abstract}
- Concept: {id, name, type, frequency}
- Author: {id, name}
- Venue: {id, name, type}

Edges:
- (Paper)-[:MENTIONS]->(Concept)
- (Paper)-[:CITES]->(Paper)
- (Paper)-[:AUTHORED_BY]->(Author)
- (Paper)-[:PUBLISHED_IN]->(Venue)
- (Concept)-[:RELATED_TO]->(Concept)
```

**Key Operations**:
- Create paper nodes with metadata
- Create concept nodes (or link to existing)
- Create relationships with weights
- Compute concept co-occurrence
- Update graph incrementally

**Example Queries**:
```cypher
// Find papers similar to a given paper
MATCH (p1:Paper {id: $paper_id})-[:MENTIONS]->(c:Concept)
      <-[:MENTIONS]-(p2:Paper)
RETURN p2.title, COUNT(c) as shared_concepts
ORDER BY shared_concepts DESC
LIMIT 10
```

**Interface**:
```python
class KnowledgeGraphBuilder:
    def create_paper_node(self, paper: Paper) -> None:
        """Create paper node in graph"""
        
    def create_concept_node(self, concept: Concept) -> None:
        """Create or update concept node"""
        
    def create_relationship(self, source: Node, target: Node, 
                          rel_type: str, properties: dict) -> None:
        """Create relationship between nodes"""
```


#### 6. Vector Store Module

**Responsibility**: Store embeddings and enable semantic similarity search

**Inputs**: Document chunks, embeddings  
**Outputs**: Similar chunks for a given query

**Key Operations**:
- Generate embeddings for each chunk using Sentence-BERT
- Store embeddings in ChromaDB with metadata
- Build HNSW index for fast approximate nearest neighbor search
- Query for top-k similar chunks

**Deep Learning**:
- **Model**: Sentence-BERT (all-MiniLM-L6-v2 or SciBERT)
- **Purpose**: Convert text to dense vector representations
- **Dimensions**: 384 (MiniLM) or 768 (SciBERT)

**How Embeddings Work**:
Each piece of text becomes a point in high-dimensional space. Texts with similar meanings are close together. Search finds nearest points to query.

**HNSW Index**:
- Hierarchical Navigable Small World graph
- Approximate nearest neighbor search in sub-linear time
- Trade-off: 95%+ accuracy, 100x faster than exact search

**Interface**:
```python
class VectorStoreModule:
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        
    def store_embedding(self, chunk_id: str, embedding: np.ndarray, 
                       metadata: dict) -> None:
        """Store embedding with metadata"""
        
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10) -> List[Chunk]:
        """Find top-k most similar chunks"""
```

#### 7. Semantic Search Engine

**Responsibility**: Provide natural language search over documents

**Inputs**: User query (natural language)  
**Outputs**: Ranked list of relevant documents and chunks

**Key Operations**:
- Convert query to embedding
- Retrieve top-k similar chunks from vector store
- Aggregate chunks by document
- Rank documents by relevance
- Apply metadata filters
- Return results with highlighted excerpts

**Search Pipeline**:
```
1. User query: "How do transformers work in computer vision?"
2. Generate query embedding
3. Vector search: Find 50 most similar chunks
4. Group chunks by document
5. Score documents by chunk similarities + metadata
6. Return top 10 documents with excerpts
```

**Interface**:
```python
class SemanticSearchEngine:
    def search(self, query: str, filters: dict = None, 
              top_k: int = 10) -> List[SearchResult]:
        """Search for documents matching query"""
        
    def rank_results(self, chunks: List[Chunk]) -> List[Document]:
        """Rank documents by relevance"""
        
    def apply_filters(self, results: List[Document], 
                     filters: dict) -> List[Document]:
        """Filter by metadata (date, author, etc.)"""
```

#### 8. AI Research Assistant (RAG System)

**Responsibility**: Answer questions using GenAI grounded in documents

**Inputs**: User question, conversation history  
**Outputs**: AI-generated answer with citations

**Key Operations**:
- Retrieve relevant chunks using semantic search
- Construct prompt with question + context
- Call LLM to generate answer
- Extract citations
- Maintain conversation context

**RAG Pipeline**:
```
1. User: "What are advantages of transformers over RNNs?"
2. Retrieve 5 most relevant chunks
3. Construct prompt with context
4. LLM generates answer with citations
5. Return answer
```

**Why RAG**:
- **Grounding**: Answers based on your documents
- **Citations**: Know which papers support claims
- **Up-to-date**: Works with latest papers
- **Accuracy**: Reduces hallucinations

**Interface**:
```python
class AIResearchAssistant:
    def answer_question(self, question: str, 
                       conversation_history: List[Message] = None) -> Answer:
        """Generate answer to question"""
        
    def retrieve_context(self, question: str) -> List[Chunk]:
        """Retrieve relevant chunks for question"""
        
    def generate_answer(self, question: str, 
                       context: List[Chunk]) -> str:
        """Generate answer using LLM"""
```

#### 9. LangGraph Orchestrator

**Responsibility**: Manage multi-step document processing workflow

**Why needed**:
- **State tracking**: Where is each document in pipeline?
- **Error handling**: What if parsing fails?
- **Retry logic**: Should we retry failed operations?
- **Concurrency**: Process multiple documents simultaneously
- **Observability**: Monitor progress

**Workflow**:
```
Upload → Parse → Chunk → Extract → Build Graph + Generate Embeddings → Complete
```

**State Management**:
```python
class DocumentState:
    document_id: str
    status: str  # "uploaded", "parsing", "chunking", etc.
    current_step: str
    retry_count: int
    error_message: Optional[str]
    parsed_data: Optional[dict]
    chunks: Optional[List[dict]]
    concepts: Optional[List[dict]]
```

**Error Handling**:
- Each step can succeed, fail, or retry
- Failed steps trigger retry with exponential backoff
- After 3 retries, mark document as failed
- Partial failures: If concept extraction fails, still store chunks

**Interface**:
```python
class LangGraphOrchestrator:
    def process_document(self, doc_id: DocumentID) -> None:
        """Start processing workflow for document"""
        
    def get_status(self, doc_id: DocumentID) -> DocumentState:
        """Get current processing status"""
        
    def retry_failed_step(self, doc_id: DocumentID) -> None:
        """Retry failed processing step"""
```

---

## Component Interactions

### Interaction Diagram: Processing a New Paper

```
User                PDF Ingestion    Orchestrator    Parser    Chunker    Concept    Graph    Vector
 |                       |                |            |          |          |         |        |
 |--upload PDF---------->|                |            |          |          |         |        |
 |                       |--validate----->|            |          |          |         |        |
 |                       |--store-------->|            |          |          |         |        |
 |                       |--trigger------>|            |          |          |         |        |
 |<--doc_id--------------|                |            |          |          |         |        |
 |                       |                |--parse---->|          |          |         |        |
 |                       |                |<--json-----|          |          |         |        |
 |                       |                |--chunk---------------->|          |         |        |
 |                       |                |<--chunks---------------|          |         |        |
 |                       |                |--extract------------------------>|         |        |
 |                       |                |<--concepts-----------------------|         |        |
 |                       |                |--build graph---------------------------->|        |
 |                       |                |--generate embeddings---------------------------->|
 |                       |                |<--complete-----------------------------------------|
 |<--notification--------|                |            |          |          |         |        |
```

### Data Flow Through Components

#### 1. Upload Phase
```
User uploads PDF
  ↓
PDF Ingestion validates file
  ↓
Store PDF in filesystem
  ↓
Create metadata record in SQLite
  ↓
Trigger LangGraph workflow
  ↓
Return document_id to user
```

#### 2. Processing Phase
```
LangGraph starts workflow
  ↓
Parser extracts text + metadata
  ↓
Chunker segments into semantic chunks
  ↓
Concept Extractor identifies entities + keyphrases
  ↓
[Parallel]
  ├─> Graph Builder creates nodes + relationships
  └─> Vector Store generates + stores embeddings
  ↓
Mark document as complete
  ↓
Notify user
```

#### 3. Query Phase
```
User submits search query
  ↓
Semantic Search converts query to embedding
  ↓
Vector Store finds similar chunks
  ↓
Search Engine ranks + filters results
  ↓
Return documents with excerpts
```

#### 4. Chat Phase
```
User asks question
  ↓
AI Assistant retrieves relevant chunks
  ↓
Construct prompt with context
  ↓
LLM generates answer
  ↓
Extract citations
  ↓
Return answer with sources
```

---

## Design Patterns

### 1. Pipeline Pattern

**What**: Data flows through a series of processing stages

**Where**: Document processing (Parse → Chunk → Extract → Store)

**Why**: 
- Clear data transformations
- Easy to add/remove stages
- Each stage is testable independently

**Example**:
```python
def process_document(pdf_path: Path) -> ProcessedDocument:
    parsed = parser.parse(pdf_path)
    chunks = chunker.chunk(parsed)
    concepts = extractor.extract(chunks)
    graph_builder.build(concepts)
    vector_store.store(chunks)
    return ProcessedDocument(parsed, chunks, concepts)
```

### 2. Repository Pattern

**What**: Abstract data access behind interfaces

**Where**: Database access (SQLite, Neo4j, ChromaDB)

**Why**:
- Decouple business logic from data storage
- Easy to swap databases
- Testable with mock repositories

**Example**:
```python
class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        pass
    
    @abstractmethod
    def find_by_id(self, doc_id: DocumentID) -> Optional[Document]:
        pass

class SQLiteDocumentRepository(DocumentRepository):
    def save(self, document: Document) -> None:
        # SQLite-specific implementation
        pass
```

### 3. Strategy Pattern

**What**: Swap algorithms at runtime

**Where**: Embedding models (MiniLM vs SciBERT), LLMs (OpenAI vs Ollama)

**Why**:
- Easy to switch implementations
- Compare different approaches
- Configure at runtime

**Example**:
```python
class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class MiniLMStrategy(EmbeddingStrategy):
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)

class SciBERTStrategy(EmbeddingStrategy):
    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text)
```

### 4. Observer Pattern

**What**: Notify interested parties when events occur

**Where**: Document processing status updates

**Why**:
- Decouple event producers from consumers
- Multiple observers can react to same event
- Easy to add new observers

**Example**:
```python
class ProcessingObserver(ABC):
    @abstractmethod
    def on_status_change(self, doc_id: DocumentID, status: str) -> None:
        pass

class NotificationObserver(ProcessingObserver):
    def on_status_change(self, doc_id: DocumentID, status: str) -> None:
        # Send notification to user
        pass
```

### 5. Circuit Breaker Pattern

**What**: Prevent cascading failures by stopping calls to failing services

**Where**: External API calls (OpenAI), database connections

**Why**:
- Fail fast instead of waiting for timeouts
- Give failing services time to recover
- Prevent resource exhaustion

**Example**:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5):
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable) -> Any:
        if self.state == "OPEN":
            raise CircuitBreakerOpenError()
        
        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

---

## Scalability

### Current Architecture (Local, Single-User)

**Characteristics**:
- Single machine
- Embedded databases (SQLite, ChromaDB)
- Local file storage
- Synchronous processing
- 10k documents capacity

**Bottlenecks**:
- SQLite: One writer at a time
- ChromaDB: Limited to single machine
- File storage: Limited by disk space
- Processing: Limited by CPU/GPU

### Scaling to Production (Multi-User, Cloud)

#### Phase 1: Vertical Scaling (Bigger Machine)

**Changes**:
- More CPU cores → Parallel processing
- More RAM → Larger models, more cache
- GPU → Faster embeddings
- SSD → Faster I/O

**Capacity**: 100k documents, 10 concurrent users

#### Phase 2: Horizontal Scaling (Multiple Machines)

**Changes**:
```
SQLite → PostgreSQL (supports concurrent writes)
ChromaDB → Milvus/Weaviate (distributed vector search)
Local files → S3 (cloud storage)
Sync processing → Async with message queue (Celery + Redis)
Single server → Load balancer + multiple API servers
```

**Architecture**:
```
                    Load Balancer
                         |
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    API Server 1    API Server 2    API Server 3
        |                |                |
        └────────────────┼────────────────┘
                         |
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   PostgreSQL         Milvus            S3
   (metadata)        (vectors)        (files)
```

**Capacity**: 1M documents, 100 concurrent users

#### Phase 3: Microservices (Large Scale)

**Changes**:
```
Monolith → Microservices
  - Ingestion Service
  - Processing Service
  - Search Service
  - Chat Service

Add caching layer (Redis)
Add monitoring (Prometheus + Grafana)
Add logging (ELK stack)
Add API gateway (Kong)
```

**Capacity**: 10M+ documents, 1000+ concurrent users

### Scaling Comparison

| Aspect | Local | Vertical | Horizontal | Microservices |
|--------|-------|----------|------------|---------------|
| **Documents** | 10k | 100k | 1M | 10M+ |
| **Users** | 1 | 10 | 100 | 1000+ |
| **Cost** | $0 | $100/mo | $1k/mo | $10k+/mo |
| **Complexity** | Low | Low | Medium | High |
| **Latency** | <1s | <1s | <2s | <2s |

---

## Interview Preparation

### System Design Questions

**Q: Design a semantic search system for research papers**

**Answer Structure**:
1. **Requirements**: Clarify scale, latency, features
2. **High-level design**: Show component diagram
3. **Deep dive**: Explain embeddings, vector search, ranking
4. **Scaling**: Discuss how to handle millions of documents
5. **Trade-offs**: Discuss accuracy vs speed, cost vs quality

**Q: How would you scale this to millions of documents?**

**Answer**:
- PostgreSQL for metadata (concurrent writes)
- Milvus for vectors (distributed, sharded)
- S3 for files (unlimited storage)
- Async processing with Celery + Redis
- Caching with Redis (frequent queries)
- Load balancing (multiple API servers)
- Monitoring (Prometheus + Grafana)

**Q: What are the bottlenecks in your current design?**

**Answer**:
- SQLite: One writer at a time
- ChromaDB: Single machine, limited scale
- Synchronous processing: Blocks on long operations
- No caching: Repeated queries recompute
- Single point of failure: One machine down = system down

**Q: How do you handle failures?**

**Answer**:
- Retry with exponential backoff (3 attempts)
- Circuit breaker for external APIs
- Graceful degradation (partial results if some components fail)
- Comprehensive logging for debugging
- Status tracking (know where each document is in pipeline)

### Key Talking Points

1. **Separation of Concerns**: Each component has single responsibility
2. **Modularity**: Easy to swap implementations
3. **Scalability**: Clear path from local to production
4. **Resilience**: Graceful error handling, no cascading failures
5. **Observability**: Logging, monitoring, status tracking
6. **Trade-offs**: Can justify every architectural decision

---

**This architecture balances simplicity for learning with production-grade principles, demonstrating system design thinking that companies value.**
