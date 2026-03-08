# Implementation Details

## Technical Design Decisions

### Why FastAPI?

FastAPI was chosen as the backend framework for several key reasons:

1. **Performance**: Built on Starlette and Pydantic, FastAPI offers exceptional performance comparable to NodeJS and Go
2. **Automatic API Documentation**: Built-in OpenAPI (Swagger) and ReDoc documentation generation
3. **Type Safety**: Leverages Python type hints for request/response validation
4. **Async Support**: Native async/await support for handling concurrent requests efficiently
5. **Modern Python**: Takes advantage of Python 3.7+ features for clean, maintainable code

### Why Neo4j?

Neo4j was selected as the knowledge graph database because:

1. **Native Graph Storage**: Purpose-built for storing and querying connected data
2. **Cypher Query Language**: Intuitive, SQL-like language for graph traversal
3. **Relationship Performance**: Constant-time relationship traversal regardless of database size
4. **Academic Use Case**: Ideal for representing research paper relationships (citations, concepts, authors)
5. **Visualization**: Built-in browser for exploring and visualizing the knowledge graph

### Why ChromaDB?

ChromaDB was chosen for vector storage due to:

1. **Simplicity**: Easy to set up and use with minimal configuration
2. **Embedding Support**: Native support for various embedding models
3. **Persistence**: Local persistence without requiring external services
4. **Performance**: Fast similarity search using HNSW algorithm
5. **Python-First**: Designed specifically for Python ML workflows

### Why Sentence Transformers?

Sentence Transformers (all-MiniLM-L6-v2) was selected for embeddings because:

1. **Quality**: State-of-the-art semantic similarity performance
2. **Speed**: Fast inference on CPU (384-dimensional embeddings)
3. **Size**: Lightweight model (~80MB) suitable for local deployment
4. **Pre-trained**: No fine-tuning required for general semantic search
5. **Community**: Well-maintained with extensive documentation

## Architecture Patterns

### Layered Architecture

The system follows a clean layered architecture:

```
Presentation Layer (Frontend)
    ↓
API Layer (FastAPI Routes)
    ↓
Service Layer (Business Logic)
    ↓
Data Access Layer (Repositories)
    ↓
Storage Layer (Databases)
```

**Benefits**:
- Clear separation of concerns
- Easy to test individual layers
- Flexible to swap implementations
- Maintainable and scalable

### Repository Pattern

Each data source (Neo4j, ChromaDB, File Storage) has a dedicated repository:

- `KnowledgeGraphBuilder`: Neo4j operations
- `VectorStore`: ChromaDB operations
- `PDFStorage`: File system operations

**Benefits**:
- Abstracts database-specific logic
- Enables easy mocking for tests
- Centralizes data access patterns

### Workflow Orchestration with LangGraph

Document processing uses LangGraph for workflow orchestration:

```python
parse → chunk → extract → build_graph → store_vectors → complete
```

**Benefits**:
- Visual workflow representation
- Error handling at each step
- Easy to add/modify processing stages
- State management across steps

## Background Processing

### Why Background Tasks?

Document processing is CPU-intensive and can take 30-60 seconds. Using FastAPI's `BackgroundTasks`:

1. **Non-blocking**: API returns immediately with document ID
2. **User Experience**: Frontend can poll for status updates
3. **Resource Management**: Prevents request timeouts
4. **Scalability**: Can be moved to Celery/RQ for distributed processing

### Implementation

```python
@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    doc_id = await document_service.upload_and_process(
        file_path=tmp_path,
        filename=file.filename,
        background_tasks=background_tasks
    )
    return {"document_id": doc_id, "status": "processing"}
```

## Ingestion Pipeline

### Phase 1: PDF Validation & Storage

**Validation Steps**:
1. Check file format (PDF magic bytes)
2. Verify file size (< 50MB default)
3. Test PDF integrity with PyMuPDF

**Storage Strategy**:
- Organized by year/month: `data/pdfs/2024/03/doc_abc123.pdf`
- Enables efficient cleanup and archival
- Document ID embedded in filename

### Phase 2: PDF Parsing

**PyMuPDF (fitz) Features**:
- Extract text with layout preservation
- Parse metadata (title, authors, year)
- Detect sections using font size/style heuristics
- Handle multi-column layouts

**Section Detection**:
```python
# Larger font = heading
if font_size > avg_font_size * 1.2:
    current_section = text
```

### Phase 3: Semantic Chunking

**Strategy**: Boundary-aware chunking

1. **Sentence Splitting**: Use NLTK to split into sentences
2. **Token Counting**: Track tokens per chunk (target: 512 tokens)
3. **Boundary Detection**: Prefer breaking at section boundaries
4. **Overlap**: 50-token overlap between chunks for context

**Why Not Fixed-Size Chunks?**
- Preserves semantic coherence
- Respects document structure
- Better retrieval quality

### Phase 4: Concept Extraction

**Two-Stage Approach**:

1. **Named Entity Recognition (NER)**:
   - Model: `en_core_sci_sm` (scientific domain)
   - Extracts: methods, datasets, metrics, concepts
   
2. **Keyphrase Extraction**:
   - Model: KeyBERT with sentence transformers
   - Extracts: important multi-word phrases
   - Scored by relevance

**Why Both?**
- NER: Structured entities with types
- KeyBERT: Domain-agnostic important phrases
- Complementary coverage

### Phase 5: Knowledge Graph Construction

**Node Types**:
- `Paper`: Research documents
- `Author`: Paper authors
- `Concept`: Extracted concepts/entities
- `Section`: Document sections

**Relationship Types**:
- `AUTHORED_BY`: Paper → Author
- `CONTAINS_CONCEPT`: Paper → Concept
- `HAS_SECTION`: Paper → Section
- `MENTIONS`: Section → Concept

**Graph Building Process**:
```python
1. Create Paper node with metadata
2. Create Author nodes and AUTHORED_BY relationships
3. Create Concept nodes from extracted entities
4. Create CONTAINS_CONCEPT relationships with frequency
5. Create Section nodes with HAS_SECTION relationships
```

### Phase 6: Vector Embedding & Storage

**Embedding Generation**:
- Model: `all-MiniLM-L6-v2`
- Dimension: 384
- Batch processing for efficiency

**ChromaDB Storage**:
```python
{
    "chunk_id": "doc_abc_chunk_0",
    "document_id": "doc_abc",
    "text": "chunk content...",
    "embedding": [0.123, -0.456, ...],  # 384 dims
    "metadata": {
        "section_heading": "Introduction",
        "title": "Paper Title",
        "authors": ["Author 1"]
    }
}
```

**Indexing Strategy**:
- HNSW algorithm for fast approximate nearest neighbor search
- Cosine similarity metric
- Automatic persistence to disk

## Search Implementation

### Semantic Search Flow

1. **Query Processing**:
   ```python
   query → embedding_generator → query_vector
   ```

2. **Vector Search**:
   ```python
   query_vector → chromadb.query() → top_k chunks
   ```

3. **Result Ranking**:
   - Cosine similarity scores
   - Metadata filtering (optional)
   - Deduplication by document

### Query Expansion (Future)

Potential improvements:
- Synonym expansion using WordNet
- Query reformulation using LLM
- Hybrid search (keyword + semantic)

## RAG (Retrieval-Augmented Generation)

### Architecture

```
User Question
    ↓
Semantic Search (retrieve relevant chunks)
    ↓
Context Assembly (top 5 chunks)
    ↓
Prompt Construction (question + context)
    ↓
LLM Generation (Google Gemini / OpenAI)
    ↓
Answer with Citations
```

### Prompt Template

```python
"""
You are a research assistant. Answer the question based on the provided context.

Context:
{context_chunks}

Question: {question}

Provide a detailed answer with citations [1], [2], etc.
"""
```

### Citation Extraction

- Parse LLM response for citation markers `[1]`, `[2]`
- Map citations back to source chunks
- Return document IDs and excerpts

## Error Handling

### Validation Errors

```python
try:
    validation_result = validator.validate(file_path)
    if not validation_result.is_valid:
        raise HTTPException(400, detail=validation_result.errors)
except Exception as e:
    logger.error(f"Validation failed: {e}")
    raise HTTPException(500, detail="Internal error")
```

### Database Connection Errors

```python
try:
    graph_builder = KnowledgeGraphBuilder(...)
except Exception as e:
    logger.warning(f"Neo4j unavailable: {e}")
    graph_builder = None  # Graceful degradation
```

### Processing Failures

- Each workflow step has error handling
- Failed documents marked with status="failed"
- Error messages stored in metadata
- Partial results preserved (e.g., parsing succeeds but graph fails)

## Performance Optimizations

### Batch Processing

```python
# Generate embeddings in batch
embeddings = embedding_generator.generate_embeddings(texts)
```

### Connection Pooling

- Neo4j: Connection pool managed by driver
- ChromaDB: Persistent client connection

### Caching

- Embedding model loaded once at startup
- NLP models cached in memory
- Configuration loaded once

### Async Operations

```python
# Non-blocking API endpoints
async def upload_document(...):
    background_tasks.add_task(process_document)
    return {"status": "processing"}
```

## Testing Strategy

### Unit Tests

- Individual component testing
- Mock external dependencies
- Fast execution

### Integration Tests

- Test component interactions
- Use test databases
- Slower but comprehensive

### End-to-End Tests

- Full pipeline testing
- Real PDF documents
- Verify all systems working together

## Security Considerations

### Input Validation

- File type checking (PDF only)
- File size limits (50MB default)
- PDF integrity verification

### API Security

- CORS configuration for frontend
- Rate limiting (future)
- Authentication (future)

### Data Privacy

- Local storage (no cloud dependencies)
- No external API calls for embeddings
- User data stays on-premises

## Scalability Considerations

### Current Limitations

- Single-server deployment
- Synchronous processing
- Local file storage

### Future Improvements

1. **Distributed Processing**:
   - Celery for background tasks
   - Redis for task queue
   - Multiple worker nodes

2. **Database Scaling**:
   - Neo4j clustering
   - ChromaDB sharding
   - Object storage (S3/MinIO)

3. **Load Balancing**:
   - Multiple API servers
   - Nginx reverse proxy
   - Horizontal scaling

4. **Caching Layer**:
   - Redis for search results
   - CDN for static assets
   - Query result caching

## Monitoring & Logging

### Logging Strategy

```python
logger.info("Document uploaded: {doc_id}")
logger.warning("Neo4j unavailable, skipping graph")
logger.error("Processing failed: {error}", exc_info=True)
```

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-critical)
- `ERROR`: Error messages with stack traces

### Future Monitoring

- Prometheus metrics
- Grafana dashboards
- Error tracking (Sentry)
- Performance monitoring (APM)

## Configuration Management

### YAML Configuration

```yaml
storage:
  pdf_directory: ./data/pdfs
  max_file_size_mb: 50

neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password

vector:
  persist_directory: ./data/chroma
  embedding_model: all-MiniLM-L6-v2
```

### Environment Variables

```bash
NEO4J_PASSWORD=secret
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

### Configuration Priority

1. Environment variables (highest)
2. YAML configuration file
3. Default values (lowest)

## Deployment Considerations

### Development

```bash
# Backend
uvicorn src.api.main:app --reload

# Frontend
cd frontend && npm run dev
```

### Production

```bash
# Backend with Gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend build
cd frontend && npm run build
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0"]
```

### Dependencies

- Python 3.10+
- Neo4j 5.x
- Node.js 18+ (frontend)
- 4GB RAM minimum
- 10GB disk space

## Code Quality

### Type Hints

```python
def search(query: str, top_k: int = 10) -> List[SearchResult]:
    ...
```

### Pydantic Models

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
```

### Docstrings

```python
def upload_document(file: UploadFile) -> str:
    """
    Upload and process a PDF document.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        Document ID
        
    Raises:
        HTTPException: If validation fails
    """
```

## Future Enhancements

### Short Term

1. User authentication and authorization
2. Document versioning
3. Batch upload support
4. Export functionality (citations, summaries)

### Medium Term

1. Advanced search filters (date, author, keywords)
2. Document comparison
3. Automatic citation network analysis
4. Research trend detection

### Long Term

1. Multi-language support
2. Collaborative features (annotations, sharing)
3. Integration with reference managers (Zotero, Mendeley)
4. Mobile application
