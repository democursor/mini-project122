# Data Processing Workflow

## Overview

This document describes the complete data processing pipeline from PDF upload to searchable knowledge base. The workflow is orchestrated using LangGraph, which provides a state machine for managing the multi-stage processing pipeline.

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      DOCUMENT UPLOAD                             │
│                                                                  │
│  User uploads PDF → Validation → Storage → Background Processing│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: PDF PARSING                          │
│                                                                  │
│  Input:  PDF file                                                │
│  Process: PyMuPDF extraction                                     │
│  Output:  Structured document data                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Extract text from all pages                           │  │
│  │  • Parse metadata (title, authors, year)                 │  │
│  │  • Identify document sections                            │  │
│  │  • Extract abstract and references                       │  │
│  │  • Count pages and calculate statistics                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: TEXT CHUNKING                          │
│                                                                  │
│  Input:  Parsed document sections                                │
│  Process: Semantic chunking                                      │
│  Output:  Document chunks with metadata                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Split text at semantic boundaries                     │  │
│  │  • Maintain context with section headings                │  │
│  │  • Target chunk size: 512 tokens                         │  │
│  │  • Overlap: 50 tokens between chunks                     │  │
│  │  • Preserve sentence integrity                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 3: CONCEPT EXTRACTION                       │
│                                                                  │
│  Input:  Document chunks                                         │
│  Process: NLP analysis                                           │
│  Output:  Entities and keyphrases                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Named Entity Recognition (spaCy):                       │  │
│  │  • PERSON - Authors, researchers                         │  │
│  │  • ORG - Institutions, organizations                     │  │
│  │  • GPE - Locations, countries                            │  │
│  │  • DATE - Publication dates, time periods                │  │
│  │  • PRODUCT - Software, tools, methods                    │  │
│  │                                                           │  │
│  │  Keyphrase Extraction (KeyBERT):                         │  │
│  │  • Extract important terms and phrases                   │  │
│  │  • Calculate relevance scores                            │  │
│  │  • Identify domain-specific terminology                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 4: KNOWLEDGE GRAPH CONSTRUCTION               │
│                                                                  │
│  Input:  Parsed data + Extracted concepts                        │
│  Process: Graph building                                         │
│  Output:  Neo4j graph nodes and relationships                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Create Nodes:                                            │  │
│  │  • Paper (document_id, title, year, abstract)            │  │
│  │  • Author (name)                                          │  │
│  │  • Concept (name, type)                                   │  │
│  │  • Entity (text, label, chunk_id)                        │  │
│  │                                                           │  │
│  │  Create Relationships:                                    │  │
│  │  • (Paper)-[:AUTHORED_BY]->(Author)                      │  │
│  │  • (Paper)-[:MENTIONS]->(Concept)                        │  │
│  │  • (Paper)-[:CONTAINS]->(Entity)                         │  │
│  │  • (Concept)-[:RELATED_TO]->(Concept)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 5: VECTOR EMBEDDING GENERATION                  │
│                                                                  │
│  Input:  Document chunks                                         │
│  Process: Embedding generation                                   │
│  Output:  384-dimensional vectors                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Model: all-MiniLM-L6-v2                                  │  │
│  │  • Generate embeddings for each chunk                     │  │
│  │  • Batch processing for efficiency                        │  │
│  │  • Normalize vectors                                      │  │
│  │  • Store with metadata                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 6: VECTOR STORAGE (ChromaDB)                  │
│                                                                  │
│  Input:  Chunk embeddings + metadata                             │
│  Process: Database insertion                                     │
│  Output:  Indexed vectors ready for search                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Store in ChromaDB:                                       │  │
│  │  • Embedding vectors                                      │  │
│  │  • Original chunk text                                    │  │
│  │  • Metadata (document_id, chunk_id, section)             │  │
│  │  • Create HNSW index for fast similarity search          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING COMPLETE                           │
│                                                                  │
│  Document is now:                                                │
│  ✓ Searchable via semantic search                                │
│  ✓ Queryable via knowledge graph                                 │
│  ✓ Available for AI-powered Q&A                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Stage Descriptions

### Stage 1: PDF Parsing

**Component**: `src/parsing/parser.py` - `PDFParser`

**Input**: PDF file path

**Process**:
1. Open PDF using PyMuPDF (fitz)
2. Extract text from each page
3. Parse document metadata:
   - Title (from first page or metadata)
   - Authors (pattern matching)
   - Year (date extraction)
   - Abstract (section detection)
4. Identify document structure:
   - Section headings
   - Paragraphs
   - References
5. Calculate statistics:
   - Page count
   - Word count
   - Character count

**Output**: `ParsedDocument` object containing:
```python
{
  "document_id": "doc_abc123",
  "metadata": {
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "year": 2024,
    "abstract": "..."
  },
  "sections": [
    {
      "heading": "Introduction",
      "content": "...",
      "page_start": 1
    }
  ],
  "page_count": 10
}
```

**Error Handling**:
- Corrupted PDF → Validation error
- Encrypted PDF → Decryption attempt
- No text content → OCR fallback (future)

### Stage 2: Text Chunking

**Component**: `src/chunking/chunker.py` - `SemanticChunker`

**Input**: Parsed document sections

**Process**:
1. For each section:
   - Split text into sentences
   - Group sentences into chunks
   - Target size: 512 tokens
   - Overlap: 50 tokens
2. Maintain context:
   - Include section heading
   - Preserve sentence boundaries
   - Add chunk metadata
3. Generate chunk IDs

**Output**: List of `DocumentChunk` objects:
```python
{
  "chunk_id": "doc_abc123_chunk_001",
  "document_id": "doc_abc123",
  "text": "Chunk text content...",
  "section_heading": "Introduction",
  "token_count": 487,
  "position": 0
}
```

**Configuration**:
- `chunk_size`: 512 tokens (default)
- `chunk_overlap`: 50 tokens (default)
- `min_chunk_size`: 100 tokens
- `max_chunk_size`: 1000 tokens

### Stage 3: Concept Extraction

**Component**: `src/extraction/extractor.py` - `ConceptExtractor`

**Input**: Document chunks

**Process**:

**Named Entity Recognition (spaCy)**:
1. Load spaCy model (en_core_web_sm)
2. Process chunk text
3. Extract entities:
   - PERSON: People, authors
   - ORG: Organizations, institutions
   - GPE: Geopolitical entities
   - DATE: Dates and time periods
   - PRODUCT: Products, methods, tools
4. Filter by confidence score

**Keyphrase Extraction (KeyBERT)**:
1. Load KeyBERT model
2. Extract keyphrases using:
   - N-gram range: 1-3 words
   - Diversity: 0.5 (MMR)
   - Top K: 20 phrases
3. Calculate relevance scores
4. Filter by minimum score (0.3)

**Output**: `ExtractionResult` object:
```python
{
  "chunk_id": "doc_abc123_chunk_001",
  "document_id": "doc_abc123",
  "entities": [
    {
      "text": "neural networks",
      "label": "PRODUCT",
      "start": 45,
      "end": 60
    }
  ],
  "keyphrases": [
    {
      "phrase": "deep learning",
      "score": 0.87
    }
  ]
}
```

### Stage 4: Knowledge Graph Construction

**Component**: `src/graph/builder.py` - `KnowledgeGraphBuilder`

**Input**: Parsed data + Extraction results

**Process**:

**Create Paper Node**:
```cypher
CREATE (p:Paper {
  document_id: $doc_id,
  title: $title,
  year: $year,
  abstract: $abstract,
  page_count: $pages
})
```

**Create Author Nodes and Relationships**:
```cypher
MERGE (a:Author {name: $author_name})
CREATE (p)-[:AUTHORED_BY]->(a)
```

**Create Concept Nodes**:
```cypher
MERGE (c:Concept {name: $concept_name, type: $type})
CREATE (p)-[:MENTIONS {count: $count}]->(c)
```

**Create Entity Nodes**:
```cypher
CREATE (e:Entity {
  text: $text,
  label: $label,
  chunk_id: $chunk_id
})
CREATE (p)-[:CONTAINS]->(e)
```

**Output**: Graph structure in Neo4j

**Indexes Created**:
- `CREATE INDEX ON :Paper(document_id)`
- `CREATE INDEX ON :Author(name)`
- `CREATE INDEX ON :Concept(name)`

### Stage 5: Vector Embedding Generation

**Component**: `src/vector/embedder.py` - `EmbeddingGenerator`

**Input**: Document chunks (text)

**Process**:
1. Load Sentence Transformer model
   - Model: `all-MiniLM-L6-v2`
   - Dimensions: 384
   - Max sequence length: 512 tokens
2. Batch processing:
   - Batch size: 32 chunks
   - Normalize embeddings
   - Convert to numpy arrays
3. Generate embeddings for each chunk

**Output**: List of embedding vectors:
```python
[
  [0.123, -0.456, 0.789, ...],  # 384 dimensions
  [0.234, -0.567, 0.890, ...],
  ...
]
```

**Performance**:
- CPU: ~2-3 seconds per batch
- GPU: ~0.5 seconds per batch
- First run: Model download (~90MB)

### Stage 6: Vector Storage

**Component**: `src/vector/store.py` - `VectorStore`

**Input**: Chunk embeddings + metadata

**Process**:
1. Connect to ChromaDB
2. Get or create collection
3. Prepare data:
   - IDs: chunk_id
   - Embeddings: vectors
   - Documents: chunk text
   - Metadata: document_id, section, etc.
4. Batch insert into ChromaDB
5. Create HNSW index automatically

**Output**: Indexed vectors in ChromaDB

**Storage Structure**:
```
data/chroma/
├── chroma.sqlite3          # Metadata database
└── [UUID]/                 # Collection data
    ├── data_level0.bin     # HNSW index
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

## Search and Retrieval Workflows

### Semantic Search Workflow

```
User Query
    │
    ▼
Query Processing
    │
    ├─→ Text normalization
    ├─→ Generate query embedding
    └─→ Prepare filters
    │
    ▼
ChromaDB Search
    │
    ├─→ Cosine similarity
    ├─→ HNSW index lookup
    └─→ Top-K results
    │
    ▼
Result Ranking
    │
    ├─→ Score normalization
    ├─→ Metadata enrichment
    └─→ Deduplication
    │
    ▼
Return Results
```

### RAG (Chat) Workflow

```
User Question
    │
    ▼
Context Retrieval
    │
    ├─→ Semantic search (top 15 chunks)
    ├─→ Rerank by relevance
    └─→ Extract source metadata
    │
    ▼
Prompt Construction
    │
    ├─→ System prompt
    ├─→ Retrieved context
    ├─→ User question
    └─→ Conversation history
    │
    ▼
LLM Generation
    │
    ├─→ Send to Gemini/GPT
    ├─→ Stream response
    └─→ Parse citations
    │
    ▼
Response Formatting
    │
    ├─→ Extract answer
    ├─→ Link citations
    └─→ Format markdown
    │
    ▼
Return Answer + Citations
```

### Knowledge Graph Query Workflow

```
Graph Query Request
    │
    ▼
Query Type Detection
    │
    ├─→ Find related papers
    ├─→ Explore concepts
    ├─→ Author network
    └─→ Citation analysis
    │
    ▼
Cypher Query Construction
    │
    ├─→ Build query
    ├─→ Add filters
    └─→ Set limits
    │
    ▼
Neo4j Execution
    │
    ├─→ Execute query
    ├─→ Fetch results
    └─→ Format data
    │
    ▼
Graph Visualization Data
    │
    ├─→ Nodes
    ├─→ Edges
    └─→ Properties
    │
    ▼
Return Graph Data
```

## Error Handling and Recovery

### Pipeline Error Handling

Each stage can fail independently:

```python
try:
    # Stage execution
    result = stage.process(input_data)
    state["status"] = "success"
except ValidationError as e:
    state["status"] = "error"
    state["error_message"] = f"Validation failed: {e}"
except ProcessingError as e:
    state["status"] = "error"
    state["error_message"] = f"Processing failed: {e}"
```

### Retry Logic

- Connection errors: 3 retries with exponential backoff
- Transient failures: Automatic retry
- Permanent failures: Log and skip

### Partial Success

- If graph building fails, vector storage still proceeds
- Document remains searchable even if graph is incomplete
- Metadata is always saved

## Performance Optimization

### Batch Processing
- Embeddings: 32 chunks per batch
- Graph operations: Bulk inserts
- Database writes: Transaction batching

### Caching
- Model loading: Once per process
- Embeddings: Cached in memory during processing
- Configuration: Loaded once at startup

### Parallel Processing (Future)
- Multiple documents in parallel
- Async I/O for database operations
- GPU acceleration for embeddings

## Monitoring and Logging

### Logged Events
- Pipeline start/end
- Stage transitions
- Error occurrences
- Processing duration
- Resource usage

### Metrics Tracked
- Documents processed
- Average processing time
- Success/failure rates
- Database sizes
- API response times

## Configuration

### Pipeline Configuration

```yaml
processing:
  max_concurrent_documents: 10
  retry_attempts: 3
  
chunking:
  chunk_size: 512
  chunk_overlap: 50
  
extraction:
  domain: general
  use_domain_models: false
  max_entities: 50
  max_keyphrases: 20
  
vector:
  embedding_model: all-MiniLM-L6-v2
  batch_size: 32
```

### Customization Options

- Chunk size and overlap
- Embedding model selection
- Entity types to extract
- Graph relationship types
- Batch sizes
- Retry policies
