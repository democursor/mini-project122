# Design Document: Autonomous Research Literature Intelligence & Discovery Platform

## Overview

### Project Vision

The Autonomous Research Literature Intelligence & Discovery Platform is an end-to-end AI system that transforms how researchers interact with academic literature. Think of it as a "smart librarian" that not only organizes your papers but understands their content, discovers hidden connections, and helps you synthesize insights across your entire research collection.

**The Core Problem We're Solving:**
Imagine you're researching "transformer architectures in computer vision." You have 50 papers downloaded, but:
- You don't know which papers are most relevant to your specific question
- You can't remember which paper discussed a specific technique
- You want to find papers that combine transformers with object detection, but keyword search fails
- You need to understand how different papers relate to each other

This platform solves these problems by understanding the semantic meaning of papers, not just matching keywords.

### Why This Project Matters

**For Your Learning:**
This project teaches you how real AI systems are built in industry. You'll learn:
- How to design multi-component AI systems (not just train a model)
- How to combine different AI techniques (NLP, embeddings, knowledge graphs, GenAI)
- How to handle real-world challenges (error handling, scalability, data persistence)
- How to think like an ML engineer, not just a data scientist

**For Your Career:**
Companies building AI products need engineers who can:
- Design end-to-end systems, not just train models
- Understand trade-offs between different technologies
- Build production-grade systems that handle failures gracefully
- Combine multiple AI techniques to solve complex problems

This project demonstrates all of these skills.

### High-Level System Flow

```
User uploads PDF → Parse & Extract Text → Chunk Intelligently → 
Extract Concepts → Build Knowledge Graph → Generate Embeddings → 
Store in Vector DB → Enable Semantic Search → Power AI Assistant
```

Each arrow represents a transformation of data, and each step adds value.


## Technology Stack

### Programming Language: Python

**Why Python?**
- **Rich AI/ML Ecosystem**: PyTorch, Transformers, LangChain, LangGraph all have excellent Python support
- **Rapid Prototyping**: Quick iteration for learning and experimentation
- **Industry Standard**: Most AI/ML roles expect Python proficiency
- **Community**: Massive community means abundant resources and libraries

**Trade-offs:**
- ✅ Fast development, great libraries, easy to learn
- ❌ Slower than compiled languages (but not a concern for our scale)
- ❌ GIL limits true parallelism (mitigated by async I/O and multiprocessing)

### Core Libraries and Frameworks

#### 1. **PDF Processing: PyMuPDF (fitz)**

**What it does:** Extracts text, metadata, and structure from PDF files

**Why this choice:**
- Fast and memory-efficient (written in C)
- Handles complex PDFs (multi-column, images, tables)
- Extracts text with position information (useful for structure preservation)

**Alternatives considered:**
- PDFMiner: More detailed but slower
- PyPDF2: Simpler but less robust with complex PDFs

**Learning point:** In production, you often need libraries that balance features with performance.

#### 2. **NLP and Deep Learning: Hugging Face Transformers + PyTorch**

**What it does:** Provides pre-trained models for embeddings, NER, and text understanding

**Why this choice:**
- **Transformers library**: Access to thousands of pre-trained models
- **PyTorch**: Industry-standard deep learning framework, great for research
- **Pre-trained models**: No need to train from scratch (transfer learning)

**Key models we'll use:**
- **Sentence-BERT (all-MiniLM-L6-v2)**: Fast, efficient embeddings (384 dimensions)
- **SciBERT**: BERT trained on scientific papers (better for academic text)
- **SpaCy with en_core_sci_md**: Scientific named entity recognition

**Trade-offs:**
- ✅ State-of-the-art performance, easy to use, well-documented
- ❌ Models require GPU for fast inference (but CPU works for learning)
- ❌ Large model files (but we use smaller variants)

**Learning point:** You don't need to train models from scratch. Transfer learning is how real products are built.


#### 3. **Orchestration: LangGraph**

**What it does:** Manages the multi-step workflow of processing documents

**Why this choice:**
- **State Management**: Tracks document progress through pipeline stages
- **Error Handling**: Built-in retry logic and failure recovery
- **Visualization**: Can visualize the processing workflow as a graph
- **Async Support**: Handles concurrent document processing

**How it works (simple analogy):**
Think of LangGraph as a factory assembly line. Each station (node) does one job:
- Station 1: Parse PDF
- Station 2: Chunk text
- Station 3: Extract concepts
- Station 4: Build graph

If a station fails, LangGraph can retry or route to error handling.

**Alternatives considered:**
- Apache Airflow: Too heavy for local deployment
- Celery: Requires message broker setup (Redis/RabbitMQ)
- Custom orchestration: Reinventing the wheel

**Learning point:** Use existing orchestration tools rather than building your own. This is what production systems do.

#### 4. **Knowledge Graph: Neo4j**

**What it does:** Stores papers, concepts, and relationships in a graph database

**Why this choice:**
- **Native Graph Storage**: Optimized for relationship queries
- **Cypher Query Language**: Intuitive graph queries (like SQL for graphs)
- **Visualization**: Built-in browser for exploring the graph
- **ACID Transactions**: Data consistency guarantees

**Example query:**
```cypher
// Find papers related to "attention mechanisms" through 2 hops
MATCH (p:Paper)-[:MENTIONS]->(c:Concept {name: "attention mechanisms"})-[:RELATED_TO*1..2]-(related:Concept)<-[:MENTIONS]-(related_paper:Paper)
RETURN related_paper.title, COUNT(related) as relevance
ORDER BY relevance DESC
```

**Alternatives considered:**
- NetworkX: In-memory graph (doesn't persist, limited scale)
- PostgreSQL with recursive CTEs: Can do graphs but not optimized
- ArangoDB: Multi-model but less mature ecosystem

**Trade-offs:**
- ✅ Excellent for relationship queries, great tooling, industry-proven
- ❌ Requires separate database server (but free community edition)
- ❌ Learning curve for Cypher (but simpler than you think)

**Learning point:** Choose specialized databases for specialized needs. Graph databases excel at relationship queries.


#### 5. **Vector Store: ChromaDB**

**What it does:** Stores embeddings and enables fast similarity search

**Why this choice:**
- **Embedded Database**: Runs in-process (no separate server needed)
- **Simple API**: Easy to use, minimal setup
- **Persistent Storage**: Data survives restarts
- **Built-in HNSW**: Fast approximate nearest neighbor search

**How vector search works (simple analogy):**
Imagine each paper is a point in a 384-dimensional space. Papers about similar topics are close together. When you search, we find the nearest points to your query.

**Alternatives considered:**
- Pinecone: Cloud-only, requires API key
- Weaviate: More features but heavier setup
- FAISS: Fast but no persistence layer (need to build yourself)
- Milvus: Production-grade but overkill for learning

**Trade-offs:**
- ✅ Zero setup, perfect for learning, good performance
- ❌ Not as fast as FAISS for huge scale (but fine for 10k papers)
- ❌ Limited distributed capabilities (but we're single-machine)

**Learning point:** Start simple. ChromaDB is perfect for learning and can scale to thousands of documents.

#### 6. **GenAI: OpenAI API or Local LLM**

**What it does:** Powers the conversational research assistant

**Why this choice:**
- **OpenAI GPT-4**: Best quality, easy API, but costs money
- **Local alternative**: Llama 3 or Mistral via Ollama (free, private, slower)

**How it works:**
1. User asks: "What papers discuss attention mechanisms in vision?"
2. We retrieve relevant chunks using vector search
3. We pass chunks + question to LLM
4. LLM generates answer citing specific papers

This is called **Retrieval-Augmented Generation (RAG)**.

**Trade-offs:**
- OpenAI: ✅ Best quality, ❌ Costs money, ❌ Sends data to API
- Local LLM: ✅ Free, ✅ Private, ❌ Slower, ❌ Lower quality

**Learning point:** RAG is how production AI assistants work. You ground LLM responses in your own data.

#### 7. **Data Persistence: SQLite + File Storage**

**What it does:** Stores metadata, processing state, and PDF files

**Why this choice:**
- **SQLite**: Embedded database, zero configuration, ACID guarantees
- **File Storage**: Simple directory structure for PDFs

**Schema design:**
```
documents: id, filename, upload_date, status, error_message
chunks: id, document_id, text, position, embedding_id
concepts: id, name, type, confidence
document_concepts: document_id, concept_id, frequency
```

**Learning point:** Use relational databases for structured data with relationships. Use file storage for binary blobs (PDFs).


### Technology Stack Summary

| Component | Technology | Why | Alternative |
|-----------|-----------|-----|-------------|
| Language | Python 3.10+ | AI/ML ecosystem, rapid development | - |
| PDF Parsing | PyMuPDF | Fast, robust, handles complex PDFs | PDFMiner, PyPDF2 |
| NLP/Embeddings | Sentence-BERT, SciBERT | Pre-trained, efficient, scientific domain | OpenAI embeddings (paid) |
| NER | SpaCy (sci-md) | Scientific entity recognition | Hugging Face NER models |
| Orchestration | LangGraph | State management, error handling | Airflow, Celery, custom |
| Knowledge Graph | Neo4j | Native graph, Cypher, visualization | NetworkX, PostgreSQL |
| Vector Store | ChromaDB | Embedded, simple, persistent | Pinecone, FAISS, Weaviate |
| GenAI | OpenAI API / Ollama | RAG capabilities | Anthropic, local Llama |
| Metadata DB | SQLite | Embedded, zero-config, ACID | PostgreSQL (overkill) |
| File Storage | Local filesystem | Simple, direct access | S3 (for cloud) |

### Development Environment

**Recommended Setup:**
- Python 3.10 or higher
- Virtual environment (venv or conda)
- GPU optional but helpful (CUDA for PyTorch)
- 8GB RAM minimum, 16GB recommended
- 50GB disk space for models and data

**Key Dependencies:**
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
spacy>=3.5.0
pymupdf>=1.22.0
chromadb>=0.4.0
neo4j>=5.0.0
langgraph>=0.1.0
langchain>=0.1.0
openai>=1.0.0  # optional
```


## Architecture

### System Components and Responsibilities

The platform follows a **modular, pipeline-based architecture** where each component has a single, well-defined responsibility. This is called **separation of concerns** - a fundamental principle in software engineering.

**Why modularity matters:**
- **Testability**: Test each component independently
- **Maintainability**: Change one component without breaking others
- **Extensibility**: Add new features by adding new components
- **Debugging**: Isolate problems to specific components

Think of it like building with LEGO blocks - each block has a specific shape and purpose, and you can rearrange them to build different things.

### Component Diagram

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
    ▼         ▼         ▼         ▼         ▼         ▼
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

### Component Details

#### 1. PDF Ingestion Module

**Responsibility:** Accept, validate, and store uploaded PDF files

**Inputs:** PDF file from user's local machine
**Outputs:** Document ID, stored PDF file, metadata record

**Key Operations:**
- Validate file format and size
- Generate unique document ID
- Store PDF in file system
- Create metadata record in SQLite
- Trigger processing workflow

**Error Handling:**
- Invalid format → Reject with clear message
- File too large → Reject with size limit
- Storage failure → Retry with exponential backoff

**Why this is separate:** Ingestion concerns (validation, storage) are different from processing concerns (parsing, analysis). Separating them makes each easier to test and modify.


#### 2. Parser Module

**Responsibility:** Extract text, metadata, and structure from PDFs

**Inputs:** PDF file path, document ID
**Outputs:** Structured JSON with text, metadata, sections

**Key Operations:**
- Extract text using PyMuPDF
- Identify document structure (title, abstract, sections, references)
- Extract metadata (authors, date, venue)
- Handle multi-column layouts
- Preserve text positioning for structure

**Deep Learning Application:** None directly, but output feeds NLP models

**Error Handling:**
- Corrupted PDF → Log error, mark document as failed
- Extraction failure → Retry with different extraction method
- Missing metadata → Continue with partial data

**Output Format:**
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
    {
      "heading": "Abstract",
      "text": "...",
      "position": 0
    },
    {
      "heading": "Introduction",
      "text": "...",
      "position": 1
    }
  ]
}
```

**Why this is separate:** Parsing is a distinct concern from semantic understanding. You might want to swap PDF parsers without changing downstream processing.

#### 3. Chunking Service

**Responsibility:** Segment documents into semantically coherent chunks

**Inputs:** Parsed document JSON
**Outputs:** List of chunks with metadata

**Key Operations:**
- Use sentence embeddings to measure semantic similarity
- Identify natural boundaries (section breaks, topic shifts)
- Create chunks of 100-500 tokens
- Maintain references to source document and position

**Deep Learning Application:** 
- **Model:** Sentence-BERT (all-MiniLM-L6-v2)
- **Purpose:** Compute sentence embeddings to detect semantic boundaries
- **How:** Compare consecutive sentence embeddings; large distance = boundary

**Algorithm (Simplified):**
```
1. Split document into sentences
2. Compute embedding for each sentence
3. Calculate cosine similarity between consecutive sentences
4. When similarity drops below threshold, create chunk boundary
5. Respect section boundaries as hard boundaries
6. Ensure chunks are 100-500 tokens
```

**Why semantic chunking matters:**
- Fixed-size chunks (e.g., every 500 tokens) can split concepts mid-thought
- Semantic chunks keep related ideas together
- Better retrieval: When you search, you get complete thoughts, not fragments

**Learning point:** This is where deep learning adds real value - understanding semantic boundaries, not just counting characters.


#### 4. Concept Extraction Module

**Responsibility:** Identify key concepts, entities, and relationships in text

**Inputs:** Document chunks
**Outputs:** Concepts, entities, relationships with confidence scores

**Key Operations:**
- Named Entity Recognition (NER) for researchers, methods, datasets
- Keyphrase extraction for important concepts
- Relationship extraction between entities
- Concept normalization (e.g., "BERT" and "Bidirectional Encoder Representations from Transformers" → same concept)

**Deep Learning Application:**
- **Model 1:** SpaCy en_core_sci_md (scientific NER)
  - Trained on scientific papers
  - Recognizes: PERSON, ORG, METHOD, DATASET, METRIC
  
- **Model 2:** KeyBERT (keyphrase extraction)
  - Uses BERT embeddings to find important phrases
  - Extracts phrases most representative of document

**Example Output:**
```json
{
  "chunk_id": "uuid",
  "entities": [
    {"text": "transformer", "type": "METHOD", "confidence": 0.95},
    {"text": "ImageNet", "type": "DATASET", "confidence": 0.98},
    {"text": "Vaswani", "type": "PERSON", "confidence": 0.92}
  ],
  "keyphrases": [
    {"phrase": "self-attention mechanism", "score": 0.87},
    {"phrase": "multi-head attention", "score": 0.82}
  ],
  "relationships": [
    {"subject": "transformer", "predicate": "uses", "object": "self-attention"}
  ]
}
```

**Why this matters:**
- Enables concept-based search ("find papers about transformers")
- Powers knowledge graph (concepts are nodes, relationships are edges)
- Helps identify paper themes without reading full text

**Learning point:** Pre-trained models for scientific text exist! You don't need to train from scratch. This is transfer learning in action.

#### 5. Knowledge Graph Builder

**Responsibility:** Construct and maintain the research knowledge graph

**Inputs:** Documents, concepts, entities, relationships
**Outputs:** Graph database with nodes and edges

**Graph Schema:**
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
- (Concept)-[:CO_OCCURS_WITH]->(Concept)
```

**Key Operations:**
- Create paper nodes with metadata
- Create concept nodes (or link to existing)
- Create relationships with weights (e.g., mention frequency)
- Compute concept co-occurrence for RELATED_TO edges
- Update graph incrementally as new papers are added

**Example Queries:**
```cypher
// Find papers similar to a given paper (through shared concepts)
MATCH (p1:Paper {id: $paper_id})-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Paper)
RETURN p2.title, COUNT(c) as shared_concepts
ORDER BY shared_concepts DESC
LIMIT 10

// Find most influential concepts (mentioned in many papers)
MATCH (c:Concept)<-[:MENTIONS]-(p:Paper)
RETURN c.name, COUNT(p) as paper_count
ORDER BY paper_count DESC
LIMIT 20

// Find research trends over time
MATCH (p:Paper)-[:MENTIONS]->(c:Concept {name: $concept})
RETURN p.year, COUNT(p) as mentions
ORDER BY p.year
```

**Why a graph database:**
- Relationship queries are natural and fast
- Can traverse connections (e.g., "papers related to papers related to X")
- Visualize research landscape
- Discover non-obvious connections

**Learning point:** Different data structures for different queries. Graphs excel at relationship queries that would be complex in SQL.


#### 6. Vector Store Module

**Responsibility:** Store embeddings and enable semantic similarity search

**Inputs:** Document chunks, embeddings
**Outputs:** Similar chunks for a given query

**Key Operations:**
- Generate embeddings for each chunk using Sentence-BERT
- Store embeddings in ChromaDB with metadata
- Build HNSW index for fast approximate nearest neighbor search
- Query for top-k similar chunks given a query embedding

**Deep Learning Application:**
- **Model:** Sentence-BERT (all-MiniLM-L6-v2 or SciBERT)
- **Purpose:** Convert text to dense vector representations
- **Dimensions:** 384 (MiniLM) or 768 (SciBERT)

**How Embeddings Work (Simple Explanation):**
Imagine each piece of text is a point in a high-dimensional space. Texts with similar meanings are close together. When you search, we find the nearest points to your query.

**Example:**
```
Query: "attention mechanisms in vision"
Embedding: [0.23, -0.45, 0.67, ..., 0.12]  # 384 numbers

Similar chunks:
1. "Vision transformers apply self-attention to image patches..." (similarity: 0.89)
2. "Attention-based models for object detection..." (similarity: 0.85)
3. "Multi-head attention in convolutional networks..." (similarity: 0.82)
```

**Why vector search matters:**
- Finds semantically similar content, not just keyword matches
- Query: "neural networks for images" matches "CNNs for computer vision"
- Handles synonyms, paraphrases, and conceptual similarity

**HNSW Index:**
- Hierarchical Navigable Small World graph
- Approximate nearest neighbor search in sub-linear time
- Trade-off: 95%+ accuracy, 100x faster than exact search

**Learning point:** Vector search is the foundation of modern semantic search. This is how Google, ChatGPT, and recommendation systems work.

#### 7. Semantic Search Engine

**Responsibility:** Provide natural language search over the document collection

**Inputs:** User query (natural language)
**Outputs:** Ranked list of relevant documents and chunks

**Key Operations:**
- Convert query to embedding
- Retrieve top-k similar chunks from vector store
- Aggregate chunks by document
- Rank documents by relevance
- Apply metadata filters (date, author, venue)
- Return results with highlighted excerpts

**Search Pipeline:**
```
1. User query: "How do transformers work in computer vision?"
2. Generate query embedding using same model as chunks
3. Vector search: Find 50 most similar chunks
4. Group chunks by document
5. Score documents by: sum of chunk similarities + metadata relevance
6. Return top 10 documents with relevant excerpts
```

**Advanced Features:**
- **Hybrid search:** Combine vector search with keyword search (BM25)
- **Re-ranking:** Use cross-encoder model to re-rank top results
- **Filters:** "papers after 2020", "papers by Hinton", "papers in NeurIPS"

**Why this is separate from vector store:**
- Search engine adds business logic (ranking, filtering, aggregation)
- Vector store is just storage and retrieval
- Separation allows swapping vector stores without changing search logic

**Learning point:** Production systems have layers. Raw storage (vector store) is separate from business logic (search engine).


#### 8. AI Research Assistant (RAG System)

**Responsibility:** Answer questions about the research collection using GenAI

**Inputs:** User question, conversation history
**Outputs:** AI-generated answer with citations

**Key Operations:**
- Retrieve relevant chunks using semantic search
- Construct prompt with question + retrieved context
- Call LLM to generate answer
- Extract citations from answer
- Maintain conversation context for follow-ups

**RAG Pipeline (Retrieval-Augmented Generation):**
```
1. User: "What are the main advantages of transformers over RNNs?"

2. Retrieval:
   - Convert question to embedding
   - Find 5 most relevant chunks
   - Extract: chunk text, paper title, authors

3. Prompt Construction:
   """
   You are a research assistant. Answer the question based on the provided context.
   Cite papers using [Paper Title, Authors].
   
   Context:
   [1] "Transformers eliminate recurrence, enabling parallel processing..." 
       (Attention Is All You Need, Vaswani et al.)
   [2] "Self-attention captures long-range dependencies better than RNNs..."
       (BERT, Devlin et al.)
   ...
   
   Question: What are the main advantages of transformers over RNNs?
   
   Answer:
   """

4. LLM Generation:
   "Transformers offer several advantages over RNNs:
   1. Parallel processing: Unlike RNNs which process sequentially, transformers 
      process all tokens simultaneously [Attention Is All You Need, Vaswani et al.]
   2. Long-range dependencies: Self-attention captures relationships between 
      distant tokens more effectively [BERT, Devlin et al.]
   ..."

5. Return answer with citations
```

**Why RAG instead of just LLM:**
- **Grounding:** Answers based on your documents, not general knowledge
- **Citations:** Know which papers support each claim
- **Up-to-date:** Works with your latest papers, not just training data
- **Accuracy:** Reduces hallucinations by constraining to retrieved context

**Conversation Context:**
- Store last 5 Q&A pairs
- Include in prompt for follow-up questions
- Example: "What about their disadvantages?" → knows "their" refers to transformers

**Learning point:** RAG is how production AI assistants work (ChatGPT with web search, GitHub Copilot, etc.). It combines retrieval with generation.

#### 9. LangGraph Orchestrator

**Responsibility:** Manage the multi-step document processing workflow

**Why we need orchestration:**
Processing a document involves many steps, each of which can fail. We need:
- **State tracking:** Where is each document in the pipeline?
- **Error handling:** What happens if parsing fails?
- **Retry logic:** Should we retry failed operations?
- **Concurrency:** Can we process multiple documents simultaneously?
- **Observability:** How do we monitor progress?

**LangGraph Workflow:**
```
                    ┌─────────────┐
                    │   Upload    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Parse    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Chunk    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Extract   │
                    │   Concepts  │
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │Build Graph  │      │  Generate   │
         │             │      │ Embeddings  │
         └──────┬──────┘      └──────┬──────┘
                │                     │
                └──────────┬──────────┘
                           │
                    ┌──────▼──────┐
                    │   Complete  │
                    └─────────────┘
```

**State Management:**
```python
class DocumentState:
    document_id: str
    status: str  # "uploaded", "parsing", "chunking", "extracting", "complete", "failed"
    current_step: str
    retry_count: int
    error_message: Optional[str]
    parsed_data: Optional[dict]
    chunks: Optional[List[dict]]
    concepts: Optional[List[dict]]
```

**Error Handling:**
- Each step can succeed, fail, or retry
- Failed steps trigger retry with exponential backoff
- After 3 retries, mark document as failed and alert user
- Partial failures: If concept extraction fails, still store chunks

**Concurrency:**
- Process up to 10 documents simultaneously
- Each document has independent state
- Shared resources (databases) handle concurrent access

**Why LangGraph:**
- Built for LLM workflows (but works for any multi-step process)
- State management out of the box
- Visualization of workflow
- Async/await support for concurrency

**Learning point:** Don't build orchestration from scratch. Use existing tools. This is what production systems do.


### How Components Interact

**Example: Processing a New Paper**

1. **User uploads "attention_is_all_you_need.pdf"**
   - PDF Ingestion validates file, stores it, creates metadata record
   - Returns document_id: "doc_123"
   - Triggers LangGraph workflow

2. **LangGraph starts workflow for doc_123**
   - State: {document_id: "doc_123", status: "parsing", retry_count: 0}

3. **Parser extracts text**
   - Reads PDF from file storage
   - Extracts: title, authors, sections, text
   - Stores parsed JSON in state
   - Updates SQLite: status = "parsed"

4. **Chunker segments document**
   - Reads parsed JSON from state
   - Computes sentence embeddings
   - Creates 45 semantic chunks
   - Stores chunks in SQLite
   - Updates state with chunk IDs

5. **Concept Extractor analyzes chunks**
   - For each chunk: run NER, extract keyphrases
   - Finds: "transformer", "self-attention", "multi-head attention", etc.
   - Stores concepts in SQLite
   - Updates state with concept IDs

6. **Graph Builder updates knowledge graph**
   - Creates Paper node in Neo4j
   - Creates Concept nodes (or links to existing)
   - Creates MENTIONS relationships
   - Computes co-occurrence for RELATED_TO edges

7. **Vector Store generates embeddings**
   - For each chunk: generate embedding using Sentence-BERT
   - Store in ChromaDB with metadata: {chunk_id, document_id, text}
   - Build/update HNSW index

8. **Workflow completes**
   - State: {status: "complete"}
   - Update SQLite: status = "complete", completed_at = timestamp
   - Notify user: "Paper processed successfully"

**If any step fails:**
- LangGraph catches exception
- Increments retry_count
- Waits (exponential backoff: 1s, 2s, 4s)
- Retries step
- If retry_count > 3: mark as failed, notify user

**This demonstrates:**
- **Separation of concerns:** Each component does one thing
- **State management:** LangGraph tracks progress
- **Error resilience:** Automatic retries
- **Data flow:** Output of one step is input to next
- **Persistence:** Data stored at each step (can resume if system crashes)

### Scaling to Production

**Current design (single-user, local):**
- SQLite for metadata
- Local file storage
- Single-machine processing
- No authentication

**Production evolution:**
- **Database:** SQLite → PostgreSQL (multi-user, concurrent access)
- **File storage:** Local → S3 (scalable, durable)
- **Processing:** Single-machine → Distributed (Celery workers, Kubernetes)
- **Vector store:** ChromaDB → Pinecone/Weaviate (managed, scalable)
- **Authentication:** Add user accounts, API keys
- **Monitoring:** Add logging, metrics, alerting (Prometheus, Grafana)
- **API:** Add REST API for programmatic access

**The architecture supports this evolution because:**
- Components are modular (swap implementations)
- Interfaces are well-defined (change internals without breaking others)
- State is externalized (can distribute processing)

**Learning point:** Start simple, design for evolution. Good architecture makes scaling easier later.


## Data Models

### Document Metadata (SQLite)

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,  -- 'uploaded', 'processing', 'complete', 'failed'
    error_message TEXT,
    
    -- Extracted metadata
    title TEXT,
    authors TEXT,  -- JSON array
    year INTEGER,
    venue TEXT,
    abstract TEXT,
    
    -- Processing metadata
    chunk_count INTEGER,
    concept_count INTEGER,
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    position INTEGER NOT NULL,  -- Order within document
    section_heading TEXT,
    token_count INTEGER,
    embedding_id TEXT,  -- Reference to ChromaDB
    
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE concepts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT,  -- 'METHOD', 'DATASET', 'METRIC', 'PERSON', 'ORG'
    normalized_name TEXT,  -- For deduplication
    total_mentions INTEGER DEFAULT 0
);

CREATE TABLE document_concepts (
    document_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    confidence REAL,
    
    PRIMARY KEY (document_id, concept_id),
    FOREIGN KEY (document_id) REFERENCES documents(id),
    FOREIGN KEY (concept_id) REFERENCES concepts(id)
);

CREATE TABLE chunk_concepts (
    chunk_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    position INTEGER,  -- Position in chunk
    
    PRIMARY KEY (chunk_id, concept_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (concept_id) REFERENCES concepts(id)
);
```

**Why this schema:**
- **documents:** Core metadata and processing state
- **chunks:** Granular text segments for retrieval
- **concepts:** Deduplicated concept catalog
- **document_concepts:** Many-to-many with frequency
- **chunk_concepts:** Precise concept locations

### Knowledge Graph Schema (Neo4j)

```cypher
// Node types
(:Paper {
    id: string,
    title: string,
    authors: [string],
    year: integer,
    venue: string,
    abstract: string,
    citation_count: integer
})

(:Concept {
    id: string,
    name: string,
    type: string,
    normalized_name: string,
    mention_count: integer
})

(:Author {
    id: string,
    name: string,
    paper_count: integer
})

(:Venue {
    id: string,
    name: string,
    type: string  -- 'conference', 'journal', 'workshop'
})

// Relationship types
(:Paper)-[:MENTIONS {frequency: integer, confidence: float}]->(:Concept)
(:Paper)-[:CITES]->(:Paper)
(:Paper)-[:AUTHORED_BY]->(:Author)
(:Paper)-[:PUBLISHED_IN {year: integer}]->(:Venue)
(:Concept)-[:RELATED_TO {strength: float}]->(:Concept)
(:Concept)-[:CO_OCCURS_WITH {frequency: integer}]->(:Concept)
(:Author)-[:COLLABORATES_WITH {paper_count: integer}]->(:Author)
```

**Why a graph:**
- Natural representation of relationships
- Efficient traversal queries
- Discover indirect connections
- Visualize research landscape

### Vector Store Schema (ChromaDB)

```python
# Collection: "paper_chunks"
{
    "ids": ["chunk_1", "chunk_2", ...],
    "embeddings": [[0.23, -0.45, ...], [0.12, 0.67, ...], ...],
    "metadatas": [
        {
            "document_id": "doc_123",
            "chunk_position": 0,
            "section": "Introduction",
            "token_count": 256,
            "document_title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "year": 2017
        },
        ...
    ],
    "documents": ["text of chunk 1", "text of chunk 2", ...]
}
```

**Why this structure:**
- **embeddings:** Dense vectors for similarity search
- **metadatas:** Filter results by document properties
- **documents:** Original text for display

### Data Flow Example

**User uploads paper → Data created in each store:**

1. **File System:**
   ```
   data/pdfs/doc_123_attention_is_all_you_need.pdf
   ```

2. **SQLite:**
   ```sql
   INSERT INTO documents VALUES (
       'doc_123', 
       'attention_is_all_you_need.pdf',
       'data/pdfs/doc_123_attention_is_all_you_need.pdf',
       '2024-01-15 10:30:00',
       'complete',
       NULL,
       'Attention Is All You Need',
       '["Vaswani", "Shazeer", "Parmar", ...]',
       2017,
       'NeurIPS',
       'The dominant sequence transduction models...',
       45,
       12,
       '2024-01-15 10:30:05',
       '2024-01-15 10:32:30'
   );
   ```

3. **Neo4j:**
   ```cypher
   CREATE (p:Paper {
       id: 'doc_123',
       title: 'Attention Is All You Need',
       authors: ['Vaswani', 'Shazeer', 'Parmar'],
       year: 2017,
       venue: 'NeurIPS'
   })
   CREATE (c1:Concept {name: 'transformer', type: 'METHOD'})
   CREATE (c2:Concept {name: 'self-attention', type: 'METHOD'})
   CREATE (p)-[:MENTIONS {frequency: 47}]->(c1)
   CREATE (p)-[:MENTIONS {frequency: 23}]->(c2)
   ```

4. **ChromaDB:**
   ```python
   collection.add(
       ids=['chunk_1', 'chunk_2', ...],
       embeddings=[embedding_1, embedding_2, ...],
       metadatas=[
           {'document_id': 'doc_123', 'section': 'Introduction', ...},
           ...
       ],
       documents=['The dominant sequence...', ...]
   )
   ```

**Why multiple stores:**
- Each optimized for different queries
- SQLite: Metadata queries, joins
- Neo4j: Relationship traversal
- ChromaDB: Similarity search
- File system: Binary storage

**Learning point:** Polyglot persistence - use the right database for each job.


## Workflow and Data Flow

### End-to-End Processing Workflow

This section describes how data flows through the system from upload to query.

#### Phase 1: Document Ingestion

```
User Action: Upload PDF
     │
     ▼
┌─────────────────────────────────────┐
│  1. Validate File                   │
│     - Check format (PDF)            │
│     - Check size (< 50MB)           │
│     - Generate document ID          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Store File                      │
│     - Save to data/pdfs/            │
│     - Create metadata record        │
│     - Status: 'uploaded'            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Trigger Processing              │
│     - Start LangGraph workflow      │
│     - Return document ID to user    │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: Binary PDF file
- Output: Stored file + metadata record + workflow trigger

#### Phase 2: Text Extraction and Parsing

```
┌─────────────────────────────────────┐
│  1. Load PDF                        │
│     - Read from file storage        │
│     - Open with PyMuPDF             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Extract Text                    │
│     - Iterate through pages         │
│     - Extract text blocks           │
│     - Preserve positioning          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Identify Structure              │
│     - Detect title (large font)     │
│     - Detect sections (headings)    │
│     - Detect abstract (position)    │
│     - Detect references (patterns)  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Extract Metadata                │
│     - Parse authors (regex)         │
│     - Parse date (regex)            │
│     - Parse venue (patterns)        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Output Structured JSON          │
│     - Sections with text            │
│     - Metadata fields               │
│     - Update database               │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: PDF file (binary)
- Output: Structured JSON (text + metadata + structure)

**Error scenarios:**
- Corrupted PDF → Log error, mark failed, notify user
- Missing metadata → Continue with partial data
- Extraction timeout → Retry with simpler method

#### Phase 3: Semantic Chunking

```
┌─────────────────────────────────────┐
│  1. Load Parsed Document            │
│     - Read JSON from previous step  │
│     - Extract text by section       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Sentence Segmentation           │
│     - Split into sentences          │
│     - Use spaCy sentence boundary   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Compute Sentence Embeddings     │
│     - Load Sentence-BERT model      │
│     - Encode each sentence          │
│     - Result: 384-dim vectors       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Detect Semantic Boundaries      │
│     - Compute cosine similarity     │
│     - Between consecutive sentences │
│     - Threshold: similarity < 0.7   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Create Chunks                   │
│     - Group sentences into chunks   │
│     - Respect section boundaries    │
│     - Ensure 100-500 tokens         │
│     - Store with position metadata  │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: Structured JSON (sections + text)
- Output: List of chunks with metadata

**Why this matters:**
- Semantic chunks preserve meaning
- Better retrieval quality
- Chunks are self-contained thoughts

**Algorithm details:**
```python
def semantic_chunking(sentences, embeddings, threshold=0.7):
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        
        if similarity < threshold or len(current_chunk) > 10:
            # Boundary detected or chunk too large
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append(' '.join(current_chunk))
    return chunks
```


#### Phase 4: Concept Extraction

```
┌─────────────────────────────────────┐
│  1. Load Chunks                     │
│     - Read from database            │
│     - Process in batches            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Named Entity Recognition        │
│     - Load SpaCy sci-md model       │
│     - For each chunk:               │
│       * Identify entities           │
│       * Extract: PERSON, ORG,       │
│         METHOD, DATASET, METRIC     │
│     - Assign confidence scores      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Keyphrase Extraction            │
│     - Use KeyBERT                   │
│     - Extract top 5 keyphrases      │
│     - Score by relevance            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Concept Normalization           │
│     - Lowercase                     │
│     - Remove punctuation            │
│     - Map synonyms:                 │
│       "BERT" → "bert"               │
│       "Bidirectional Encoder..." →  │
│       "bert"                        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Relationship Extraction         │
│     - Identify co-occurring concepts│
│     - Within same chunk = related   │
│     - Store with frequency          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. Store Concepts                  │
│     - Insert into concepts table    │
│     - Link to documents and chunks  │
│     - Update mention counts         │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: Text chunks
- Output: Concepts, entities, relationships

**Example:**
```
Input chunk: "The transformer architecture uses self-attention mechanisms 
              to process sequences in parallel, unlike RNNs."

Extracted:
- Entities: 
  * "transformer" (METHOD, confidence: 0.95)
  * "self-attention" (METHOD, confidence: 0.92)
  * "RNN" (METHOD, confidence: 0.88)
  
- Keyphrases:
  * "transformer architecture" (score: 0.87)
  * "self-attention mechanisms" (score: 0.85)
  * "parallel processing" (score: 0.78)
  
- Relationships:
  * (transformer, uses, self-attention)
  * (transformer, differs_from, RNN)
```

#### Phase 5: Knowledge Graph Construction

```
┌─────────────────────────────────────┐
│  1. Create Paper Node               │
│     - Node type: Paper              │
│     - Properties: title, authors,   │
│       year, venue, abstract         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Create/Link Concept Nodes       │
│     - For each concept:             │
│       * Check if exists (by name)   │
│       * Create if new               │
│       * Link to existing if present │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Create MENTIONS Relationships   │
│     - (Paper)-[:MENTIONS]->(Concept)│
│     - Properties: frequency,        │
│       confidence                    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Create Author/Venue Nodes       │
│     - Extract from metadata         │
│     - Create AUTHORED_BY edges      │
│     - Create PUBLISHED_IN edges     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Compute Concept Relationships   │
│     - Find co-occurring concepts    │
│     - Create RELATED_TO edges       │
│     - Weight by co-occurrence freq  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. Update Graph Statistics         │
│     - Increment mention counts      │
│     - Update paper counts           │
│     - Compute centrality metrics    │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: Documents, concepts, relationships
- Output: Graph database with nodes and edges

**Cypher example:**
```cypher
// Create paper node
CREATE (p:Paper {
    id: 'doc_123',
    title: 'Attention Is All You Need',
    authors: ['Vaswani', 'Shazeer'],
    year: 2017
})

// Link to concepts
MATCH (p:Paper {id: 'doc_123'})
MERGE (c:Concept {name: 'transformer'})
CREATE (p)-[:MENTIONS {frequency: 47, confidence: 0.95}]->(c)

// Create concept relationships
MATCH (c1:Concept {name: 'transformer'})
MATCH (c2:Concept {name: 'self-attention'})
MERGE (c1)-[:RELATED_TO {strength: 0.85}]->(c2)
```


#### Phase 6: Vector Embedding and Storage

```
┌─────────────────────────────────────┐
│  1. Load Embedding Model            │
│     - Sentence-BERT (MiniLM)        │
│     - Or SciBERT for scientific     │
│     - Load to GPU if available      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Generate Embeddings             │
│     - For each chunk:               │
│       * Encode text to vector       │
│       * Normalize vector            │
│       * Result: 384-dim array       │
│     - Batch process for efficiency  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Store in ChromaDB               │
│     - Collection: "paper_chunks"    │
│     - Store: id, embedding,         │
│       metadata, text                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Build HNSW Index                │
│     - Hierarchical graph structure  │
│     - Enables fast ANN search       │
│     - Automatically maintained      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Update Metadata                 │
│     - Link embedding_id to chunk    │
│     - Mark document as indexed      │
│     - Status: 'complete'            │
└─────────────────────────────────────┘
```

**Data transformations:**
- Input: Text chunks
- Output: Vector embeddings in ChromaDB

**Code example:**
```python
from sentence_transformers import SentenceTransformer
import chromadb

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
chunks = ["text of chunk 1", "text of chunk 2", ...]
embeddings = model.encode(chunks, normalize_embeddings=True)

# Store in ChromaDB
client = chromadb.PersistentClient(path="data/chromadb")
collection = client.get_or_create_collection("paper_chunks")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings.tolist(),
    metadatas=[{"document_id": "doc_123", ...} for _ in chunks],
    documents=chunks
)
```

**Why normalize embeddings:**
- Cosine similarity becomes dot product
- Faster computation
- Consistent similarity scores

#### Phase 7: Query and Retrieval

**Semantic Search Flow:**

```
User Query: "How do transformers work?"
     │
     ▼
┌─────────────────────────────────────┐
│  1. Generate Query Embedding        │
│     - Use same model as chunks      │
│     - Encode query to 384-dim vector│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Vector Search                   │
│     - Query ChromaDB                │
│     - Find top-k similar chunks     │
│     - Use HNSW for fast search      │
│     - k = 50 (retrieve more, rank)  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Apply Filters                   │
│     - Filter by metadata:           │
│       * Year range                  │
│       * Authors                     │
│       * Venue                       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Aggregate by Document           │
│     - Group chunks by document_id   │
│     - Sum similarity scores         │
│     - Rank documents                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Fetch Document Metadata         │
│     - Query SQLite for details      │
│     - Get: title, authors, abstract │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. Return Results                  │
│     - Top 10 documents              │
│     - With relevant excerpts        │
│     - Similarity scores             │
└─────────────────────────────────────┘
```

**Graph Query Flow:**

```
User Query: "Papers related to attention mechanisms"
     │
     ▼
┌─────────────────────────────────────┐
│  1. Find Concept Node               │
│     - Query Neo4j                   │
│     - MATCH (c:Concept {name: ...}) │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Traverse Relationships          │
│     - Find papers mentioning concept│
│     - Find related concepts         │
│     - Find papers mentioning related│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Rank by Relevance               │
│     - Direct mentions: high score   │
│     - Related concepts: medium score│
│     - Weight by mention frequency   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Return Results                  │
│     - Papers with relevance scores  │
│     - Explanation of connections    │
└─────────────────────────────────────┘
```

**RAG Flow:**

```
User Question: "What are the advantages of transformers?"
     │
     ▼
┌─────────────────────────────────────┐
│  1. Semantic Search                 │
│     - Retrieve top 5 relevant chunks│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Construct Prompt                │
│     - System: "You are a research   │
│       assistant..."                 │
│     - Context: Retrieved chunks     │
│     - Question: User's question     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. Call LLM                        │
│     - OpenAI API or local Ollama    │
│     - Generate answer               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Extract Citations               │
│     - Parse paper references        │
│     - Link to source documents      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Return Answer                   │
│     - Generated text                │
│     - Citations with links          │
│     - Confidence indicator          │
└─────────────────────────────────────┘
```

### Error and Failure Handling

**Error Categories:**

1. **User Errors:**
   - Invalid file format → Reject with clear message
   - File too large → Reject with size limit
   - Empty query → Prompt for input

2. **Processing Errors:**
   - PDF parsing failure → Retry with different parser, then fail gracefully
   - Model inference error → Retry, then use fallback method
   - Database connection error → Retry with exponential backoff

3. **System Errors:**
   - Out of memory → Process in smaller batches
   - Disk full → Alert user, pause processing
   - Model not found → Download automatically or fail with instructions

**Retry Strategy:**

```python
def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_time)
```

**Graceful Degradation:**

- If concept extraction fails → Continue with chunks only
- If graph update fails → Continue with vector search only
- If embedding fails → Fall back to keyword search
- If LLM fails → Return retrieved chunks without generation

**Logging and Monitoring:**

```python
import logging

logger = logging.getLogger(__name__)

# Log levels
logger.debug("Processing chunk 5/45")  # Detailed progress
logger.info("Document processed successfully")  # Key events
logger.warning("Metadata extraction incomplete")  # Recoverable issues
logger.error("PDF parsing failed", exc_info=True)  # Failures
logger.critical("Database connection lost")  # System failures
```

**User Feedback:**

- Processing: "Processing document... (Step 3/7: Extracting concepts)"
- Success: "Document processed successfully. 45 chunks, 12 concepts extracted."
- Partial failure: "Document processed with warnings. Metadata extraction incomplete."
- Failure: "Processing failed: Unable to parse PDF. Please check file integrity."


## Correctness Properties

### What Are Correctness Properties?

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

Think of properties as universal rules: "For ANY valid input of type X, the system MUST produce output with characteristic Y." Unlike unit tests that check specific examples, property-based tests verify these rules across hundreds or thousands of randomly generated inputs, catching edge cases you might never think to test manually.

**Why properties matter for this project:**
- Catch bugs that specific examples miss
- Document system behavior formally
- Provide confidence that the system works correctly across all inputs
- Demonstrate understanding of correctness to interviewers

### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies and consolidations:

**Redundancies eliminated:**
- Properties 1.1 and 1.4 both test PDF acceptance - consolidated into single property about valid PDF handling
- Properties 2.1 and 2.3 both test content preservation - consolidated into single property about lossless extraction
- Properties 6.1 and 6.2 both test embedding storage - consolidated into single property about embedding persistence
- Properties 7.1 and 7.2 both test search retrieval - consolidated into single property about search functionality
- Properties 10.1, 10.2, and 10.7 all test data persistence - consolidated into single property about referential integrity

**Properties combined:**
- Chunking properties 3.1, 3.4, 3.5, 3.6 combined into comprehensive chunking correctness property
- Concept extraction properties 4.1, 4.3, 4.6, 4.7 combined into comprehensive extraction property
- Graph construction properties 5.1, 5.2, 5.3 combined into comprehensive graph building property

This reduces 80+ potential properties to 35 focused, non-redundant properties that provide comprehensive coverage.

### Core Properties

#### Property 1: Valid PDF Acceptance and Storage
*For any* valid PDF file under 50MB, when uploaded to the platform, the system should accept it, store it securely, and return a unique document identifier that can be used to retrieve the file.

**Validates: Requirements 1.1, 1.4**

#### Property 2: Invalid File Rejection
*For any* file that is not a valid PDF format or exceeds 50MB, when uploaded to the platform, the system should reject it and return a descriptive error message explaining why the file was rejected.

**Validates: Requirements 1.2, 1.3**

#### Property 3: Concurrent Upload Independence
*For any* set of PDF files uploaded concurrently, each document should be processed independently such that the success or failure of one upload does not affect the processing of others.

**Validates: Requirements 1.6**

#### Property 4: Lossless Text Extraction
*For any* valid PDF document, when parsed, all readable text content should be extracted and preserved in the output JSON, with no text loss or corruption.

**Validates: Requirements 2.1, 2.3**

#### Property 5: Metadata Extraction Completeness
*For any* PDF document containing metadata fields (title, authors, date, abstract), when parsed, all available metadata should be extracted and included in the structured output.

**Validates: Requirements 2.2**

#### Property 6: Structured Output Format
*For any* successfully parsed PDF, the output should be valid JSON conforming to the defined schema with required fields: document_id, metadata, and sections.

**Validates: Requirements 2.7**

#### Property 7: Parse Failure Logging
*For any* PDF that fails to parse, the system should log detailed error information and notify the user with specific failure details, without exposing internal system details.

**Validates: Requirements 2.5, 11.2**

#### Property 8: Semantic Chunking Correctness
*For any* parsed document, when chunked, every chunk should: (1) contain between 100-500 tokens, (2) respect section boundaries (never split across sections), (3) maintain a reference to its source document and position, and (4) preserve semantic coherence.

**Validates: Requirements 3.1, 3.4, 3.5, 3.6**

#### Property 9: Chunk Semantic Coherence
*For any* chunk created by the chunking service, the sentences within that chunk should have higher average semantic similarity to each other than to sentences in adjacent chunks.

**Validates: Requirements 3.3**

#### Property 10: Concept Extraction Completeness
*For any* document chunk, when processed by the concept extractor, the output should include: (1) identified concepts with types, (2) named entities with categories, (3) confidence scores for each extraction, and (4) structured data conforming to the defined schema.

**Validates: Requirements 4.1, 4.3, 4.6, 4.7**

#### Property 11: Concept Normalization Consistency
*For any* two mentions of the same concept using different phrasings (e.g., "BERT" and "Bidirectional Encoder Representations from Transformers"), the concept extractor should normalize them to the same canonical form.

**Validates: Requirements 4.5**

#### Property 12: Relationship Extraction
*For any* document chunk containing multiple concepts, the concept extractor should identify and output relationships between co-occurring concepts.

**Validates: Requirements 4.4**

#### Property 13: Knowledge Graph Construction
*For any* document with extracted concepts, when added to the knowledge graph, the system should: (1) create a Paper node with metadata, (2) create or link to Concept nodes, (3) create MENTIONS relationships with frequency counts, and (4) maintain referential integrity.

**Validates: Requirements 5.1, 5.2, 5.3**

#### Property 14: Concept Aggregation Accuracy
*For any* concept that appears in multiple papers, the knowledge graph should correctly aggregate the total mention count across all papers.

**Validates: Requirements 5.5**

#### Property 15: Bidirectional Relationship Navigation
*For any* relationship (A)-[R]->(B) created in the knowledge graph, it should be possible to traverse from B back to A (either through a reverse relationship or bidirectional query).

**Validates: Requirements 5.6**

#### Property 16: Incremental Graph Updates
*For any* existing knowledge graph, when a new paper is added, the graph should be updated without modifying or corrupting existing nodes and relationships.

**Validates: Requirements 5.7**

#### Property 17: Embedding Generation and Storage
*For any* document chunk, the system should generate a semantic embedding and store it in the vector store with metadata linking it back to the source chunk and document.

**Validates: Requirements 6.1, 6.2**

#### Property 18: Embedding Normalization
*For any* generated embedding vector, it should be normalized (L2 norm = 1) to ensure consistent similarity computation.

**Validates: Requirements 6.5**

#### Property 19: Batch Embedding Processing
*For any* batch of document chunks, the system should process them together and store all embeddings successfully, with no partial failures leaving some chunks without embeddings.

**Validates: Requirements 6.6**

#### Property 20: Search Query Processing
*For any* natural language search query, the semantic search engine should convert it to an embedding and retrieve the top-k most similar chunks from the vector store.

**Validates: Requirements 7.1, 7.2**

#### Property 21: Search Result Ranking
*For any* search results returned, they should be ordered by descending semantic similarity score, with the most similar chunks appearing first.

**Validates: Requirements 7.3**

#### Property 22: Search Result Completeness
*For any* search result, the returned data should include complete document metadata (title, authors, year) and the relevant chunk excerpt.

**Validates: Requirements 7.4, 7.5**

#### Property 23: Search Metadata Filtering
*For any* search query with metadata filters (e.g., year > 2020, author = "Hinton"), all returned results should satisfy the specified filter conditions.

**Validates: Requirements 7.6**

#### Property 24: RAG Retrieval Integration
*For any* user question to the research assistant, the system should retrieve relevant chunks using semantic search before generating an answer.

**Validates: Requirements 8.1**

#### Property 25: RAG Answer Generation
*For any* user question with retrieved context, the research assistant should generate an answer using the LLM that references the retrieved content.

**Validates: Requirements 8.2**

#### Property 26: Citation Inclusion
*For any* answer generated by the research assistant, it should include citations to specific papers and sections from the retrieved context.

**Validates: Requirements 8.3**

#### Property 27: Multi-Source Synthesis
*For any* question that can be answered using multiple papers, the research assistant should synthesize information from multiple retrieved chunks when generating the answer.

**Validates: Requirements 8.4**

#### Property 28: Confidence Indication
*For any* answer generated by the research assistant, it should include a confidence indicator or explicitly acknowledge when information is not available in the collection.

**Validates: Requirements 8.5**

#### Property 29: Conversation Context Maintenance
*For any* follow-up question in a conversation, the research assistant should maintain context from previous questions and answers to interpret the follow-up correctly.

**Validates: Requirements 8.6**

#### Property 30: Summarization Capability
*For any* request for a summary of a paper or concept area, the research assistant should generate a concise overview based on the relevant documents.

**Validates: Requirements 8.7**

#### Property 31: Processing State Tracking
*For any* document in the processing pipeline, the system should track and update its status at each stage (uploaded, parsing, chunking, extracting, complete, failed).

**Validates: Requirements 9.2, 9.6**

#### Property 32: Retry on Failure
*For any* processing stage that fails, the system should automatically retry the operation up to 3 times with exponential backoff before marking the document as failed.

**Validates: Requirements 9.3, 9.4**

#### Property 33: Completion Notification
*For any* document that completes processing successfully, the system should notify the user and make the document available for search.

**Validates: Requirements 9.7**

#### Property 34: Data Persistence and Recovery
*For any* data stored in the system (PDFs, metadata, chunks, concepts, embeddings, graph), when the system restarts, all data should be restored without loss.

**Validates: Requirements 10.1, 10.2, 10.5**

#### Property 35: Referential Integrity
*For any* chunk, concept, or embedding in the system, all references to parent documents should be valid and resolvable (no orphaned data).

**Validates: Requirements 10.7**

#### Property 36: Error Logging
*For any* error encountered by any component, detailed error information should be logged for debugging purposes.

**Validates: Requirements 11.1**

#### Property 37: Fallback on Model Failure
*For any* deep learning model that fails during processing, the system should attempt to fall back to a simpler processing method when possible, rather than failing completely.

**Validates: Requirements 11.4**

#### Property 38: Operation Queuing on Service Unavailability
*For any* external service that becomes unavailable, the system should queue operations for retry rather than failing immediately.

**Validates: Requirements 11.6**

#### Property 39: Concurrent Processing Support
*For any* set of up to 10 documents being processed simultaneously, the system should handle them concurrently without errors or data corruption.

**Validates: Requirements 12.3**

#### Property 40: Graceful Degradation Under Load
*For any* system state where load exceeds capacity, the system should degrade gracefully by queuing requests rather than failing or crashing.

**Validates: Requirements 12.7**


## Error Handling

### Error Categories and Strategies

#### 1. Input Validation Errors

**Scenario:** User uploads invalid file
**Detection:** File format validation, size check
**Handling:**
- Reject immediately with clear error message
- No retry (user must provide valid file)
- Log for analytics (track common issues)

**Example:**
```python
def validate_upload(file):
    if not file.filename.endswith('.pdf'):
        raise ValidationError("Only PDF files are supported. Please upload a .pdf file.")
    
    if file.size > 50 * 1024 * 1024:  # 50MB
        raise ValidationError(f"File too large ({file.size / 1024 / 1024:.1f}MB). Maximum size is 50MB.")
    
    # Validate PDF magic bytes
    if not file.read(4) == b'%PDF':
        raise ValidationError("File is not a valid PDF. Please check the file format.")
```

#### 2. Processing Errors

**Scenario:** PDF parsing fails
**Detection:** Exception during PyMuPDF processing
**Handling:**
- Retry with exponential backoff (3 attempts)
- Try alternative parser if available
- If all retries fail, mark document as failed
- Provide specific error to user

**Example:**
```python
@retry(max_attempts=3, backoff=exponential)
def parse_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return extract_text(doc)
    except fitz.FileDataError:
        raise ParseError("PDF file is corrupted or encrypted")
    except Exception as e:
        raise ParseError(f"Failed to parse PDF: {str(e)}")
```

#### 3. Model Inference Errors

**Scenario:** Embedding model fails (OOM, model not loaded)
**Detection:** Exception during model.encode()
**Handling:**
- Retry once
- If OOM, reduce batch size and retry
- If model missing, attempt to download
- Fall back to simpler method (keyword extraction)

**Example:**
```python
def generate_embeddings(chunks):
    try:
        return model.encode(chunks, batch_size=32)
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM during embedding, reducing batch size")
        return model.encode(chunks, batch_size=8)
    except Exception as e:
        logger.error(f"Embedding failed: {e}, falling back to TF-IDF")
        return fallback_tfidf_embeddings(chunks)
```

#### 4. Database Errors

**Scenario:** Database connection lost, transaction fails
**Detection:** Database exception
**Handling:**
- Retry with exponential backoff
- Use connection pooling with health checks
- Implement circuit breaker pattern
- Queue operations if database unavailable

**Example:**
```python
@retry(max_attempts=5, backoff=exponential, max_delay=30)
def store_in_database(data):
    try:
        with db.transaction():
            db.insert(data)
    except ConnectionError:
        logger.warning("Database connection lost, retrying...")
        raise  # Trigger retry
    except IntegrityError as e:
        logger.error(f"Data integrity violation: {e}")
        raise  # Don't retry, data issue
```

#### 5. External Service Errors

**Scenario:** OpenAI API fails, Neo4j unavailable
**Detection:** HTTP error, connection timeout
**Handling:**
- Retry with exponential backoff
- Implement circuit breaker (stop calling if consistently failing)
- Queue operations for later retry
- Provide degraded functionality if possible

**Example:**
```python
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

@circuit_breaker
@retry(max_attempts=3, backoff=exponential)
def call_openai_api(prompt):
    try:
        response = openai.ChatCompletion.create(...)
        return response
    except openai.error.RateLimitError:
        logger.warning("Rate limited, backing off")
        raise  # Trigger retry
    except openai.error.APIError:
        logger.error("OpenAI API error")
        raise  # Trigger circuit breaker if repeated
```

### Graceful Degradation

When components fail, the system should degrade gracefully rather than failing completely:

**Degradation Hierarchy:**

1. **Full Functionality:** All components working
   - Semantic search with embeddings
   - Knowledge graph queries
   - AI-powered answers with citations

2. **Degraded Mode 1:** Embedding model fails
   - Fall back to keyword search (BM25)
   - Graph queries still work
   - AI answers without semantic retrieval

3. **Degraded Mode 2:** Graph database unavailable
   - Semantic search still works
   - No relationship queries
   - AI answers from vector search only

4. **Degraded Mode 3:** LLM unavailable
   - Search still works
   - Return raw chunks instead of generated answers
   - User can read retrieved content directly

5. **Minimal Mode:** Only core storage works
   - Can upload and store PDFs
   - Can view uploaded documents
   - Processing queued for when services recover

**Implementation:**
```python
def search(query):
    try:
        # Try semantic search
        return semantic_search(query)
    except EmbeddingError:
        logger.warning("Semantic search failed, falling back to keyword search")
        return keyword_search(query)
    except Exception as e:
        logger.error(f"All search methods failed: {e}")
        return {"error": "Search temporarily unavailable", "status": "degraded"}
```

### Logging and Monitoring

**Log Levels:**
- **DEBUG:** Detailed processing steps (chunk 5/45, embedding batch 2/10)
- **INFO:** Key events (document uploaded, processing complete)
- **WARNING:** Recoverable issues (retry triggered, fallback used)
- **ERROR:** Failures (parsing failed, model error)
- **CRITICAL:** System failures (database down, disk full)

**What to Log:**
```python
# Good logging
logger.info(f"Document {doc_id} uploaded: {filename} ({size}MB)")
logger.debug(f"Parsing page {page_num}/{total_pages}")
logger.warning(f"Metadata extraction incomplete for {doc_id}: missing authors")
logger.error(f"Failed to parse {doc_id} after 3 retries", exc_info=True)
logger.critical("Database connection pool exhausted")
```

**Monitoring Metrics:**
- Documents processed per hour
- Average processing time per document
- Error rate by component
- Retry rate by operation
- Database query latency
- Model inference latency
- Storage usage

### User-Facing Error Messages

**Principles:**
- Be specific about what went wrong
- Suggest corrective action when possible
- Don't expose internal details
- Be empathetic and helpful

**Examples:**

❌ Bad: "Error 500: Internal server error"
✅ Good: "We couldn't process your PDF. This might be because the file is corrupted or password-protected. Please try a different file."

❌ Bad: "NoneType object has no attribute 'text'"
✅ Good: "We couldn't extract text from this PDF. This sometimes happens with scanned documents. Try uploading a text-based PDF instead."

❌ Bad: "Database connection failed"
✅ Good: "We're experiencing technical difficulties. Your upload has been saved and will be processed automatically when the system recovers."


## Testing Strategy

### Dual Testing Approach

This project requires both **unit tests** and **property-based tests** for comprehensive coverage. They serve different purposes and are complementary:

**Unit Tests:**
- Test specific examples and edge cases
- Verify integration between components
- Test error conditions and failure modes
- Fast to run, easy to debug
- Example: "Test that uploading a 51MB file is rejected"

**Property-Based Tests:**
- Verify universal properties across all inputs
- Generate hundreds of random test cases
- Catch edge cases you didn't think of
- Higher confidence in correctness
- Example: "For ANY valid PDF, parsing should extract all text"

**Why both are necessary:**
- Unit tests catch known issues and regressions
- Property tests catch unknown edge cases
- Together they provide comprehensive coverage
- This is how production systems are tested

### Property-Based Testing Framework

**For Python: Hypothesis**

Hypothesis is the industry-standard property-based testing library for Python. It integrates with pytest and generates random test inputs.

**Installation:**
```bash
pip install hypothesis pytest
```

**Basic Example:**
```python
from hypothesis import given, strategies as st
import pytest

@given(st.text(min_size=1, max_size=1000))
def test_chunking_preserves_content(text):
    """Property: Chunking should preserve all content"""
    chunks = chunk_text(text)
    reconstructed = ''.join(chunks)
    assert reconstructed == text
```

**Configuration:**
- Minimum 100 iterations per property test (due to randomization)
- Each test should reference its design property
- Tag format: `# Feature: research-literature-platform, Property N: [property text]`

### Test Organization

```
tests/
├── unit/
│   ├── test_ingestion.py
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_concept_extractor.py
│   ├── test_graph_builder.py
│   ├── test_vector_store.py
│   ├── test_search.py
│   └── test_rag.py
├── property/
│   ├── test_ingestion_properties.py
│   ├── test_parser_properties.py
│   ├── test_chunker_properties.py
│   ├── test_concept_properties.py
│   ├── test_graph_properties.py
│   ├── test_vector_properties.py
│   ├── test_search_properties.py
│   └── test_rag_properties.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_workflow.py
└── fixtures/
    ├── sample_pdfs/
    └── test_data.py
```

### Example Property Tests

#### Property 1: Valid PDF Acceptance and Storage

```python
from hypothesis import given, strategies as st
import pytest
from pathlib import Path

# Feature: research-literature-platform, Property 1: Valid PDF Acceptance and Storage
@given(st.binary(min_size=1024, max_size=50*1024*1024))
@pytest.mark.property
def test_valid_pdf_acceptance(pdf_content):
    """For any valid PDF under 50MB, system should accept and store it"""
    # Create temporary PDF file
    pdf_file = create_temp_pdf(pdf_content)
    
    # Upload
    result = ingestion_module.upload(pdf_file)
    
    # Verify acceptance
    assert result.success is True
    assert result.document_id is not None
    
    # Verify storage
    stored_file = file_storage.get(result.document_id)
    assert stored_file.exists()
    assert stored_file.read_bytes() == pdf_content
```

#### Property 8: Semantic Chunking Correctness

```python
# Feature: research-literature-platform, Property 8: Semantic Chunking Correctness
@given(st.text(min_size=1000, max_size=10000))
@pytest.mark.property
def test_chunking_correctness(document_text):
    """For any document, chunks should meet all correctness criteria"""
    chunks = chunking_service.chunk(document_text)
    
    for chunk in chunks:
        # (1) Token count between 100-500
        token_count = len(chunk.text.split())
        assert 100 <= token_count <= 500, f"Chunk has {token_count} tokens"
        
        # (2) Has valid document reference
        assert chunk.document_id is not None
        assert chunk.position >= 0
        
        # (3) Text is not empty
        assert len(chunk.text.strip()) > 0
    
    # (4) Chunks cover entire document
    reconstructed = ' '.join(c.text for c in chunks)
    assert all(word in reconstructed for word in document_text.split())
```

#### Property 11: Concept Normalization Consistency

```python
# Feature: research-literature-platform, Property 11: Concept Normalization Consistency
@given(st.lists(st.text(min_size=3, max_size=50), min_size=2, max_size=10))
@pytest.mark.property
def test_concept_normalization(concept_variants):
    """For any concept with multiple phrasings, normalization should be consistent"""
    # Extract concepts from each variant
    normalized_concepts = []
    for variant in concept_variants:
        concepts = concept_extractor.extract(variant)
        normalized_concepts.extend([c.normalized_name for c in concepts])
    
    # If same concept appears multiple times, should have same normalized form
    concept_groups = {}
    for concept in normalized_concepts:
        if concept not in concept_groups:
            concept_groups[concept] = []
        concept_groups[concept].append(concept)
    
    # Each normalized form should map to exactly one canonical form
    for normalized, variants in concept_groups.items():
        assert all(v == variants[0] for v in variants)
```

#### Property 20: Search Query Processing

```python
# Feature: research-literature-platform, Property 20: Search Query Processing
@given(st.text(min_size=5, max_size=200))
@pytest.mark.property
def test_search_query_processing(query):
    """For any natural language query, search should return results"""
    # Assume we have some documents indexed
    results = search_engine.search(query, top_k=10)
    
    # Should return results (or empty list if no matches)
    assert isinstance(results, list)
    assert len(results) <= 10
    
    # Each result should have required fields
    for result in results:
        assert result.document_id is not None
        assert result.chunk_text is not None
        assert result.similarity_score is not None
        assert 0 <= result.similarity_score <= 1
```

### Example Unit Tests

#### Unit Test: Invalid File Rejection

```python
def test_reject_non_pdf_file():
    """Test that non-PDF files are rejected"""
    # Create a text file
    text_file = create_temp_file("test.txt", "This is not a PDF")
    
    # Attempt upload
    with pytest.raises(ValidationError) as exc_info:
        ingestion_module.upload(text_file)
    
    # Verify error message
    assert "Only PDF files are supported" in str(exc_info.value)
```

#### Unit Test: Metadata Extraction

```python
def test_metadata_extraction():
    """Test that parser extracts metadata correctly"""
    # Use a sample PDF with known metadata
    pdf_path = "tests/fixtures/sample_pdfs/attention_paper.pdf"
    
    # Parse
    result = parser.parse(pdf_path)
    
    # Verify metadata
    assert result.metadata.title == "Attention Is All You Need"
    assert "Vaswani" in result.metadata.authors
    assert result.metadata.year == 2017
```

#### Unit Test: Retry Logic

```python
def test_retry_on_failure():
    """Test that processing retries on failure"""
    # Mock a component that fails twice then succeeds
    mock_parser = Mock(side_effect=[ParseError(), ParseError(), {"text": "success"}])
    
    # Process with retry
    result = orchestrator.process_with_retry(mock_parser, max_retries=3)
    
    # Verify retries
    assert mock_parser.call_count == 3
    assert result == {"text": "success"}
```

### Integration Tests

Integration tests verify that components work together correctly:

```python
def test_end_to_end_processing():
    """Test complete document processing pipeline"""
    # Upload PDF
    pdf_path = "tests/fixtures/sample_pdfs/test_paper.pdf"
    doc_id = ingestion_module.upload(pdf_path)
    
    # Wait for processing to complete
    wait_for_completion(doc_id, timeout=60)
    
    # Verify all stages completed
    doc = database.get_document(doc_id)
    assert doc.status == "complete"
    
    # Verify chunks created
    chunks = database.get_chunks(doc_id)
    assert len(chunks) > 0
    
    # Verify concepts extracted
    concepts = database.get_concepts(doc_id)
    assert len(concepts) > 0
    
    # Verify graph nodes created
    graph_node = graph_db.get_paper(doc_id)
    assert graph_node is not None
    
    # Verify embeddings stored
    embeddings = vector_store.get_embeddings(doc_id)
    assert len(embeddings) == len(chunks)
    
    # Verify searchable
    results = search_engine.search("test query")
    doc_ids = [r.document_id for r in results]
    assert doc_id in doc_ids
```

### Test Data and Fixtures

**Sample PDFs:**
- Simple single-column paper (5 pages)
- Complex multi-column paper (20 pages)
- Paper with tables and figures
- Paper with mathematical equations
- Corrupted PDF (for error testing)
- Password-protected PDF (for error testing)

**Generated Test Data:**
```python
# Hypothesis strategies for domain-specific data
@st.composite
def pdf_documents(draw):
    """Generate realistic PDF document structures"""
    return {
        "title": draw(st.text(min_size=10, max_size=100)),
        "authors": draw(st.lists(st.text(min_size=5, max_size=30), min_size=1, max_size=5)),
        "year": draw(st.integers(min_value=1990, max_value=2024)),
        "abstract": draw(st.text(min_size=100, max_size=500)),
        "sections": draw(st.lists(
            st.dictionaries(
                keys=st.just("heading") | st.just("text"),
                values=st.text(min_size=50, max_size=1000)
            ),
            min_size=3,
            max_size=10
        ))
    }
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only property tests
pytest tests/property/ -m property

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific property test with verbose output
pytest tests/property/test_chunker_properties.py::test_chunking_correctness -v

# Run property tests with more iterations
pytest tests/property/ --hypothesis-iterations=1000
```

### Test Coverage Goals

- **Unit test coverage:** 80%+ of code
- **Property test coverage:** All 40 correctness properties
- **Integration test coverage:** All major workflows
- **Edge case coverage:** All identified edge cases from prework

### Learning Points

**For Interviews:**
- "I used property-based testing to verify correctness across all inputs"
- "I wrote 40 properties covering the entire system specification"
- "I combined unit tests for specific cases with property tests for general correctness"
- "I used Hypothesis to generate thousands of test cases automatically"

**What This Demonstrates:**
- Understanding of software correctness
- Knowledge of advanced testing techniques
- Ability to write testable code
- Production-grade quality assurance


## Project Structure and Organization

### Directory Structure

```
research-literature-platform/
├── README.md                          # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules
│
├── data/                             # Data storage (gitignored)
│   ├── pdfs/                        # Uploaded PDF files
│   ├── chromadb/                    # Vector store persistence
│   ├── metadata.db                  # SQLite database
│   └── logs/                        # Application logs
│
├── models/                           # Downloaded ML models (gitignored)
│   ├── sentence-transformers/
│   └── spacy/
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── ingestion/                   # PDF ingestion module
│   │   ├── __init__.py
│   │   ├── uploader.py             # File upload handling
│   │   └── validator.py            # File validation
│   │
│   ├── parsing/                     # PDF parsing module
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # PyMuPDF wrapper
│   │   ├── metadata_extractor.py  # Metadata extraction
│   │   └── structure_analyzer.py  # Document structure analysis
│   │
│   ├── chunking/                    # Semantic chunking module
│   │   ├── __init__.py
│   │   ├── semantic_chunker.py    # Embedding-based chunking
│   │   └── boundary_detector.py   # Semantic boundary detection
│   │
│   ├── extraction/                  # Concept extraction module
│   │   ├── __init__.py
│   │   ├── concept_extractor.py   # Main extraction logic
│   │   ├── ner.py                 # Named entity recognition
│   │   ├── keyphrase.py           # Keyphrase extraction
│   │   └── normalizer.py          # Concept normalization
│   │
│   ├── graph/                       # Knowledge graph module
│   │   ├── __init__.py
│   │   ├── graph_builder.py       # Graph construction
│   │   ├── neo4j_client.py        # Neo4j connection
│   │   └── queries.py             # Cypher query templates
│   │
│   ├── embeddings/                  # Embedding generation module
│   │   ├── __init__.py
│   │   ├── embedding_generator.py # Sentence-BERT wrapper
│   │   └── model_manager.py       # Model loading and caching
│   │
│   ├── vector_store/                # Vector storage module
│   │   ├── __init__.py
│   │   ├── chroma_client.py       # ChromaDB wrapper
│   │   └── indexer.py             # Embedding indexing
│   │
│   ├── search/                      # Search module
│   │   ├── __init__.py
│   │   ├── semantic_search.py     # Vector-based search
│   │   ├── graph_search.py        # Graph-based queries
│   │   └── hybrid_search.py       # Combined search
│   │
│   ├── rag/                         # RAG module
│   │   ├── __init__.py
│   │   ├── research_assistant.py  # Main RAG logic
│   │   ├── retriever.py           # Context retrieval
│   │   ├── generator.py           # LLM integration
│   │   └── citation_extractor.py  # Citation parsing
│   │
│   ├── orchestration/               # Workflow orchestration
│   │   ├── __init__.py
│   │   ├── workflow.py            # LangGraph workflow
│   │   ├── state.py               # State management
│   │   └── retry.py               # Retry logic
│   │
│   ├── storage/                     # Data persistence
│   │   ├── __init__.py
│   │   ├── database.py            # SQLite operations
│   │   ├── file_storage.py        # File system operations
│   │   └── models.py              # Data models (SQLAlchemy)
│   │
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py             # Logging configuration
│   │   ├── config.py              # Configuration management
│   │   └── errors.py              # Custom exceptions
│   │
│   └── api/                         # API layer (optional)
│       ├── __init__.py
│       ├── app.py                 # FastAPI application
│       └── routes.py              # API endpoints
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   │
│   ├── unit/                       # Unit tests
│   │   ├── test_ingestion.py
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_concept_extractor.py
│   │   ├── test_graph_builder.py
│   │   ├── test_vector_store.py
│   │   ├── test_search.py
│   │   └── test_rag.py
│   │
│   ├── property/                   # Property-based tests
│   │   ├── test_ingestion_properties.py
│   │   ├── test_parser_properties.py
│   │   ├── test_chunker_properties.py
│   │   ├── test_concept_properties.py
│   │   ├── test_graph_properties.py
│   │   ├── test_vector_properties.py
│   │   ├── test_search_properties.py
│   │   └── test_rag_properties.py
│   │
│   ├── integration/                # Integration tests
│   │   ├── test_end_to_end.py
│   │   └── test_workflow.py
│   │
│   └── fixtures/                   # Test data
│       ├── sample_pdfs/
│       └── test_data.py
│
├── notebooks/                        # Jupyter notebooks (optional)
│   ├── 01_exploration.ipynb       # Data exploration
│   ├── 02_model_testing.ipynb     # Model experimentation
│   └── 03_visualization.ipynb     # Graph visualization
│
├── scripts/                          # Utility scripts
│   ├── setup_environment.sh       # Environment setup
│   ├── download_models.py         # Download ML models
│   ├── init_databases.py          # Initialize databases
│   └── run_pipeline.py            # Run processing pipeline
│
└── docs/                            # Documentation
    ├── architecture.md            # Architecture details
    ├── api.md                     # API documentation
    ├── deployment.md              # Deployment guide
    └── development.md             # Development guide
```

### Why This Structure?

**Separation of Concerns:**
- Each module has a single, well-defined responsibility
- Easy to find code related to specific functionality
- Easy to test modules independently

**Modularity:**
- Modules can be developed and tested independently
- Easy to swap implementations (e.g., different embedding models)
- Clear interfaces between modules

**Scalability:**
- Structure supports growth (add new modules without restructuring)
- Clear separation makes it easy to split into microservices later
- Data and code are separated

**Testability:**
- Test structure mirrors source structure
- Easy to find tests for specific modules
- Fixtures and test data are organized

**Professional Standards:**
- Follows Python packaging conventions
- Clear separation of source, tests, data, and docs
- Gitignore prevents committing large files (models, data)

### Key Files Explained

**requirements.txt:**
```
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
spacy>=3.5.0

# PDF processing
pymupdf>=1.22.0

# Databases
chromadb>=0.4.0
neo4j>=5.0.0
sqlalchemy>=2.0.0

# Orchestration
langgraph>=0.1.0
langchain>=0.1.0

# API (optional)
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing
pytest>=7.4.0
hypothesis>=6.82.0
pytest-cov>=4.1.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="research-literature-platform",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Read from requirements.txt
    ],
    python_requires=">=3.10",
)
```

**.env.example:**
```
# Database paths
SQLITE_DB_PATH=data/metadata.db
CHROMADB_PATH=data/chromadb
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Model paths
MODELS_DIR=models/
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
SPACY_MODEL=en_core_sci_md

# OpenAI (optional)
OPENAI_API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/logs/app.log

# Processing
MAX_CONCURRENT_DOCUMENTS=10
CHUNK_SIZE_MIN=100
CHUNK_SIZE_MAX=500
```

### Virtual Environment Setup

**Why virtual environments:**
- Isolate project dependencies
- Avoid conflicts with system Python
- Reproducible environment
- Easy to share with others

**Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .

# Download models
python scripts/download_models.py

# Initialize databases
python scripts/init_databases.py
```

### Configuration Management

**Using python-dotenv:**
```python
# src/utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Database
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/metadata.db")
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "data/chromadb")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    
    # Models
    MODELS_DIR = os.getenv("MODELS_DIR", "models/")
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    
    # Processing
    MAX_CONCURRENT_DOCUMENTS = int(os.getenv("MAX_CONCURRENT_DOCUMENTS", "10"))
    CHUNK_SIZE_MIN = int(os.getenv("CHUNK_SIZE_MIN", "100"))
    CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", "500"))

config = Config()
```

### Dependency Injection

**Why dependency injection:**
- Makes code testable (inject mocks)
- Makes code flexible (swap implementations)
- Makes dependencies explicit
- Follows SOLID principles

**Example:**
```python
# src/orchestration/workflow.py
class DocumentProcessor:
    def __init__(
        self,
        parser: PDFParser,
        chunker: SemanticChunker,
        extractor: ConceptExtractor,
        graph_builder: GraphBuilder,
        vector_store: VectorStore
    ):
        self.parser = parser
        self.chunker = chunker
        self.extractor = extractor
        self.graph_builder = graph_builder
        self.vector_store = vector_store
    
    def process(self, document_id: str):
        # Use injected dependencies
        parsed = self.parser.parse(document_id)
        chunks = self.chunker.chunk(parsed)
        concepts = self.extractor.extract(chunks)
        self.graph_builder.build(document_id, concepts)
        self.vector_store.store(chunks)

# In tests, inject mocks
def test_processing():
    mock_parser = Mock()
    mock_chunker = Mock()
    # ... other mocks
    
    processor = DocumentProcessor(
        parser=mock_parser,
        chunker=mock_chunker,
        # ... other mocks
    )
    
    processor.process("test_doc")
    
    # Verify interactions
    mock_parser.parse.assert_called_once()
```

### Learning Points

**For Interviews:**
- "I organized the project with clear separation of concerns"
- "Each module has a single responsibility and well-defined interfaces"
- "I used dependency injection to make the code testable and flexible"
- "The structure supports scaling to microservices if needed"

**What This Demonstrates:**
- Software engineering best practices
- Understanding of modularity and separation of concerns
- Ability to structure large projects
- Production-grade code organization


## Implementation Phases

This section breaks the project into manageable engineering phases. Each phase builds on the previous one, allowing you to learn incrementally and have working functionality at each step.

### Phase 0: Environment and Project Setup

**Objective:** Set up development environment and project structure

**Duration:** 1-2 days

**What You'll Learn:**
- Python virtual environments
- Dependency management
- Project structure best practices
- Configuration management

**Tasks:**
1. Create project directory structure
2. Set up virtual environment
3. Install core dependencies
4. Configure logging
5. Set up Git repository
6. Create initial documentation

**Inputs:** None (starting from scratch)

**Outputs:**
- Working development environment
- Project skeleton with all directories
- Configuration files
- README with setup instructions

**Key Challenges:**
- Getting Neo4j running locally
- Understanding ChromaDB setup
- Managing multiple dependencies

**Success Criteria:**
- Can run `python -c "import torch; import transformers; import chromadb"` without errors
- Neo4j accessible at localhost:7687
- All directories created
- Git repository initialized

**Skills Learned:**
- Environment management
- Dependency resolution
- Database setup
- Project organization

---

### Phase 1: PDF Ingestion and Parsing

**Objective:** Accept PDF uploads and extract text with metadata

**Duration:** 3-5 days

**What You'll Learn:**
- File handling in Python
- PDF structure and parsing
- Error handling and validation
- Database operations (SQLite)

**Tasks:**
1. Implement file upload validation
2. Build PDF parser using PyMuPDF
3. Extract text and preserve structure
4. Extract metadata (title, authors, date)
5. Store files and metadata in SQLite
6. Implement error handling
7. Write unit tests for parser
8. Write property tests for validation

**Inputs:** PDF files from user

**Outputs:**
- Stored PDF files
- Structured JSON with text and metadata
- Database records

**Key Challenges:**
- Handling corrupted PDFs
- Extracting metadata reliably (papers have inconsistent formats)
- Preserving document structure (sections, paragraphs)
- Multi-column layouts

**Success Criteria:**
- Can upload and parse 10 different research papers
- Metadata extracted for 80%+ of papers
- All text content preserved
- Error handling works for invalid files

**Skills Learned:**
- PDF processing
- Text extraction
- Metadata parsing with regex
- File I/O
- Database operations
- Error handling

**Code Example:**
```python
# src/parsing/pdf_parser.py
import fitz  # PyMuPDF

class PDFParser:
    def parse(self, pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        
        # Extract text
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Extract metadata
        metadata = self._extract_metadata(doc, text)
        
        # Identify structure
        sections = self._identify_sections(doc)
        
        return {
            "text": text,
            "metadata": metadata,
            "sections": sections
        }
```

---

### Phase 2: Deep Learning-Based Chunking and Concept Extraction

**Objective:** Segment documents semantically and extract key concepts

**Duration:** 5-7 days

**What You'll Learn:**
- Using pre-trained transformers
- Sentence embeddings
- Semantic similarity
- Named entity recognition
- Keyphrase extraction

**Tasks:**
1. Load Sentence-BERT model
2. Implement semantic chunking algorithm
3. Test chunking on sample papers
4. Load SpaCy scientific NER model
5. Implement concept extraction
6. Implement keyphrase extraction
7. Build concept normalization
8. Write property tests for chunking
9. Write unit tests for extraction

**Inputs:** Parsed document JSON

**Outputs:**
- Semantic chunks (100-500 tokens each)
- Extracted concepts with types
- Named entities
- Keyphrases

**Key Challenges:**
- Understanding embeddings and cosine similarity
- Tuning semantic boundary threshold
- Handling scientific terminology
- Concept normalization (synonyms)
- Model memory usage

**Success Criteria:**
- Chunks preserve semantic coherence
- All chunks within token limits
- Concepts extracted from 90%+ of chunks
- Normalization works for common synonyms

**Skills Learned:**
- Using Hugging Face transformers
- Sentence embeddings
- Semantic similarity computation
- NLP model inference
- Scientific NER
- Keyphrase extraction

**Code Example:**
```python
# src/chunking/semantic_chunker.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk(self, text: str, threshold: float = 0.7) -> list:
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Generate embeddings
        embeddings = self.model.encode(sentences)
        
        # Detect boundaries
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [embeddings[i-1]], 
                [embeddings[i]]
            )[0][0]
            
            if similarity < threshold:
                # Boundary detected
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        chunks.append(' '.join(current_chunk))
        return chunks
```

---

### Phase 3: Knowledge Graph Construction

**Objective:** Build a graph database of papers, concepts, and relationships

**Duration:** 4-6 days

**What You'll Learn:**
- Graph databases (Neo4j)
- Cypher query language
- Graph data modeling
- Relationship extraction
- Graph algorithms

**Tasks:**
1. Set up Neo4j connection
2. Design graph schema
3. Implement graph builder
4. Create paper nodes
5. Create concept nodes
6. Create relationships (MENTIONS, RELATED_TO)
7. Implement graph queries
8. Write property tests for graph integrity
9. Visualize graph in Neo4j browser

**Inputs:** Documents, concepts, entities

**Outputs:**
- Neo4j graph database
- Nodes: Papers, Concepts, Authors, Venues
- Edges: MENTIONS, CITES, AUTHORED_BY, RELATED_TO

**Key Challenges:**
- Learning Cypher query language
- Designing efficient graph schema
- Computing concept relationships
- Handling graph updates incrementally
- Avoiding duplicate nodes

**Success Criteria:**
- Graph contains all papers and concepts
- Relationships are accurate
- Can query for related papers
- Can find concept clusters
- Graph updates work incrementally

**Skills Learned:**
- Graph databases
- Cypher queries
- Graph data modeling
- Relationship extraction
- Graph traversal

**Code Example:**
```python
# src/graph/graph_builder.py
from neo4j import GraphDatabase

class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_paper(self, paper_data: dict):
        with self.driver.session() as session:
            session.run("""
                CREATE (p:Paper {
                    id: $id,
                    title: $title,
                    authors: $authors,
                    year: $year
                })
            """, **paper_data)
    
    def link_concept(self, paper_id: str, concept: str, frequency: int):
        with self.driver.session() as session:
            session.run("""
                MATCH (p:Paper {id: $paper_id})
                MERGE (c:Concept {name: $concept})
                CREATE (p)-[:MENTIONS {frequency: $frequency}]->(c)
            """, paper_id=paper_id, concept=concept, frequency=frequency)
```

---

### Phase 4: Semantic Search and Retrieval

**Objective:** Enable vector-based semantic search over documents

**Duration:** 4-6 days

**What You'll Learn:**
- Vector databases
- Approximate nearest neighbor search
- Embedding storage and retrieval
- Search ranking
- Metadata filtering

**Tasks:**
1. Set up ChromaDB
2. Generate embeddings for all chunks
3. Store embeddings with metadata
4. Implement semantic search
5. Implement result ranking
6. Add metadata filtering
7. Implement hybrid search (vector + keyword)
8. Write property tests for search
9. Optimize search performance

**Inputs:** User query (natural language)

**Outputs:**
- Ranked list of relevant documents
- Relevant chunk excerpts
- Similarity scores

**Key Challenges:**
- Understanding vector search
- Tuning search parameters (top-k, threshold)
- Balancing precision and recall
- Implementing efficient filtering
- Handling large result sets

**Success Criteria:**
- Search returns relevant results
- Results ranked by relevance
- Search completes in < 2 seconds
- Filters work correctly
- Handles 1000+ documents

**Skills Learned:**
- Vector databases
- Semantic search
- Approximate nearest neighbor
- Search ranking
- Result aggregation

**Code Example:**
```python
# src/search/semantic_search.py
import chromadb
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.collection = self.client.get_collection("paper_chunks")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query: str, top_k: int = 10, filters: dict = None):
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters  # Metadata filtering
        )
        
        # Format results
        return self._format_results(results)
```

---

### Phase 5: GenAI Research Assistant (RAG)

**Objective:** Build AI assistant that answers questions about papers

**Duration:** 5-7 days

**What You'll Learn:**
- Retrieval-Augmented Generation (RAG)
- LLM integration (OpenAI or local)
- Prompt engineering
- Citation extraction
- Conversation management

**Tasks:**
1. Set up LLM (OpenAI API or Ollama)
2. Implement retrieval component
3. Build prompt templates
4. Implement answer generation
5. Extract and format citations
6. Add conversation context
7. Implement summarization
8. Write property tests for RAG
9. Test with various questions

**Inputs:** User question, conversation history

**Outputs:**
- AI-generated answer
- Citations to source papers
- Confidence indication

**Key Challenges:**
- Prompt engineering for accuracy
- Handling context window limits
- Extracting citations reliably
- Managing conversation state
- Avoiding hallucinations

**Success Criteria:**
- Answers are grounded in documents
- Citations are accurate
- Handles follow-up questions
- Acknowledges when info not available
- Generates useful summaries

**Skills Learned:**
- RAG architecture
- LLM integration
- Prompt engineering
- Citation extraction
- Conversation management

**Code Example:**
```python
# src/rag/research_assistant.py
from openai import OpenAI

class ResearchAssistant:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.client = OpenAI()
        self.conversation_history = []
    
    def ask(self, question: str) -> dict:
        # Retrieve relevant chunks
        results = self.search_engine.search(question, top_k=5)
        
        # Build context
        context = self._build_context(results)
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate answer
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant..."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Extract citations
        citations = self._extract_citations(answer, results)
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": results
        }
```

---

### Phase 6: Orchestration and Integration (Optional)

**Objective:** Integrate all components with LangGraph orchestration

**Duration:** 3-5 days

**What You'll Learn:**
- Workflow orchestration
- State management
- Async processing
- Error recovery
- System integration

**Tasks:**
1. Design LangGraph workflow
2. Implement state management
3. Add retry logic
4. Implement error handling
5. Add progress tracking
6. Test end-to-end workflow
7. Add monitoring and logging
8. Write integration tests

**Inputs:** Uploaded PDF

**Outputs:**
- Fully processed document
- Status updates
- Error notifications

**Key Challenges:**
- Coordinating multiple components
- Handling partial failures
- Managing state across steps
- Implementing retries correctly
- Monitoring progress

**Success Criteria:**
- Documents process end-to-end
- Failures are handled gracefully
- Progress is tracked
- Retries work correctly
- System is observable

**Skills Learned:**
- Workflow orchestration
- State management
- Error recovery
- System integration
- Observability

---

### Phase 7: API and Interface (Optional)

**Objective:** Build REST API or simple UI

**Duration:** 3-5 days

**What You'll Learn:**
- API design
- FastAPI framework
- Request/response handling
- Authentication (optional)
- Frontend integration (optional)

**Tasks:**
1. Design API endpoints
2. Implement FastAPI application
3. Add upload endpoint
4. Add search endpoint
5. Add chat endpoint
6. Add documentation (Swagger)
7. Test API with Postman
8. (Optional) Build simple web UI

**Inputs:** HTTP requests

**Outputs:**
- REST API
- API documentation
- (Optional) Web interface

**Key Challenges:**
- API design
- Async request handling
- File upload handling
- Error responses
- CORS (if building UI)

**Success Criteria:**
- API endpoints work correctly
- Documentation is clear
- Handles errors gracefully
- (Optional) UI is functional

**Skills Learned:**
- API development
- FastAPI
- HTTP protocols
- API documentation
- (Optional) Frontend basics

---

### Recommended Learning Path

**For Beginners:**
1. Start with Phase 0 (setup)
2. Complete Phase 1 (parsing) - builds confidence
3. Move to Phase 2 (NLP) - introduces ML
4. Complete Phase 4 (search) - practical application
5. Finish with Phase 5 (RAG) - impressive capstone

**For Intermediate:**
1. Phases 0-2 quickly (familiar territory)
2. Focus on Phase 3 (graphs) - new skill
3. Phases 4-5 (search + RAG) - core value
4. Add Phase 6 (orchestration) - production thinking

**For Advanced:**
1. Complete all phases
2. Add Phase 7 (API)
3. Extend with advanced features:
   - Citation network analysis
   - Research trend detection
   - Multi-modal understanding (figures, tables)
   - Collaborative features

### Time Estimates

**Minimum Viable Product (MVP):**
- Phases 0, 1, 2, 4, 5
- Total: 3-4 weeks part-time
- Demonstrates core functionality

**Complete System:**
- All phases
- Total: 6-8 weeks part-time
- Production-grade implementation

**Portfolio-Ready:**
- Complete system + documentation + tests
- Total: 8-10 weeks part-time
- Interview-ready project

### What Makes This Project Valuable

**For Companies:**
- Solves real problem (knowledge management)
- Demonstrates end-to-end thinking
- Shows understanding of multiple AI techniques
- Production-grade architecture

**For Interviews:**
- Lots to discuss (architecture, trade-offs, challenges)
- Demonstrates breadth (NLP, graphs, search, GenAI)
- Shows depth (property-based testing, error handling)
- Proves you can build, not just train models

**For Learning:**
- Covers full ML engineering stack
- Hands-on with modern tools
- Real-world challenges
- Portfolio piece that stands out
