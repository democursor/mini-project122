# Architecture Documentation

## System Architecture Overview

The Research Literature Knowledge Graph System follows a modern, layered architecture with clear separation of concerns. The system is divided into three main tiers: **Frontend**, **Backend**, and **Data Storage**.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                               │
│                         (React + Vite)                               │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Dashboard   │  │   Upload     │  │  Documents   │             │
│  │    Page      │  │    Page      │  │    Page      │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Search     │  │    Graph     │  │     Chat     │             │
│  │    Page      │  │    Page      │  │     Page     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  └─────────────────── API Client (Axios) ────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP/REST
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND LAYER                                │
│                         (FastAPI)                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    API ENDPOINTS                              │  │
│  │                                                               │  │
│  │  /api/documents/*  - Document management                     │  │
│  │  /api/search/*     - Semantic search                         │  │
│  │  /api/graph/*      - Knowledge graph queries                 │  │
│  │  /api/chat/*       - AI assistant (RAG)                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SERVICE LAYER                              │  │
│  │                                                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │  │
│  │  │  Document    │  │    Search    │  │     Chat     │      │  │
│  │  │   Service    │  │   Service    │  │   Service    │      │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │  │
│  │                                                               │  │
│  │  ┌──────────────┐  ┌──────────────┐                         │  │
│  │  │    Graph     │  │   Document   │                         │  │
│  │  │   Service    │  │  Processor   │                         │  │
│  │  └──────────────┘  └──────────────┘                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              DATA PROCESSING PIPELINE                         │  │
│  │                   (LangGraph Workflow)                        │  │
│  │                                                               │  │
│  │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐         │  │
│  │  │  PDF   │ → │ Chunk  │ → │Extract │ → │ Build  │         │  │
│  │  │ Parse  │   │  Text  │   │Concepts│   │ Graph  │         │  │
│  │  └────────┘   └────────┘   └────────┘   └────────┘         │  │
│  │                                              │                │  │
│  │                                              ▼                │  │
│  │                                         ┌────────┐            │  │
│  │                                         │Generate│            │  │
│  │                                         │Embeddin│            │  │
│  │                                         └────────┘            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  CORE COMPONENTS                              │  │
│  │                                                               │  │
│  │  • PDF Validator & Storage                                   │  │
│  │  • PDF Parser (PyMuPDF)                                      │  │
│  │  • Semantic Chunker                                          │  │
│  │  • Concept Extractor (spaCy + KeyBERT)                       │  │
│  │  • Knowledge Graph Builder                                   │  │
│  │  • Embedding Generator (Sentence Transformers)               │  │
│  │  • RAG Retriever & LLM Client                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│     Neo4j        │  │    ChromaDB      │  │   File System    │
│  Graph Database  │  │  Vector Database │  │   PDF Storage    │
│                  │  │                  │  │                  │
│  • Papers        │  │  • Embeddings    │  │  • Original PDFs │
│  • Authors       │  │  • Chunks        │  │  • Metadata JSON │
│  • Concepts      │  │  • Metadata      │  │  • Parsed Data   │
│  • Relationships │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Layer Descriptions

### 1. Frontend Layer (Presentation)

**Technology**: React 18 + Vite + Tailwind CSS

**Responsibilities**:
- User interface rendering
- User interaction handling
- API communication
- State management
- Client-side routing

**Key Components**:
- **Dashboard**: System overview and statistics
- **Upload**: Document upload interface
- **Documents**: Document library management
- **Search**: Semantic search interface
- **Knowledge Graph**: Graph visualization
- **Chat**: AI assistant interface

**Communication**: REST API calls via Axios client

### 2. Backend Layer (Application)

**Technology**: FastAPI + Python 3.10+

**Responsibilities**:
- Request handling and validation
- Business logic execution
- Data processing orchestration
- Authentication and authorization (future)
- Error handling and logging

#### 2.1 API Endpoints

**Documents API** (`/api/documents/*`):
- `POST /upload` - Upload new document
- `GET /` - List all documents
- `GET /{id}` - Get document details
- `DELETE /{id}` - Delete document

**Search API** (`/api/search/*`):
- `POST /` - Semantic search
- `GET /similar/{id}` - Find similar documents

**Graph API** (`/api/graph/*`):
- `GET /stats` - Graph statistics
- `GET /papers` - List papers
- `GET /papers/{id}/related` - Related papers
- `GET /concepts` - List concepts
- `POST /concepts/search` - Search concepts

**Chat API** (`/api/chat/*`):
- `POST /` - Ask question (RAG)

#### 2.2 Service Layer

**DocumentService**:
- Handles document upload and validation
- Orchestrates processing pipeline
- Manages document metadata
- Coordinates deletion across all storage systems

**SearchService**:
- Performs semantic search
- Finds similar documents
- Ranks and filters results

**ChatService**:
- Implements RAG (Retrieval-Augmented Generation)
- Retrieves relevant context
- Generates AI responses
- Manages conversation history

**GraphService**:
- Queries knowledge graph
- Retrieves graph statistics
- Finds related entities
- Performs graph traversals

#### 2.3 Data Processing Pipeline

**Orchestration**: LangGraph state machine

**Pipeline Stages**:

1. **Parse**: Extract text and metadata from PDF
2. **Chunk**: Split document into semantic chunks
3. **Extract**: Identify entities and concepts
4. **Build Graph**: Create knowledge graph nodes and relationships
5. **Generate Embeddings**: Create vector representations
6. **Store**: Persist to databases

**Error Handling**: Each stage can fail independently with proper error propagation

### 3. Data Storage Layer

#### 3.1 Neo4j (Knowledge Graph)

**Purpose**: Store structured relationships between entities

**Schema**:
- **Nodes**: Paper, Author, Concept, Entity
- **Relationships**: AUTHORED_BY, MENTIONS, CITES, RELATED_TO

**Queries**: Cypher query language

**Use Cases**:
- Find related papers
- Explore concept relationships
- Author collaboration networks
- Citation analysis

#### 3.2 ChromaDB (Vector Database)

**Purpose**: Store and search vector embeddings

**Data**:
- Document chunk embeddings
- Metadata (document_id, chunk_id, section)
- Original text

**Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)

**Use Cases**:
- Semantic similarity search
- Document retrieval for RAG
- Find similar documents

#### 3.3 File System

**Purpose**: Store original files and metadata

**Structure**:
```
data/
├── pdfs/
│   └── YYYY/
│       └── MM/
│           └── doc_xxxxx.pdf
├── parsed/
│   └── doc_xxxxx.json
├── chroma/
│   └── [ChromaDB files]
└── documents_metadata.json
```

## Component Interactions

### Document Upload Flow

```
User → Frontend → POST /api/documents/upload
                      ↓
                  DocumentService
                      ↓
              ┌───────┴───────┐
              ▼               ▼
         Validate         Store PDF
              ↓               ↓
         Background Task: Process Document
              ↓
         LangGraph Workflow
              ↓
    ┌────────┼────────┐
    ▼        ▼        ▼
  Neo4j  ChromaDB  Metadata
```

### Search Flow

```
User → Frontend → POST /api/search/
                      ↓
                  SearchService
                      ↓
              Query Processor
                      ↓
           Generate Query Embedding
                      ↓
                  ChromaDB
                      ↓
              Similarity Search
                      ↓
              Ranked Results
                      ↓
                  Frontend
```

### Chat (RAG) Flow

```
User → Frontend → POST /api/chat/
                      ↓
                  ChatService
                      ↓
              RAG Retriever
                      ↓
         ┌────────────┴────────────┐
         ▼                         ▼
    Semantic Search           LLM Client
    (ChromaDB)                (Gemini/GPT)
         │                         │
         └──────────┬──────────────┘
                    ▼
            Generate Answer
            with Citations
                    ▼
                Frontend
```

## Design Patterns

### 1. Repository Pattern
Services interact with data storage through repository interfaces, abstracting database details.

### 2. Service Layer Pattern
Business logic is encapsulated in service classes, separating concerns from API endpoints.

### 3. Pipeline Pattern
Document processing follows a sequential pipeline with state management via LangGraph.

### 4. Dependency Injection
Configuration and dependencies are injected into components for testability.

### 5. RESTful API
Standard HTTP methods and resource-based URLs for API design.

## Scalability Considerations

### Current Architecture
- Single-server deployment
- Synchronous processing
- In-memory state

### Future Enhancements
- **Horizontal Scaling**: Load balancer + multiple API servers
- **Async Processing**: Celery/RabbitMQ for background tasks
- **Caching**: Redis for frequently accessed data
- **CDN**: Static asset delivery
- **Database Replication**: Read replicas for Neo4j and ChromaDB
- **Microservices**: Split into document, search, and chat services

## Security Considerations

### Current Implementation
- CORS configuration
- Input validation
- File type validation
- Size limits

### Future Enhancements
- **Authentication**: JWT tokens
- **Authorization**: Role-based access control
- **Rate Limiting**: API throttling
- **Encryption**: TLS/SSL for data in transit
- **Secrets Management**: Vault for API keys
- **Audit Logging**: Track all operations

## Technology Choices

### Why FastAPI?
- High performance (async support)
- Automatic API documentation (OpenAPI/Swagger)
- Type hints and validation (Pydantic)
- Modern Python features
- Easy testing

### Why Neo4j?
- Native graph database
- Powerful query language (Cypher)
- Excellent for relationship queries
- Visualization capabilities
- Scalable graph operations

### Why ChromaDB?
- Lightweight and easy to deploy
- Built for embeddings
- Fast similarity search
- Python-native
- No separate server required

### Why React?
- Component-based architecture
- Large ecosystem
- Excellent developer experience
- Virtual DOM performance
- Strong community support

## Deployment Architecture

### Development
```
localhost:3000 (Frontend - Vite dev server)
localhost:8000 (Backend - Uvicorn)
localhost:7687 (Neo4j)
./data/chroma (ChromaDB)
```

### Production (Recommended)
```
Nginx → React (Static files)
Nginx → FastAPI (Gunicorn + Uvicorn workers)
Neo4j (Dedicated server)
ChromaDB (Persistent volume)
```

## Monitoring and Observability

### Logging
- Structured logging with Python logging module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log files in `data/logs/`

### Metrics (Future)
- API response times
- Document processing duration
- Database query performance
- Error rates
- User activity

### Health Checks
- `/health` endpoint
- Database connectivity checks
- Service status monitoring
