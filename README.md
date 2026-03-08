# 🔬 Research Literature Knowledge Graph System

An AI-powered platform for processing, analyzing, and exploring academic research papers using knowledge graphs, semantic search, and large language models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.14+-008CC1.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Research Literature Knowledge Graph System transforms static PDF research papers into an interactive, queryable knowledge base. It combines advanced NLP, knowledge graph technology, and semantic search to help researchers efficiently process, analyze, and explore academic literature.

### Problem Statement

Researchers face challenges with:
- **Information overload** from the volume of published research
- **Knowledge fragmentation** across multiple documents
- **Limited searchability** with traditional keyword search
- **Manual analysis** of concepts and relationships
- **Context loss** when exploring related work

### Solution

This system provides:
- ✅ Automated PDF processing and metadata extraction
- ✅ Semantic understanding using NLP models
- ✅ Knowledge graph construction with Neo4j
- ✅ Vector-based semantic search with ChromaDB
- ✅ AI-powered Q&A using RAG (Retrieval-Augmented Generation)
- ✅ Interactive web interface for exploration

## ✨ Features

### 📄 Document Processing
- PDF upload and validation
- Automatic text extraction and parsing
- Metadata extraction (title, authors, year, abstract)
- Section-aware document chunking
- Batch processing support

### 🕸️ Knowledge Graph
- Automatic entity and concept extraction
- Relationship mapping between papers and concepts
- Author and citation network visualization
- Graph-based exploration and discovery
- Cypher query support

### 🔍 Semantic Search
- Vector-based similarity search
- Context-aware query understanding
- Ranked results by relevance
- Find similar documents functionality
- Hybrid search capabilities

### 🤖 AI Research Assistant
- Natural language question answering
- Context-aware responses using RAG
- Source citation and verification
- Conversation history support
- Multi-turn dialogue

### 💻 Interactive Web Interface
- Modern React-based UI
- Document management dashboard
- Visual knowledge graph explorer
- Real-time search and chat interfaces
- Responsive design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  Dashboard | Upload | Documents | Search | Graph | Chat     │
└─────────────────────────────────────────────────────────────┘
                           │ REST API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  API Layer: /documents /search /graph /chat           │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Service Layer: DocumentService, SearchService, etc.  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Processing Pipeline (LangGraph)                       │ │
│  │  Parse → Chunk → Extract → Graph → Embed              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Neo4j      │  │  ChromaDB    │  │  File System │
│  (Knowledge  │  │   (Vector    │  │   (PDF       │
│    Graph)    │  │  Embeddings) │  │   Storage)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## 📊 Data Flow Diagram

### 1. Document Upload & Processing Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │ 1. Upload PDF
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • File validation (size, type)                             │
│  • Upload progress tracking                                 │
└──────┬──────────────────────────────────────────────────────┘
       │ 2. POST /api/documents/upload
       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend - API Layer                     │
│  • Request validation                                        │
│  • Multipart form handling                                  │
└──────┬──────────────────────────────────────────────────────┘
       │ 3. Process document
       ▼
┌─────────────────────────────────────────────────────────────┐
│            DocumentService (Service Layer)                   │
│  • PDF validation (format, integrity)                       │
│  • Generate document_id                                     │
│  • Store PDF to file system                                 │
└──────┬──────────────────────────────────────────────────────┘
       │ 4. Trigger processing pipeline
       ▼
┌─────────────────────────────────────────────────────────────┐
│         LangGraph Workflow (Orchestration)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ STAGE 1: PDF Parsing                                 │  │
│  │  • Extract text using PyMuPDF                        │  │
│  │  • Extract metadata (title, authors, year)           │  │
│  │  • Identify sections and structure                   │  │
│  │  Output: ParsedDocument                              │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ STAGE 2: Semantic Chunking                           │  │
│  │  • Split text into semantic chunks                   │  │
│  │  • Detect section boundaries                         │  │
│  │  • Maintain context windows                          │  │
│  │  Output: List[TextChunk]                             │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ STAGE 3: Concept Extraction                          │  │
│  │  • Named Entity Recognition (spaCy)                  │  │
│  │  • Keyphrase extraction (KeyBERT)                    │  │
│  │  • Identify concepts and relationships               │  │
│  │  Output: List[ExtractedConcept]                      │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ STAGE 4: Knowledge Graph Construction                │  │
│  │  • Create Paper node                                 │  │
│  │  • Create Author nodes                               │  │
│  │  • Create Concept/Entity nodes                       │  │
│  │  • Create relationships (AUTHORED_BY, MENTIONS)      │  │
│  │  Output: Graph nodes & relationships                 │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ STAGE 5: Vector Embedding Generation                 │  │
│  │  • Generate embeddings (Sentence Transformers)       │  │
│  │  • Create 384-dim vectors for each chunk             │  │
│  │  • Store embeddings with metadata                    │  │
│  │  Output: List[ChunkEmbedding]                        │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Neo4j     │ │  ChromaDB   │ │ File System │
│             │ │             │ │             │
│ • Paper     │ │ • Chunk     │ │ • PDF file  │
│ • Author    │ │   embeddings│ │ • Metadata  │
│ • Concept   │ │ • Metadata  │ │   JSON      │
│ • Relations │ │ • Text      │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 2. Semantic Search Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │ 1. Enter search query
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Search input component                                   │
│  • Display loading state                                    │
└──────┬──────────────────────────────────────────────────────┘
       │ 2. POST /api/search/
       │    { "query": "machine learning", "top_k": 10 }
       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend - API Layer                     │
│  • Validate query parameters                                │
│  • Parse request body                                       │
└──────┬──────────────────────────────────────────────────────┘
       │ 3. Execute search
       ▼
┌─────────────────────────────────────────────────────────────┐
│              SearchService (Service Layer)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 1: Query Processing                             │  │
│  │  • Clean and normalize query                         │  │
│  │  • Extract key terms                                 │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 2: Generate Query Embedding                     │  │
│  │  • Use Sentence Transformer model                    │  │
│  │  • Create 384-dim vector                             │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 3: Vector Similarity Search                     │  │
│  │  • Query ChromaDB with embedding                     │  │
│  │  • Cosine similarity matching                        │  │
│  │  • Retrieve top_k results                            │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 4: Result Ranking & Filtering                   │  │
│  │  • Sort by relevance score                           │  │
│  │  • Group by document                                 │  │
│  │  • Add metadata                                      │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  ChromaDB   │
              │             │
              │ • Similarity│
              │   search    │
              │ • Return    │
              │   matches   │
              └─────────────┘
                    │
                    │ 4. Return results
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Display search results                                   │
│  • Show relevance scores                                    │
│  • Enable result filtering                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3. AI Chat (RAG) Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │ 1. Ask question
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Chat interface                                           │
│  • Message history                                          │
└──────┬──────────────────────────────────────────────────────┘
       │ 2. POST /api/chat/
       │    { "question": "What is RAG?", "session_id": "..." }
       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend - API Layer                     │
│  • Validate question                                        │
│  • Check session context                                    │
└──────┬──────────────────────────────────────────────────────┘
       │ 3. Process question
       ▼
┌─────────────────────────────────────────────────────────────┐
│               ChatService (Service Layer)                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 1: Question Understanding                       │  │
│  │  • Parse user question                               │  │
│  │  • Extract intent                                    │  │
│  │  • Consider conversation history                     │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 2: Context Retrieval (RAG Retriever)            │  │
│  │  • Generate question embedding                       │  │
│  │  • Search ChromaDB for relevant chunks               │  │
│  │  • Retrieve top 5-10 most relevant passages          │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 3: Context Preparation                          │  │
│  │  • Combine retrieved chunks                          │  │
│  │  • Format context with metadata                      │  │
│  │  • Add source citations                              │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 4: Prompt Construction                          │  │
│  │  • Build system prompt                               │  │
│  │  • Add retrieved context                             │  │
│  │  • Add user question                                 │  │
│  │  • Include instructions for citations                │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 5: LLM Generation                               │  │
│  │  • Send prompt to LLM (Gemini/GPT)                   │  │
│  │  • Stream or wait for response                       │  │
│  │  • Parse LLM output                                  │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 6: Response Post-Processing                     │  │
│  │  • Extract citations                                 │  │
│  │  • Format answer                                     │  │
│  │  • Add source references                             │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  ChromaDB   │ │ LLM Service │ │   Neo4j     │
│             │ │  (Gemini/   │ │  (Optional) │
│ • Retrieve  │ │   OpenAI)   │ │             │
│   context   │ │             │ │ • Graph     │
│ • Get       │ │ • Generate  │ │   context   │
│   chunks    │ │   answer    │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
                    │
                    │ 4. Return answer with citations
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Display AI answer                                        │
│  • Show source citations                                    │
│  • Enable follow-up questions                               │
└─────────────────────────────────────────────────────────────┘
```

### 4. Knowledge Graph Query Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │ 1. Explore graph / Search concepts
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Graph visualization component                            │
│  • Concept search interface                                 │
└──────┬──────────────────────────────────────────────────────┘
       │ 2. GET /api/graph/papers/{id}/related
       │    or POST /api/graph/concepts/search
       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend - API Layer                     │
│  • Parse graph query parameters                             │
│  • Validate node IDs                                        │
└──────┬──────────────────────────────────────────────────────┘
       │ 3. Execute graph query
       ▼
┌─────────────────────────────────────────────────────────────┐
│              GraphService (Service Layer)                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 1: Query Construction                           │  │
│  │  • Build Cypher query                                │  │
│  │  • Add filters and parameters                        │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 2: Execute on Neo4j                             │  │
│  │  • Run Cypher query                                  │  │
│  │  • Traverse relationships                            │  │
│  │  • Collect nodes and edges                           │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                          │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ Step 3: Result Formatting                            │  │
│  │  • Convert Neo4j records to JSON                     │  │
│  │  • Format for visualization                          │  │
│  │  • Add metadata                                      │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │
                    ▼
              ┌─────────────┐
              │   Neo4j     │
              │             │
              │ • Execute   │
              │   Cypher    │
              │ • Return    │
              │   graph data│
              └─────────────┘
                    │
                    │ 4. Return graph data
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  • Render graph visualization                               │
│  • Show node details                                        │
│  • Enable interactive exploration                           │
└─────────────────────────────────────────────────────────────┘
```

### Data Storage Summary

| Database | Data Type | Purpose | Access Pattern |
|----------|-----------|---------|----------------|
| **Neo4j** | Graph | Papers, Authors, Concepts, Relationships | Cypher queries for graph traversal |
| **ChromaDB** | Vectors | Text embeddings (384-dim), Chunk metadata | Similarity search by cosine distance |
| **File System** | Files | Original PDFs, Parsed JSON, Metadata | Direct file I/O |

For more detailed workflow documentation, see [docs/workflow.md](docs/workflow.md).

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **Neo4j** - Graph database for knowledge graph storage
- **ChromaDB** - Vector database for semantic search
- **PyMuPDF** - PDF parsing and text extraction
- **spaCy** - Named entity recognition and NLP
- **Sentence Transformers** - Text embedding generation
- **LangGraph** - Workflow orchestration
- **Google Gemini / OpenAI** - Large language models

### Frontend
- **React 18** - Modern UI framework
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Data visualization
- **Lucide React** - Icon library

### Data Processing
- **Transformers** - Pre-trained NLP models
- **KeyBERT** - Keyphrase extraction
- **NLTK** - Text processing utilities
- **scikit-learn** - Machine learning utilities

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Neo4j 5.14 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/research-knowledge-graph.git
cd research-knowledge-graph
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

#### Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# LLM Configuration (choose one)
GOOGLE_API_KEY=your_google_api_key
# OR
OPENAI_API_KEY=your_openai_api_key

# Application Settings
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
```

#### Start Neo4j

```bash
# Using Neo4j Desktop or Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

#### Run Backend

```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Or using the run script
python run_api.py
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### 3. Frontend Setup

#### Navigate to Frontend Directory

```bash
cd frontend
```

#### Install Dependencies

```bash
npm install
```

#### Configure API Endpoint

Update `frontend/src/api/client.js` if needed:

```javascript
const API_BASE_URL = 'http://localhost:8000/api';
```

#### Run Frontend

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 🚀 Usage

### 1. Upload Documents

1. Navigate to the **Upload** page
2. Select a PDF research paper
3. Click **Upload** and wait for processing
4. View processing status and results

### 2. Search Documents

1. Go to the **Search** page
2. Enter a natural language query
3. View semantically similar documents
4. Click on results to view details

### 3. Explore Knowledge Graph

1. Visit the **Knowledge Graph** page
2. Browse papers, authors, and concepts
3. Explore relationships and connections
4. Filter and search within the graph

### 4. Ask Questions

1. Open the **Chat** page
2. Ask questions about your document collection
3. Receive AI-generated answers with citations
4. Follow up with additional questions

### 5. Manage Documents

1. Access the **Documents** page
2. View all uploaded documents
3. Search and filter documents
4. Delete documents as needed

## 📚 API Documentation

### Documents API

```http
POST /api/documents/upload
GET  /api/documents/
GET  /api/documents/{document_id}
DELETE /api/documents/{document_id}
```

### Search API

```http
POST /api/search/
GET  /api/search/similar/{document_id}
```

### Knowledge Graph API

```http
GET  /api/graph/stats
GET  /api/graph/papers
GET  /api/graph/papers/{paper_id}/related
GET  /api/graph/concepts
POST /api/graph/concepts/search
```

### Chat API

```http
POST /api/chat/
```

For detailed API documentation, visit `http://localhost:8000/docs` when the backend is running, or see [docs/api_documentation.md](docs/api_documentation.md).

## 📁 Project Structure

```
research-knowledge-graph/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── App.jsx          # Main app component
│   │   └── main.jsx         # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── src/                     # Backend source code
│   ├── api/                 # FastAPI application
│   │   ├── routes/          # API endpoints
│   │   ├── models.py        # Pydantic models
│   │   └── main.py          # FastAPI app
│   │
│   ├── services/            # Business logic layer
│   │   ├── document_service.py
│   │   ├── search_service.py
│   │   ├── chat_service.py
│   │   └── graph_service.py
│   │
│   ├── ingestion/           # Document upload and validation
│   ├── parsing/             # PDF parsing
│   ├── chunking/            # Text chunking
│   ├── extraction/          # Concept extraction
│   ├── graph/               # Knowledge graph builder
│   ├── vector/              # Vector embeddings
│   ├── rag/                 # RAG implementation
│   ├── orchestration/       # LangGraph workflows
│   └── utils/               # Utilities and config
│
├── docs/                    # Documentation
│   ├── system_overview.md
│   ├── architecture.md
│   ├── workflow.md
│   ├── database_design.md
│   ├── api_documentation.md
│   ├── implementation_details.md
│   ├── deployment_guide.md
│   └── future_improvements.md
│
├── data/                    # Data storage
│   ├── pdfs/               # Uploaded PDFs
│   ├── parsed/             # Parsed data
│   ├── chroma/             # ChromaDB storage
│   └── logs/               # Application logs
│
├── config/                  # Configuration files
│   └── default.yaml
│
├── .env                     # Environment variables
├── .gitignore
├── requirements.txt         # Python dependencies
├── README.md
└── main.py                  # CLI entry point
```

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[System Overview](docs/system_overview.md)** - Project purpose, features, and benefits
- **[Architecture](docs/architecture.md)** - System architecture and component interactions
- **[Workflow](docs/workflow.md)** - Data processing pipeline details
- **[Database Design](docs/database_design.md)** - Neo4j schema and ChromaDB structure
- **[API Documentation](docs/api_documentation.md)** - Complete API reference
- **[Implementation Details](docs/implementation_details.md)** - Technical decisions and rationale
- **[Deployment Guide](docs/deployment_guide.md)** - Production deployment instructions
- **[Future Improvements](docs/future_improvements.md)** - Planned enhancements

## 🧪 Testing

### Run Backend Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_phase1.py

# Run with coverage
pytest --cov=src tests/
```

### Run Frontend Tests

```bash
cd frontend
npm test
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide (Python)
- Tests are included for new features
- Documentation is updated
- Commit messages are descriptive

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** for NLP capabilities
- **Sentence Transformers** for embedding models
- **Neo4j** for graph database technology
- **ChromaDB** for vector storage
- **FastAPI** for the excellent web framework
- **React** community for frontend tools

## 📧 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/research-knowledge-graph/issues)
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/yourusername/research-knowledge-graph](https://github.com/yourusername/research-knowledge-graph)

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Built with ❤️ for the research community**
