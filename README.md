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

### 🔐 User Authentication
- Secure sign up and sign in with email/password
- JWT-based authentication
- Session management and persistence
- User-scoped data isolation
- Row Level Security (RLS) at database level
- Automatic token refresh

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
- Conversation history support (persisted to database)
- Multi-turn dialogue
- Session management and resumption

### 💻 Interactive Web Interface
- Modern React-based UI with premium design
- User authentication (sign up, sign in, sign out)
- Document management dashboard
- Visual knowledge graph explorer
- Real-time search and chat interfaces
- Responsive design
- User-scoped data isolation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  Dashboard | Upload | Documents | Search | Graph | Chat     │
│                   + Supabase Auth                            │
└─────────────────────────────────────────────────────────────┘
                           │ REST API (JWT Auth)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  API Layer: /documents /search /graph /chat           │ │
│  │  + JWT Authentication Middleware                       │ │
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
        ┌──────────────────┼──────────────────┬──────────────┐
        ▼                  ▼                   ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│   Neo4j      │  │  ChromaDB    │  │  File System │  │  Supabase   │
│  (Knowledge  │  │   (Vector    │  │   (PDF       │  │  (Auth +    │
│    Graph)    │  │  Embeddings) │  │   Storage)   │  │  User Data) │
└──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘
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
- **Supabase** - Authentication and PostgreSQL database
- **Neo4j** - Graph database for knowledge graph storage
- **ChromaDB** - Vector database for semantic search
- **PyMuPDF** - PDF parsing and text extraction
- **spaCy** - Named entity recognition and NLP
- **Sentence Transformers** - Text embedding generation
- **LangGraph** - Workflow orchestration
- **Google Gemini / OpenAI** - Large language models
- **python-jose** - JWT token handling

### Frontend
- **React 18** - Modern UI framework
- **React Router** - Client-side routing
- **Supabase JS Client** - Authentication and database access
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
- Supabase account (free tier available at [supabase.com](https://supabase.com))
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

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_JWT_SECRET=your_supabase_jwt_secret

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

### 2.5. Supabase Setup

#### Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up/sign in
2. Click "New Project"
3. Fill in project details:
   - Name: `research-knowledge-graph` (or your preferred name)
   - Database Password: Choose a strong password
   - Region: Select closest to your location
4. Wait for project to be provisioned (1-2 minutes)

#### Get Supabase Credentials

1. Go to **Project Settings** > **API**
2. Copy the following values:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon/public key** (under "Project API keys")
   - **service_role key** (under "Project API keys" - click "Reveal")
3. Go to **Project Settings** > **API** > **JWT Settings**
4. Copy the **JWT Secret**

#### Run Database Schema

1. In your Supabase dashboard, click **SQL Editor** in the left sidebar
2. Click **New Query**
3. Open the `supabase_schema.sql` file from this repository
4. Copy the entire contents and paste into the SQL editor
5. Click **Run** to execute the schema

This will create:
- `documents` table (user-scoped document metadata)
- `chat_sessions` table (conversation sessions)
- `chat_messages` table (individual messages)
- `search_history` table (user search tracking)
- `profiles` table (extended user information)
- Row Level Security (RLS) policies
- Indexes for performance
- Triggers for auto-updating timestamps

#### Configure Supabase Authentication

1. Go to **Authentication** > **Providers** in your Supabase dashboard
2. Enable **Email** provider (enabled by default)
3. Configure email templates (optional):
   - Go to **Authentication** > **Email Templates**
   - Customize confirmation, reset password, and magic link emails
4. Set **Site URL** for development:
   - Go to **Authentication** > **URL Configuration**
   - Add `http://localhost:3000` to **Site URL**
   - Add `http://localhost:3000/**` to **Redirect URLs**

#### Update Environment Variables

Update your root `.env` file with the Supabase credentials:

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=your_actual_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_actual_service_role_key_here
SUPABASE_JWT_SECRET=your_actual_jwt_secret_here
```

Create `frontend/.env` file with frontend-specific variables:

```env
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=your_actual_anon_key_here
```

**⚠️ Security Note**: Never commit `.env` files to version control. Use `.env.example` as a template.

#### Run Backend

```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Or using the run script
python run_api.py
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

**Note**: Backend takes 15-20 seconds to start due to ML model loading (spaCy, Sentence Transformers).

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

Create `frontend/.env` file (if not already created):

```env
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=your_actual_anon_key_here
```

The API endpoint is configured in `frontend/vite.config.js` proxy settings (defaults to `http://localhost:8000`).

#### Run Frontend

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000` (or `http://localhost:3001` if port 3000 is in use)

## 🚀 Usage

### 0. User Authentication

Before using the application, you need to create an account:

1. Navigate to `http://localhost:3000/login`
2. Click **Sign Up** tab
3. Enter your email and password (minimum 6 characters)
4. Click **Sign Up**
5. Check your email for confirmation (if email confirmation is enabled in Supabase)
6. Sign in with your credentials

**Note**: All features require authentication. Each user has their own isolated data (documents, chat sessions, search history).

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

### Authentication

All API endpoints require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

The frontend automatically handles token management through the Supabase client.

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
GET  /api/chat/sessions
GET  /api/chat/sessions/{session_id}/messages
PATCH /api/chat/sessions/{session_id}
DELETE /api/chat/sessions/{session_id}
```

For detailed API documentation, visit `http://localhost:8000/docs` when the backend is running, or see [docs/api_documentation.md](docs/api_documentation.md).

## 📁 Project Structure

```
research-knowledge-graph/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # React components
│   │   ├── context/         # Auth context
│   │   ├── lib/             # Supabase client
│   │   ├── pages/           # Page components
│   │   ├── App.jsx          # Main app component
│   │   └── main.jsx         # Entry point
│   ├── .env                 # Frontend environment variables
│   ├── package.json
│   └── vite.config.js
│
├── src/                     # Backend source code
│   ├── api/                 # FastAPI application
│   │   ├── routes/          # API endpoints
│   │   ├── models.py        # Pydantic models
│   │   └── main.py          # FastAPI app
│   │
│   ├── auth/                # Authentication module
│   │   ├── dependencies.py  # JWT validation
│   │   └── supabase_db.py   # Database operations
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
├── .env                     # Environment variables (not in git)
├── .env.example             # Environment template
├── .gitignore
├── requirements.txt         # Python dependencies
├── supabase_schema.sql      # Database schema
├── SUPABASE_INTEGRATION_SUMMARY.md  # Integration docs
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

## 🔧 Troubleshooting

### Backend Issues

**Backend takes long to start**
- Normal behavior: ML models (spaCy, Sentence Transformers) take 15-20 seconds to load
- Wait for "Application startup complete" message

**401 Unauthorized errors**
- Verify Supabase credentials in `.env` are correct
- Check that JWT_SECRET matches your Supabase project
- Ensure user is signed in (check browser localStorage for `sb-*-auth-token`)
- Try signing out and signing in again

**Neo4j connection failed**
- Verify Neo4j is running: `docker ps` or check Neo4j Desktop
- Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in `.env`
- System will fallback to ChromaDB-only mode if Neo4j is unavailable

**ChromaDB errors**
- Delete `data/chroma/` directory and restart
- Ensure sufficient disk space

### Frontend Issues

**Login page not loading**
- Check `frontend/.env` has correct VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
- Verify Supabase project is active
- Check browser console for errors

**API calls failing**
- Ensure backend is running on port 8000
- Check vite proxy configuration in `frontend/vite.config.js`
- Verify CORS settings in backend

**Port 3000 already in use**
- Frontend will automatically use port 3001
- Or stop the process using port 3000

### Supabase Issues

**Email confirmation not received**
- Check spam folder
- Disable email confirmation in Supabase: Authentication > Providers > Email > "Enable email confirmations" (toggle off)
- Use magic link instead

**RLS policy errors**
- Verify `supabase_schema.sql` was executed completely
- Check Supabase logs: Project > Logs > Postgres Logs
- Ensure user is authenticated

**Database connection errors**
- Verify SUPABASE_URL and keys are correct
- Check Supabase project status (not paused)
- Verify network/firewall not blocking supabase.co

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
