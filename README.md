# Research Literature Platform

An AI-powered platform for processing, analyzing, and querying research papers with intelligent document understanding, knowledge graph construction, and conversational AI assistance.

## 🌟 Features

### Core Capabilities
- **PDF Processing**: Automated ingestion, validation, and text extraction
- **Semantic Analysis**: Intelligent chunking and concept extraction using NLP
- **Knowledge Graph**: Neo4j-powered relationship mapping between papers and concepts
- **Vector Search**: ChromaDB-based semantic search across document chunks
- **AI Assistant**: RAG-powered conversational interface with citation support
- **Modern UI**: React-based frontend with real-time chat and visualization

### Key Highlights
- Upload and process research papers automatically
- Ask questions and get AI-generated answers with citations
- Explore knowledge graphs to discover concept relationships
- Semantic search to find relevant papers by meaning, not just keywords
- Chat history persistence for continuous research sessions

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
                                    ┌─────────────────────────────────────┐
                                    │      External AI Services           │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  Google Gemini 2.5 Flash     │   │
                                    │  │  (LLM for RAG & Insights)    │   │
                                    │  └──────────────────────────────┘   │
                                    └─────────────────────────────────────┘
                                                    ↕
┌──────────────┐                    ┌─────────────────────────────────────┐
│              │      Requests      │                                     │
│    Users     │ ──────────────────→│      Backend Services               │
│              │                    │  ┌──────────────────────────────┐   │
└──────────────┘                    │  │  Document Service            │   │
       ↓                            │  │  - Upload & Validation       │   │
   Requests                         │  │  - Processing Orchestration  │   │
       ↓                            │  └──────────────────────────────┘   │
┌──────────────┐                    │  ┌──────────────────────────────┐   │
│              │      Requests      │  │  Search Service              │   │
│ Web/Mobile   │ ──────────────────→│  │  - Semantic Search           │   │
│   Client     │                    │  │  - Vector Similarity         │   │
│  (React +    │←─────────────────  │  └──────────────────────────────┘   │
│   Vite)      │     Responses      │  ┌──────────────────────────────┐   │
│              │                    │  │  Chat Service                │   │
└──────────────┘                    │  │  - RAG Pipeline              │   │
                                    │  │  - AI Assistant              │   │
                                    │  └──────────────────────────────┘   │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  Graph Service               │   │
                                    │  │  - Knowledge Graph Queries   │   │
                                    │  │  - Relationship Mapping      │   │
                                    │  └──────────────────────────────┘   │
                                    └─────────────────────────────────────┘
                                                    ↕
                                    ┌─────────────────────────────────────┐
                                    │      AI/ML Processing Layer         │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  spaCy (en_core_web_sm)      │   │
                                    │  │  - Named Entity Recognition  │   │
                                    │  └──────────────────────────────┘   │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  all-MiniLM-L6-v2            │   │
                                    │  │  - Sentence Embeddings       │   │
                                    │  │  - 384-dimensional vectors   │   │
                                    │  └──────────────────────────────┘   │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  KeyBERT                     │   │
                                    │  │  - Keyword Extraction        │   │
                                    │  └──────────────────────────────┘   │
                                    └─────────────────────────────────────┘
                                                    ↕
                                    ┌─────────────────────────────────────┐
                                    │      Data Storage Layer             │
                                    │                                     │
                                    │  ┌──────────────┐  ┌────────────┐  │
                                    │  │  ChromaDB    │  │  Neo4j     │  │
                                    │  │  (Vector DB) │  │  (Graph DB)│  │
                                    │  │  - Embeddings│  │  - Concepts│  │
                                    │  │  - HNSW Index│  │  - Relations│ │
                                    │  └──────────────┘  └────────────┘  │
                                    │                                     │
                                    │  ┌──────────────────────────────┐   │
                                    │  │  Local File Storage          │   │
                                    │  │  - PDF Documents             │   │
                                    │  │  - Parsed Text               │   │
                                    │  │  - Metadata (JSON)           │   │
                                    │  └──────────────────────────────┘   │
                                    └─────────────────────────────────────┘
```

### Technology Stack

#### Frontend Layer
- **React 18**: Modern UI framework with hooks
- **Vite**: Lightning-fast build tool and dev server
- **TailwindCSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icon library
- **Recharts**: Composable charting library

#### Backend Services (FastAPI)
- **Document Service**: PDF upload, validation, processing orchestration
- **Search Service**: Semantic search with vector similarity
- **Chat Service**: RAG pipeline with AI assistant
- **Graph Service**: Knowledge graph queries and relationship mapping

#### AI/ML Processing
- **spaCy (en_core_web_sm)**: Named Entity Recognition (12.8 MB)
- **all-MiniLM-L6-v2**: Sentence embeddings (90 MB, 384 dimensions)
- **KeyBERT**: Keyword and keyphrase extraction
- **Google Gemini 2.5 Flash**: LLM for insights and chat responses

#### Data Storage
- **ChromaDB**: Vector database for embeddings with HNSW indexing
- **Neo4j**: Graph database for concepts and relationships
- **Local File System**: PDF storage, parsed text, JSON metadata

### Processing Pipelines

#### Document Processing Pipeline
```
PDF Upload → Validation → Text Extraction (PyMuPDF) → 
Semantic Chunking → Concept Extraction (spaCy + KeyBERT) → 
Embedding Generation (all-MiniLM-L6-v2) → 
Storage (ChromaDB + Neo4j + JSON)
```

#### Search & RAG Pipeline
```
User Query → Query Embedding → Vector Search (ChromaDB) → 
Retrieve Top-K Chunks → Context Augmentation → 
LLM Generation (Gemini) → Response with Citations
```

### Key Architecture Decisions

1. **Microservices Pattern**: Separate services for documents, search, chat, and graph operations
2. **Vector + Graph Hybrid**: ChromaDB for semantic search, Neo4j for relationship mapping
3. **Lightweight Models**: Using efficient models (103MB total) for fast processing
4. **RAG Architecture**: Retrieval-Augmented Generation for accurate, cited responses
5. **Local-First**: PDF storage and processing on local filesystem for privacy

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Neo4j Database
- Google Gemini API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/democursor/mini-project122.git
cd mini-project122
```

2. **Setup Python environment**
```bash
python -m venv vnv
vnv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Configure environment**
Create `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=miniproject
```

4. **Setup Neo4j**
- Install Neo4j Desktop
- Create database named "miniproject"
- Start the database

5. **Install frontend dependencies**
```bash
cd frontend
npm install
```

### Running the Application

1. **Start the backend server**
```bash
python run_api.py
```
Backend runs on `http://localhost:8000`

2. **Start the frontend** (in a new terminal)
```bash
cd frontend
npm run dev
```
Frontend runs on `http://localhost:3000`

3. **Access the application**
Open your browser to `http://localhost:3000`

## 📖 Usage

### Upload Documents
1. Navigate to the Upload page
2. Select a PDF research paper
3. System automatically processes and extracts information

### Search Papers
1. Go to Search page
2. Enter your query
3. Get semantically relevant results with similarity scores

### Chat with AI
1. Open Chat page
2. Ask questions about your uploaded papers
3. Receive AI-generated answers with citations
4. Chat history is automatically saved

### Explore Knowledge Graph
1. Visit Knowledge Graph page
2. View statistics on papers, concepts, and relationships
3. Explore connections between research concepts

## 🎯 System Components

### Document Processing Pipeline
```
PDF Upload → Validation → Parsing → Chunking → NLP Extraction → Storage
```

1. **Ingestion**: Validates and stores PDFs with metadata
2. **Parsing**: Extracts text, title, authors, and abstract
3. **Chunking**: Creates semantic chunks using sentence transformers
4. **Extraction**: Identifies entities (NER) and keyphrases (KeyBERT)
5. **Graph Building**: Constructs knowledge graph in Neo4j
6. **Vector Storage**: Stores embeddings in ChromaDB

### AI Assistant (RAG)
```
User Query → Query Expansion → Vector Search → Context Retrieval → LLM Generation → Response with Citations
```

- Expands queries with related terms for better retrieval
- Retrieves top-10 relevant chunks from vector database
- Uses Google Gemini for intelligent answer generation
- Extracts and displays citations from source documents

## 📁 Project Structure

```
mini-project122/
├── src/
│   ├── api/              # FastAPI routes and models
│   ├── ingestion/        # PDF upload and validation
│   ├── parsing/          # Text extraction
│   ├── chunking/         # Semantic chunking
│   ├── extraction/       # NER and keyphrase extraction
│   ├── graph/            # Neo4j knowledge graph
│   ├── vector/           # ChromaDB vector storage
│   ├── rag/              # RAG and AI assistant
│   ├── services/         # Business logic layer
│   ├── orchestration/    # Workflow management
│   └── utils/            # Configuration and logging
├── frontend/
│   ├── src/
│   │   ├── pages/        # React pages
│   │   ├── components/   # Reusable components
│   │   └── api/          # API client
│   └── package.json
├── config/               # Configuration files
├── data/                 # Storage (PDFs, parsed data)
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload and process PDF
- `GET /api/documents` - List all documents
- `GET /api/documents/{doc_id}` - Get document details

### Search
- `POST /api/search` - Semantic search across documents

### Chat
- `POST /api/chat` - Ask questions and get AI responses

### Knowledge Graph
- `GET /api/graph/stats` - Get graph statistics
- `GET /api/graph/concepts` - List all concepts
- `GET /api/graph/papers/{paper_id}/concepts` - Get paper concepts

## 🎨 Features in Detail

### Intelligent Query Expansion
The system automatically expands your queries with related terms:
- "COVID-19" → includes "pandemic", "coronavirus", "SARS-CoV-2"
- "machine learning" → includes "ML", "artificial intelligence", "deep learning"

### Citation Extraction
AI responses include citations with:
- Document title
- Relevant excerpt
- Page numbers (when available)

### Knowledge Graph Visualization
- View total papers and concepts
- Explore relationships between concepts
- Discover research connections

### Chat History Persistence
- Conversations saved in browser localStorage
- Resume research sessions anytime
- Clear history option available

## 🧪 Testing

Run backend tests:
```bash
python test_phase1.py  # PDF ingestion
python test_phase2.py  # Chunking and extraction
python test_phase3.py  # Knowledge graph
python test_phase4.py  # Vector search
python test_phase5.py  # RAG assistant
python test_phase6.py  # Full integration
```

Test API endpoints:
```bash
python test_api.py
```

## 🛠️ Configuration

Edit `config/default.yaml` to customize:

```yaml
storage:
  pdf_dir: "data/pdfs"
  parsed_dir: "data/parsed"

validation:
  max_size_mb: 50
  allowed_formats: [".pdf"]

chunking:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  max_chunk_size: 512

rag:
  model: "gemini-1.5-flash"
  temperature: 0.4
  max_tokens: 4000
  top_k: 10
```

## 🚨 Troubleshooting

### Neo4j Connection Issues
- Verify Neo4j is running
- Check credentials in `.env`
- Ensure database "miniproject" exists

### Frontend Not Loading
- Check if backend is running on port 8000
- Verify CORS settings in FastAPI
- Clear browser cache

### PDF Upload Fails
- Check file size (max 50MB)
- Ensure PDF is not corrupted
- Verify storage directory permissions

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep API keys secure
- Use environment variables for sensitive data
- Neo4j credentials should be strong passwords

## 📊 Performance

- Processes typical research paper (20 pages) in ~30 seconds
- Semantic search returns results in <1 second
- AI responses generated in 2-5 seconds
- Supports concurrent document processing

## 🌐 Deployment

For production deployment:
1. Use production-grade WSGI server (Gunicorn/Uvicorn)
2. Setup reverse proxy (Nginx)
3. Use managed Neo4j instance
4. Configure proper CORS policies
5. Enable HTTPS
6. Setup monitoring and logging

## 🎓 Learning Outcomes

This project demonstrates:
- **Full-stack AI application development**
- **RAG (Retrieval-Augmented Generation) implementation**
- **Knowledge graph construction and querying**
- **Vector database integration**
- **Modern React frontend development**
- **RESTful API design**
- **NLP pipeline orchestration**
- **Production-ready error handling**

## 🤝 Contributing

This is an educational project. Feel free to fork and experiment!

## 📝 License

MIT License - feel free to use for learning and portfolio purposes.

## 🙏 Acknowledgments

Built with:
- Google Gemini for AI capabilities
- Neo4j for graph database
- ChromaDB for vector storage
- FastAPI for backend framework
- React for frontend framework

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Built for research, powered by AI** 🚀
