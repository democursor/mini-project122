# Project Structure and Organization

## Overview

This document explains the directory structure, file organization, naming conventions, and configuration management for the Autonomous Research Literature Intelligence & Discovery Platform. Understanding project structure is crucial for maintainability, scalability, and team collaboration.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Module Organization](#module-organization)
3. [File Naming Conventions](#file-naming-conventions)
4. [Virtual Environment Setup](#virtual-environment-setup)
5. [Configuration Management](#configuration-management)
6. [Dependency Injection Patterns](#dependency-injection-patterns)
7. [Interview Preparation](#interview-preparation)

---

## Directory Structure

### Complete Project Layout

```
research-literature-platform/
│
├── docs/                           # Documentation
│   ├── README.md                   # Project overview
│   ├── REQUIREMENTS.md             # Detailed requirements
│   ├── DESIGN.md                   # System design
│   ├── TECHSTACK.md                # Technology choices
│   ├── ARCHITECTURE.md             # System architecture
│   ├── WORKFLOW.md                 # Data flow and execution
│   ├── PROJECT_STRUCTURE.md        # This file
│   ├── PHASES.md                   # Implementation phases
│   ├── INTERVIEW_GUIDE.md          # Interview preparation
│   ├── LEARNING_OUTCOMES.md        # Skills learned
│   ├── EXTENSIONS.md               # Future enhancements
│   └── SCALING.md                  # Scaling strategies
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── ingestion/                  # PDF ingestion module
│   │   ├── __init__.py
│   │   ├── uploader.py             # File upload handler
│   │   ├── validator.py            # File validation
│   │   └── storage.py              # File storage manager
│   │
│   ├── parsing/                    # PDF parsing module
│   │   ├── __init__.py
│   │   ├── parser.py               # Main parser
│   │   ├── text_extractor.py      # Text extraction
│   │   ├── metadata_extractor.py  # Metadata extraction
│   │   └── structure_analyzer.py  # Document structure
│   │
│   ├── chunking/                   # Semantic chunking module
│   │   ├── __init__.py
│   │   ├── chunker.py              # Main chunker
│   │   ├── semantic_boundary.py   # Boundary detection
│   │   └── embedding_generator.py # Sentence embeddings
│   │
│   ├── extraction/                 # Concept extraction module
│   │   ├── __init__.py
│   │   ├── concept_extractor.py   # Main extractor
│   │   ├── ner.py                  # Named entity recognition
│   │   ├── keyphrase.py            # Keyphrase extraction
│   │   └── normalizer.py           # Concept normalization
│   │
│   ├── graph/                      # Knowledge graph module
│   │   ├── __init__.py
│   │   ├── graph_builder.py       # Graph construction
│   │   ├── node_manager.py        # Node operations
│   │   ├── relationship_manager.py # Relationship operations
│   │   └── query_engine.py        # Graph queries
│   │
│   ├── vector/                     # Vector storage module
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Vector storage
│   │   ├── embedding_generator.py # Embedding generation
│   │   └── similarity_search.py   # Similarity queries
│   │
│   ├── search/                     # Semantic search module
│   │   ├── __init__.py
│   │   ├── search_engine.py       # Main search engine
│   │   ├── query_processor.py     # Query processing
│   │   ├── ranker.py               # Result ranking
│   │   └── filter.py               # Metadata filtering
│   │
│   ├── assistant/                  # AI research assistant module
│   │   ├── __init__.py
│   │   ├── rag_system.py          # RAG implementation
│   │   ├── retriever.py            # Context retrieval
│   │   ├── generator.py            # Answer generation
│   │   ├── citation_extractor.py  # Citation extraction
│   │   └── conversation_manager.py # Conversation context
│   │
│   ├── orchestration/              # Workflow orchestration
│   │   ├── __init__.py
│   │   ├── workflow.py             # LangGraph workflow
│   │   ├── state_manager.py       # State management
│   │   ├── error_handler.py       # Error handling
│   │   └── retry_logic.py         # Retry mechanisms
│   │
│   ├── persistence/                # Data persistence
│   │   ├── __init__.py
│   │   ├── database.py             # SQLite operations
│   │   ├── models.py               # Data models
│   │   ├── repositories/           # Repository pattern
│   │   │   ├── __init__.py
│   │   │   ├── document_repository.py
│   │   │   ├── chunk_repository.py
│   │   │   └── concept_repository.py
│   │   └── migrations/             # Database migrations
│   │       └── 001_initial_schema.sql
│   │
│   ├── models/                     # ML models
│   │   ├── __init__.py
│   │   ├── embedding_model.py     # Embedding model wrapper
│   │   ├── ner_model.py            # NER model wrapper
│   │   └── llm_client.py           # LLM client
│   │
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py              # Logging configuration
│   │   ├── config.py               # Configuration loader
│   │   ├── validators.py           # Validation utilities
│   │   └── helpers.py              # Helper functions
│   │
│   └── api/                        # API layer (optional)
│       ├── __init__.py
│       ├── app.py                  # FastAPI application
│       ├── routes/                 # API routes
│       │   ├── __init__.py
│       │   ├── upload.py
│       │   ├── search.py
│       │   └── chat.py
│       └── schemas/                # Pydantic schemas
│           ├── __init__.py
│           ├── document.py
│           └── search.py
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_parser.py
│   │   ├── test_chunker.py
│   │   ├── test_extractor.py
│   │   └── ...
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   └── test_search.py
│   ├── property/                   # Property-based tests
│   │   ├── __init__.py
│   │   ├── test_chunking_properties.py
│   │   └── test_extraction_properties.py
│   └── fixtures/                   # Test fixtures
│       ├── sample_papers/
│       └── test_data.py
│
├── data/                           # Data storage
│   ├── pdfs/                       # Uploaded PDF files
│   │   └── .gitkeep
│   ├── databases/                  # Database files
│   │   ├── metadata.db             # SQLite database
│   │   └── chroma/                 # ChromaDB storage
│   └── logs/                       # Log files
│       └── app.log
│
├── config/                         # Configuration files
│   ├── default.yaml                # Default configuration
│   ├── development.yaml            # Development config
│   ├── production.yaml             # Production config
│   └── models.yaml                 # Model configurations
│
├── scripts/                        # Utility scripts
│   ├── setup_environment.sh        # Environment setup
│   ├── download_models.py          # Download ML models
│   ├── init_databases.py           # Initialize databases
│   └── run_tests.sh                # Run test suite
│
├── notebooks/                      # Jupyter notebooks
│   ├── exploration/                # Data exploration
│   ├── experiments/                # Experiments
│   └── demos/                      # Demonstrations
│
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Project metadata
├── README.md                       # Project README
└── LICENSE                         # License file
```

### Directory Purpose Explanation

#### `/docs` - Documentation
**Purpose:** All project documentation for learning and reference

**Contents:**
- Requirements, design, and architecture documents
- Technology stack explanations
- Implementation guides
- Interview preparation materials

**Why separate:** Documentation should be easily accessible and not mixed with code

#### `/src` - Source Code
**Purpose:** All application source code organized by module

**Organization Principle:** Each subdirectory represents a major system component with clear responsibilities

**Why this structure:**
- **Modularity:** Each component is independent
- **Testability:** Easy to test components in isolation
- **Maintainability:** Changes are localized to specific modules
- **Scalability:** Easy to add new modules without affecting existing ones

#### `/tests` - Test Suite
**Purpose:** All test code organized by test type

**Organization:**
- `unit/`: Test individual functions and classes
- `integration/`: Test component interactions
- `property/`: Property-based tests for correctness
- `fixtures/`: Shared test data and utilities

**Why separate:** Tests should mirror source structure but remain independent

#### `/data` - Data Storage
**Purpose:** All runtime data (PDFs, databases, logs)

**Why separate:** Data should not be in version control (except `.gitkeep` files)

**Backup strategy:** This directory should be backed up regularly

#### `/config` - Configuration
**Purpose:** Environment-specific configuration files

**Why separate:** Configuration should be external to code for easy deployment

#### `/scripts` - Utility Scripts
**Purpose:** Setup, maintenance, and utility scripts

**Why separate:** Scripts are tools, not application code

#### `/notebooks` - Jupyter Notebooks
**Purpose:** Exploration, experiments, and demonstrations

**Why separate:** Notebooks are for interactive work, not production code

---

## Module Organization

### Module Design Principles

#### 1. Single Responsibility Principle
Each module has one clear purpose:
- `ingestion/`: Only handles file upload and validation
- `parsing/`: Only handles PDF parsing
- `chunking/`: Only handles text segmentation

#### 2. Dependency Direction
Dependencies flow in one direction (no circular dependencies):
```
orchestration → ingestion → parsing → chunking → extraction → [graph, vector]
                                                                    ↓
                                                                 search
                                                                    ↓
                                                                assistant
```

#### 3. Interface Segregation
Each module exposes a clean interface:
```python
# Good: Clear interface
class Parser:
    def parse(self, pdf_path: Path) -> ParsedDocument:
        pass

# Bad: Exposing internals
class Parser:
    def _extract_text(self, page): pass  # Internal method exposed
    def _parse_metadata(self, text): pass  # Internal method exposed
```

#### 4. Dependency Injection
Modules receive dependencies, don't create them:
```python
# Good: Dependencies injected
class ChunkingService:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

# Bad: Creating dependencies internally
class ChunkingService:
    def __init__(self):
        self.embedding_model = SentenceBERT()  # Hard-coded dependency
```

### Module Structure Template

Each module follows this structure:
```
module_name/
├── __init__.py           # Public interface
├── main_class.py         # Main implementation
├── helpers.py            # Helper functions
├── models.py             # Data models
└── exceptions.py         # Custom exceptions
```

**Example: `chunking/` module**
```python
# chunking/__init__.py
from .chunker import ChunkingService
from .models import Chunk, ChunkingConfig

__all__ = ['ChunkingService', 'Chunk', 'ChunkingConfig']

# chunking/chunker.py
class ChunkingService:
    """Main chunking implementation"""
    pass

# chunking/models.py
@dataclass
class Chunk:
    """Chunk data model"""
    chunk_id: str
    text: str
    position: int
    token_count: int

# chunking/exceptions.py
class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass
```

---

## File Naming Conventions

### Python Files

**Convention:** `snake_case.py`

**Examples:**
- `pdf_parser.py` (not `PDFParser.py` or `pdfParser.py`)
- `embedding_generator.py` (not `embeddingGenerator.py`)
- `concept_extractor.py` (not `ConceptExtractor.py`)

**Why:** Python convention (PEP 8)

### Classes

**Convention:** `PascalCase`

**Examples:**
- `class PDFParser`
- `class EmbeddingGenerator`
- `class ConceptExtractor`

**Why:** Python convention (PEP 8)

### Functions and Variables

**Convention:** `snake_case`

**Examples:**
- `def parse_pdf()`
- `def generate_embedding()`
- `chunk_size = 500`

**Why:** Python convention (PEP 8)

### Constants

**Convention:** `UPPER_SNAKE_CASE`

**Examples:**
- `MAX_FILE_SIZE = 50 * 1024 * 1024`  # 50MB
- `DEFAULT_CHUNK_SIZE = 500`
- `EMBEDDING_DIMENSIONS = 384`

**Why:** Python convention (PEP 8)

### Configuration Files

**Convention:** `lowercase.yaml` or `lowercase.json`

**Examples:**
- `default.yaml`
- `development.yaml`
- `models.yaml`

**Why:** Consistency and readability

### Test Files

**Convention:** `test_<module_name>.py`

**Examples:**
- `test_parser.py`
- `test_chunker.py`
- `test_extractor.py`

**Why:** pytest convention (auto-discovery)

---

## Virtual Environment Setup

### Why Virtual Environments?

**Problem:** Different projects need different package versions

**Solution:** Isolate dependencies per project

**Benefits:**
- No conflicts between projects
- Reproducible environments
- Easy to share requirements
- Clean system Python installation

### Setup Instructions

#### Option 1: venv (Built-in)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate
deactivate
```

#### Option 2: conda

```bash
# Create environment
conda create -n research-platform python=3.10

# Activate
conda activate research-platform

# Install dependencies
pip install -r requirements.txt

# Deactivate
conda deactivate
```

### Requirements Files

#### `requirements.txt` - Production Dependencies
```
# Deep Learning & NLP
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
spacy>=3.5.0
keybert>=0.8.0

# PDF Processing
pymupdf>=1.22.0

# Vector Store
chromadb>=0.4.0

# Knowledge Graph
neo4j>=5.0.0

# Orchestration
langgraph>=0.1.0
langchain>=0.1.0

# GenAI
openai>=1.0.0

# Data & Utilities
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pydantic>=2.0.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0
```

#### `requirements-dev.txt` - Development Dependencies
```
# Include production dependencies
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
hypothesis>=6.82.0

# Code Quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
pylint>=2.17.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=1.2.0

# Development Tools
jupyter>=1.0.0
ipython>=8.14.0
```

### Environment Variables

#### `.env.example` Template
```bash
# Application
APP_ENV=development
LOG_LEVEL=INFO

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/databases/chroma

# LLM
OPENAI_API_KEY=your_api_key_here
LLM_PROVIDER=openai  # or 'ollama'
LLM_MODEL=gpt-4

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
NER_MODEL=en_core_sci_md

# Storage
PDF_STORAGE_PATH=./data/pdfs
MAX_FILE_SIZE_MB=50

# Processing
MAX_CONCURRENT_DOCUMENTS=10
CHUNK_MIN_TOKENS=100
CHUNK_MAX_TOKENS=500
```

#### Loading Environment Variables
```python
# src/utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Application
    APP_ENV = os.getenv('APP_ENV', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Database
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    # LLM
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
    
    # Models
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
```

---

## Configuration Management

### Configuration Hierarchy

```
default.yaml (base configuration)
  ↓
development.yaml (overrides for dev)
  ↓
.env (environment-specific secrets)
  ↓
Command-line arguments (runtime overrides)
```

### Configuration Files

#### `config/default.yaml`
```yaml
# Application
app:
  name: "Research Literature Platform"
  version: "1.0.0"
  environment: "development"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./data/logs/app.log"

# Processing
processing:
  max_concurrent_documents: 10
  retry_attempts: 3
  retry_backoff_factor: 2

# Chunking
chunking:
  min_tokens: 100
  max_tokens: 500
  similarity_threshold: 0.7

# Models
models:
  embedding:
    name: "all-MiniLM-L6-v2"
    dimensions: 384
    device: "cpu"
  ner:
    name: "en_core_sci_md"
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.3
    max_tokens: 1000

# Storage
storage:
  pdf_directory: "./data/pdfs"
  database_path: "./data/databases/metadata.db"
  chroma_directory: "./data/databases/chroma"
  max_file_size_mb: 50

# Search
search:
  top_k_chunks: 50
  top_k_documents: 10
  min_similarity: 0.5
```

#### `config/development.yaml`
```yaml
# Override for development
logging:
  level: "DEBUG"

models:
  embedding:
    device: "cpu"  # Use CPU for development

processing:
  max_concurrent_documents: 5  # Lower for development
```

#### `config/production.yaml`
```yaml
# Override for production
logging:
  level: "WARNING"

models:
  embedding:
    device: "cuda"  # Use GPU in production

processing:
  max_concurrent_documents: 20  # Higher for production

storage:
  pdf_directory: "/var/data/pdfs"
  database_path: "/var/data/databases/metadata.db"
```

### Loading Configuration

```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, env: str = "development"):
        self.env = env
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        # Load default config
        default_path = Path("config/default.yaml")
        with open(default_path) as f:
            config = yaml.safe_load(f)
        
        # Load environment-specific config
        env_path = Path(f"config/{self.env}.yaml")
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f)
                config = self._deep_merge(config, env_config)
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Usage
config = ConfigLoader(env="development")
embedding_model = config.get('models.embedding.name')  # "all-MiniLM-L6-v2"
max_concurrent = config.get('processing.max_concurrent_documents')  # 5
```

---

## Dependency Injection Patterns

### Why Dependency Injection?

**Problem:** Hard-coded dependencies make code difficult to test and modify

**Solution:** Inject dependencies from outside

**Benefits:**
- Easy to test (inject mocks)
- Easy to swap implementations
- Clear dependencies
- Loose coupling

### Pattern 1: Constructor Injection

```python
# Good: Dependencies injected via constructor
class ChunkingService:
    def __init__(self, embedding_model: EmbeddingModel, config: Config):
        self.embedding_model = embedding_model
        self.config = config
    
    def chunk(self, text: str) -> List[Chunk]:
        # Use injected dependencies
        embeddings = self.embedding_model.encode(text)
        threshold = self.config.get('chunking.similarity_threshold')
        # ...

# Usage
embedding_model = SentenceBERT()
config = ConfigLoader()
chunker = ChunkingService(embedding_model, config)
```

### Pattern 2: Factory Pattern

```python
# Factory creates objects with dependencies
class ServiceFactory:
    def __init__(self, config: Config):
        self.config = config
        self._embedding_model = None
        self._ner_model = None
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            model_name = self.config.get('models.embedding.name')
            self._embedding_model = SentenceBERT(model_name)
        return self._embedding_model
    
    @property
    def ner_model(self) -> NERModel:
        if self._ner_model is None:
            model_name = self.config.get('models.ner.name')
            self._ner_model = SpacyNER(model_name)
        return self._ner_model
    
    def create_chunker(self) -> ChunkingService:
        return ChunkingService(
            embedding_model=self.embedding_model,
            config=self.config
        )
    
    def create_extractor(self) -> ConceptExtractor:
        return ConceptExtractor(
            ner_model=self.ner_model,
            config=self.config
        )

# Usage
factory = ServiceFactory(config)
chunker = factory.create_chunker()
extractor = factory.create_extractor()
```

### Pattern 3: Dependency Injection Container

```python
# Container manages all dependencies
class DIContainer:
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, name: str, factory: Callable, singleton: bool = False):
        """Register a service factory"""
        self._services[name] = {
            'factory': factory,
            'singleton': singleton
        }
    
    def get(self, name: str) -> Any:
        """Get a service instance"""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")
        
        service = self._services[name]
        
        # Return singleton if already created
        if service['singleton'] and name in self._singletons:
            return self._singletons[name]
        
        # Create new instance
        instance = service['factory'](self)
        
        # Store singleton
        if service['singleton']:
            self._singletons[name] = instance
        
        return instance

# Setup container
container = DIContainer()

# Register services
container.register('config', lambda c: ConfigLoader(), singleton=True)
container.register('embedding_model', lambda c: SentenceBERT(), singleton=True)
container.register('ner_model', lambda c: SpacyNER(), singleton=True)
container.register('chunker', lambda c: ChunkingService(
    c.get('embedding_model'),
    c.get('config')
))
container.register('extractor', lambda c: ConceptExtractor(
    c.get('ner_model'),
    c.get('config')
))

# Usage
chunker = container.get('chunker')
extractor = container.get('extractor')
```

---

## Interview Preparation

### Key Talking Points

1. **Project Structure**
   - "The project follows a modular architecture with clear separation of concerns"
   - "Each module has a single responsibility and well-defined interfaces"
   - "Dependencies flow in one direction to avoid circular dependencies"

2. **Configuration Management**
   - "We use a hierarchical configuration system with environment-specific overrides"
   - "Secrets are stored in environment variables, not in code"
   - "Configuration is external to code for easy deployment across environments"

3. **Dependency Injection**
   - "We use dependency injection to make code testable and maintainable"
   - "Dependencies are injected via constructors, not created internally"
   - "This allows easy mocking for tests and swapping implementations"

4. **Virtual Environments**
   - "We use virtual environments to isolate project dependencies"
   - "This prevents conflicts between projects and ensures reproducibility"
   - "Requirements files make it easy to set up the same environment anywhere"

### Sample Interview Questions

**Q: How is your project organized?**

**A:** "The project follows a modular architecture with clear separation of concerns. The main source code is in `/src`, organized by component (ingestion, parsing, chunking, extraction, etc.). Each component is independent with well-defined interfaces.

Tests mirror the source structure in `/tests`, with separate directories for unit, integration, and property-based tests. Documentation is in `/docs`, configuration in `/config`, and runtime data in `/data`.

This structure makes the codebase easy to navigate, test, and maintain. New developers can quickly understand the system by looking at the directory structure."

**Q: How do you manage configuration across different environments?**

**A:** "We use a hierarchical configuration system:

1. `default.yaml` contains base configuration
2. Environment-specific files (`development.yaml`, `production.yaml`) override defaults
3. Environment variables (`.env`) store secrets
4. Command-line arguments provide runtime overrides

This allows us to:
- Keep secrets out of version control
- Easily deploy to different environments
- Override configuration without changing code
- Share common configuration across environments

For example, in development we use CPU for models, but in production we override to use GPU."

**Q: How do you handle dependencies between components?**

**A:** "We use dependency injection to manage component dependencies. Instead of components creating their own dependencies, they receive them via constructor injection.

For example, the ChunkingService receives an EmbeddingModel and Config as constructor parameters, rather than creating them internally. This makes the code:
- Testable: We can inject mocks for testing
- Flexible: Easy to swap implementations
- Clear: Dependencies are explicit, not hidden

We also use a factory pattern to centralize object creation, ensuring dependencies are wired correctly throughout the application."

**Q: Why separate tests from source code?**

**A:** "Separating tests from source code provides several benefits:

1. **Clarity**: Clear distinction between production and test code
2. **Packaging**: Production packages don't include test code
3. **Organization**: Tests can be organized differently than source (by type: unit, integration, property)
4. **Independence**: Tests can import source code, but source never imports tests

The test structure mirrors the source structure, making it easy to find tests for any module. For example, `src/chunking/chunker.py` has tests in `tests/unit/test_chunker.py`."

---

**This project structure demonstrates production-grade organization principles that companies value: modularity, testability, configurability, and maintainability.**
