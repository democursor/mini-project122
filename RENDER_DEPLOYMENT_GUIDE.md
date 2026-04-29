# Render Free Tier Deployment Guide

## Complete Implementation Guide for All 7 Issues

This document provides the complete code changes needed to deploy on Render free tier (512MB RAM) with Neo4j AuraDB Free, Supabase Free, and external ChromaDB.

---

## ISSUE 1: ChromaDB HTTP Client (Ephemeral Storage Fix)

### Create: `src/vector/chroma_client.py`

```python
"""
ChromaDB HTTP client singleton for Render deployment
"""
import chromadb
from chromadb.config import Settings
import os
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Thread-safe singleton
_chroma_client: Optional[chromadb.HttpClient] = None
_chroma_lock = threading.Lock()


def get_chroma_client() -> chromadb.HttpClient:
    """
    Get ChromaDB HTTP client using thread-safe singleton pattern.
    Connects to external ChromaDB server instead of local disk.
    """
    global _chroma_client
    
    # Fast path
    if _chroma_client is not None:
        logger.debug("Reusing existing ChromaDB HTTP client (singleton)")
        return _chroma_client
    
    # Slow path with lock
    with _chroma_lock:
        if _chroma_client is not None:
            return _chroma_client
        
        try:
            host = os.getenv("CHROMA_HOST", "localhost")
            port = int(os.getenv("CHROMA_PORT", "8000"))
            
            logger.info(f"Creating ChromaDB HTTP client: {host}:{port}")
            
            _chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Test connection
            _chroma_client.heartbeat()
            
            logger.info("ChromaDB HTTP client created successfully (singleton)")
            return _chroma_client
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB HTTP client: {e}")
            raise ConnectionError(f"ChromaDB unavailable: {e}")


def check_chroma_health() -> dict:
    """Check ChromaDB connection health"""
    try:
        client = get_chroma_client()
        client.heartbeat()
        return {"status": "connected", "type": "http"}
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}
```

### Update: `src/vector/store.py`

```python
"""Vector storage using ChromaDB HTTP client"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from .models import ChunkEmbedding
from .chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage using ChromaDB HTTP client"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize with HTTP client (persist_directory ignored for HTTP mode)
        """
        self.logger = logger
        try:
            self.client = get_chroma_client()
            self.collection = self._get_or_create_collection()
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorStore: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create the main document collection"""
        collection_name = "research_papers"
        
        try:
            collection = self.client.get_collection(collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
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
    
    # ... rest of methods remain the same ...
```

---

## ISSUE 2: Neo4j Connection Pool + Reconnection

### Create: `src/graph/neo4j_singleton.py`

```python
"""
Neo4j driver singleton with connection pooling and retry logic
"""
import logging
import os
import threading
from typing import Optional
from functools import wraps
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

logger = logging.getLogger(__name__)

# Thread-safe singleton
_neo4j_driver: Optional[Driver] = None
_neo4j_lock = threading.Lock()


def get_neo4j_driver() -> Driver:
    """
    Get Neo4j driver using thread-safe singleton pattern.
    Configured for Render free tier (512MB RAM).
    """
    global _neo4j_driver
    
    # Fast path
    if _neo4j_driver is not None:
        logger.debug("Reusing existing Neo4j driver (singleton)")
        return _neo4j_driver
    
    # Slow path with lock
    with _neo4j_lock:
        if _neo4j_driver is not None:
            return _neo4j_driver
        
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            logger.info(f"Creating Neo4j driver: {uri}")
            
            _neo4j_driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=5,        # Limit for free tier
                connection_timeout=10,              # Fail fast
                max_connection_lifetime=200,        # Recycle stale connections
                keep_alive=True                     # Keep connections alive
            )
            
            # Test connection
            _neo4j_driver.verify_connectivity()
            
            logger.info("Neo4j driver created successfully (singleton)")
            return _neo4j_driver
            
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            raise ConnectionError(f"Neo4j unavailable: {e}")


def neo4j_retry(func):
    """
    Decorator for Neo4j operations with automatic reconnection.
    Handles AuraDB Free pause/resume (72hr inactivity).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ServiceUnavailable, SessionExpired) as e:
            logger.warning(f"Neo4j connection stale/expired: {e}")
            logger.info("Attempting to reconnect to Neo4j...")
            
            # Force reconnection by clearing singleton
            global _neo4j_driver
            with _neo4j_lock:
                if _neo4j_driver:
                    try:
                        _neo4j_driver.close()
                    except:
                        pass
                    _neo4j_driver = None
            
            # Retry once with new connection
            try:
                get_neo4j_driver()
                logger.info("Neo4j reconnection successful, retrying operation")
                return func(*args, **kwargs)
            except Exception as retry_error:
                logger.error(f"Neo4j reconnection failed: {retry_error}")
                raise ConnectionError("Graph database temporarily unavailable")
        except Exception as e:
            logger.error(f"Neo4j operation failed: {e}")
            raise
    
    return wrapper


def check_neo4j_health() -> dict:
    """Check Neo4j connection health"""
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        return {"status": "connected", "pool_size": 5}
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}
```

### Update: `src/graph/neo4j_client.py`

```python
"""
Neo4j client wrapper using singleton driver
"""
import logging
from src.graph.neo4j_singleton import get_neo4j_driver, neo4j_retry

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client using singleton driver"""
    
    def __init__(self):
        """Initialize using singleton driver"""
        try:
            self.driver = get_neo4j_driver()
            self.database = "neo4j"  # Default database
            logger.debug("Neo4jClient initialized with singleton driver")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j client: {e}")
            self.driver = None
            self.database = None
    
    @neo4j_retry
    def is_connected(self) -> bool:
        """Check if Neo4j is connected"""
        try:
            if self.driver is None:
                self.driver = get_neo4j_driver()
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j connectivity check failed: {e}")
            return False
    
    def close(self):
        """Close is handled by singleton - no-op here"""
        logger.debug("Neo4j close called (singleton manages lifecycle)")
```

---

## ISSUE 3: Supabase Pause Handling

### Update: `src/auth/supabase_db.py`

Add timeout and error handling to the existing singleton:

```python
from supabase import create_client, Client
from typing import List, Optional
import os
import logging
import threading
from functools import wraps

logger = logging.getLogger(__name__)

# Thread-safe singleton
_supabase_client: Optional[Client] = None
_supabase_lock = threading.Lock()


def get_db() -> Client:
    """Get Supabase client with 8-second timeout"""
    global _supabase_client
    
    if _supabase_client is not None:
        logger.debug("Reusing existing Supabase client (singleton)")
        return _supabase_client
    
    with _supabase_lock:
        if _supabase_client is not None:
            return _supabase_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        
        logger.info(f"Creating Supabase client with URL: {url[:30]}...")
        
        try:
            # Create client with timeout configuration
            import httpx
            _supabase_client = create_client(
                url, 
                key,
                options={
                    "timeout": httpx.Timeout(8.0, connect=5.0)  # 8s total, 5s connect
                }
            )
            logger.info("Supabase client created successfully (singleton)")
            return _supabase_client
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            raise RuntimeError(f"Failed to create Supabase client: {e}")


def supabase_error_handler(func):
    """
    Decorator to handle Supabase pause/connection errors.
    Returns HTTP 503 instead of crashing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for connection/timeout errors
            if any(keyword in error_msg for keyword in [
                'timeout', 'connection', 'unreachable', 'refused', 'paused'
            ]):
                logger.error(f"Supabase connection error: {e}", exc_info=True)
                raise ConnectionError(
                    "Database temporarily unavailable, please try again in 30 seconds"
                )
            else:
                # Re-raise other errors
                logger.error(f"Supabase operation error: {e}", exc_info=True)
                raise
    
    return wrapper


def check_supabase_health() -> dict:
    """Check Supabase connection health"""
    try:
        client = get_db()
        # Simple query to test connection
        client.table("documents").select("id").limit(1).execute()
        return {"status": "connected", "timeout": "8s"}
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        return {"status": "disconnected", "error": str(e)}


# Update all DB classes to use decorator
class DocumentDB:
    def __init__(self):
        self.db = get_db()

    @supabase_error_handler
    def upsert_document(self, doc_data: dict) -> dict:
        result = self.db.table("documents").upsert(doc_data).execute()
        return result.data[0] if result.data else {}

    @supabase_error_handler
    def get_document(self, document_id: str, user_id: str) -> Optional[dict]:
        result = (
            self.db.table("documents")
            .select("*")
            .eq("id", document_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return result.data
    
    # ... apply @supabase_error_handler to all other methods ...
```

---

## ISSUE 4: Memory Guard Middleware

### Update: `src/api/main.py`

Add memory guard middleware:

```python
"""
FastAPI Backend with Render Free Tier optimizations
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import psutil

from src.api.routes import documents, search, graph, chat
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Research Literature Processing API",
    description="AI-powered platform for research papers",
    version="1.0.0"
)

# Memory guard middleware (ISSUE 4)
@app.middleware("http")
async def memory_guard(request: Request, call_next):
    """
    Memory guard for Render free tier (512MB RAM).
    Returns 503 if available memory < 120MB.
    """
    # Skip memory check for health endpoint
    if request.url.path == "/health":
        return await call_next(request)
    
    try:
        mem = psutil.virtual_memory()
        available_mb = mem.available // 1024 // 1024
        
        if available_mb < 120:
            logger.warning(f"Low memory: {available_mb}MB available")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Server under load. Try again in a moment.",
                    "available_memory_mb": available_mb
                }
            )
        
        return await call_next(request)
    except Exception as e:
        logger.error(f"Memory guard error: {e}")
        return await call_next(request)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(graph.router, prefix="/api/graph", tags=["Knowledge Graph"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Assistant"])

# ... rest of the file ...
```

---

## ISSUE 5: Background Processing + Job Status

### Create: `src/api/models.py` additions

```python
# Add to existing models

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
```

### Create: `src/services/job_service.py`

```python
"""
Background job service for async document processing
"""
import logging
import uuid
from datetime import datetime
from typing import Optional
from src.auth.supabase_db import get_db, supabase_error_handler

logger = logging.getLogger(__name__)


class JobService:
    """Manages background job status in Supabase"""
    
    def __init__(self):
        self.db = get_db()
    
    @supabase_error_handler
    def create_job(self, job_type: str = "document_processing") -> str:
        """Create a new job and return job_id"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "status": "pending",
            "job_type": job_type,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.db.table("jobs").insert(job_data).execute()
        logger.info(f"Created job: {job_id}")
        return job_id
    
    @supabase_error_handler
    def update_job_status(self, job_id: str, status: str, 
                         result: dict = None, error: str = None):
        """Update job status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if result:
            update_data["result"] = result
        if error:
            update_data["error"] = error
        
        self.db.table("jobs").update(update_data).eq("id", job_id).execute()
        logger.info(f"Updated job {job_id}: {status}")
    
    @supabase_error_handler
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status"""
        result = self.db.table("jobs").select("*").eq("id", job_id).single().execute()
        return result.data if result.data else None
```

### Update: `src/api/routes/documents.py`

```python
from fastapi import BackgroundTasks
from src.services.job_service import JobService

job_service = JobService()

@router.post("/", response_model=JobResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Upload document - returns immediately with job_id.
    Processing happens in background.
    """
    try:
        # Create job
        job_id = job_service.create_job("document_processing")
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            job_id=job_id,
            file=file,
            user_id=current_user["id"]
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Document upload accepted. Processing in background."
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(job_id: str, file: UploadFile, user_id: str):
    """Background task for document processing"""
    try:
        job_service.update_job_status(job_id, "processing")
        
        # Your existing processing logic here
        # ... chunking, embedding, graph building ...
        
        result = {"document_id": "doc_123", "chunks": 50}
        job_service.update_job_status(job_id, "completed", result=result)
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        job_service.update_job_status(job_id, "failed", error=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
```

---

## ISSUE 6: Startup Warmup + Health Check

### Update: `src/api/main.py`

```python
from src.vector.chroma_client import get_chroma_client, check_chroma_health
from src.graph.neo4j_singleton import get_neo4j_driver, check_neo4j_health
from src.auth.supabase_db import get_db, check_supabase_health
from src.extraction.extractor import get_sentence_transformer

@app.on_event("startup")
async def startup():
    """
    Warm up all singletons at startup.
    Loads ML models and establishes connections.
    """
    logger.info("Starting application warmup...")
    
    try:
        # Warm Supabase
        get_db()
        logger.info("✓ Supabase client initialized")
    except Exception as e:
        logger.error(f"✗ Supabase initialization failed: {e}")
    
    try:
        # Warm Neo4j
        get_neo4j_driver()
        logger.info("✓ Neo4j driver initialized")
    except Exception as e:
        logger.error(f"✗ Neo4j initialization failed: {e}")
    
    try:
        # Warm ChromaDB
        get_chroma_client()
        logger.info("✓ ChromaDB client initialized")
    except Exception as e:
        logger.error(f"✗ ChromaDB initialization failed: {e}")
    
    try:
        # Load ML model
        get_sentence_transformer("all-MiniLM-L6-v2")
        logger.info("✓ SentenceTransformer model loaded")
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
    
    logger.info("Application warmup complete")


@app.get("/health")
async def health():
    """
    Comprehensive health check for all services.
    Returns status of Supabase, Neo4j, ChromaDB, and memory.
    """
    import psutil
    
    mem = psutil.virtual_memory()
    
    return {
        "status": "ok",
        "supabase": check_supabase_health(),
        "neo4j": check_neo4j_health(),
        "chroma": check_chroma_health(),
        "memory_mb": mem.available // 1024 // 1024,
        "memory_percent": mem.percent
    }
```

---

## ISSUE 7: Uvicorn Start Command + render.yaml

### Create: `render.yaml`

```yaml
services:
  - type: web
    name: research-platform-api
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port 10000 --workers 1 --timeout-keep-alive 75
    envVars:
      - key: NEO4J_URI
        sync: false
      - key: NEO4J_USER
        sync: false
      - key: NEO4J_PASSWORD
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_ROLE_KEY
        sync: false
      - key: CHROMA_HOST
        sync: false
      - key: CHROMA_PORT
        value: 8000
      - key: GOOGLE_API_KEY
        sync: false
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Update: `requirements.txt`

Add psutil if not present:

```txt
psutil>=5.9.0
httpx>=0.24.0
```

---

## Supabase Schema Update

Add jobs table:

```sql
-- Create jobs table for background processing
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    job_type TEXT NOT NULL DEFAULT 'document_processing',
    result JSONB,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);
```

---

## Environment Variables for Render

Set these in Render dashboard:

```bash
# Neo4j AuraDB Free
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Supabase Free
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# ChromaDB (external server)
CHROMA_HOST=your-chroma-server.com
CHROMA_PORT=8000

# Google AI
GOOGLE_API_KEY=your_google_api_key
```

---

## Testing Locally

```bash
# Set environment variables
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export SUPABASE_URL=https://xxxxx.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=your_key

# Run with production settings
uvicorn src.api.main:app --host 0.0.0.0 --port 10000 --workers 1 --timeout-keep-alive 75
```

---

## Deployment Checklist

- [ ] Update all files as shown above
- [ ] Add `render.yaml` to project root
- [ ] Update `requirements.txt` with psutil and httpx
- [ ] Run Supabase schema update (jobs table)
- [ ] Set up external ChromaDB server
- [ ] Configure environment variables in Render
- [ ] Deploy to Render
- [ ] Test `/health` endpoint
- [ ] Test document upload (background processing)
- [ ] Monitor memory usage in Render logs

---

## Expected Behavior

1. **Startup**: All singletons warm up, models load once
2. **Memory**: Stays under 400MB with guard at 120MB
3. **ChromaDB**: Connects to external server, no data loss
4. **Neo4j**: Auto-reconnects after AuraDB pause
5. **Supabase**: Returns 503 gracefully when paused
6. **Processing**: Runs in background, returns job_id immediately
7. **Health**: Shows status of all services

---

## Monitoring

Check Render logs for:
- "Application warmup complete"
- "Reusing existing [service] client (singleton)"
- "Low memory: XXmb available" (should be rare)
- "Neo4j reconnection successful"
- "Supabase connection error" (when paused)

---

This completes all 7 deployment fixes for Render free tier!
