"""
FastAPI Backend for Research Literature Processing Platform
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import psutil

from src.api.routes import documents, search, graph, chat
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.vector.chroma_client import get_chroma_client, check_chroma_health
from src.graph.neo4j_singleton import get_neo4j_driver, check_neo4j_health
from src.auth.supabase_db import get_db, check_supabase_health
from src.extraction.extractor import get_sentence_transformer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Create FastAPI app
app = FastAPI(
    title="Research Literature Processing API",
    description="AI-powered platform for processing research papers with semantic search and knowledge graphs",
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
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(graph.router, prefix="/api/graph", tags=["Knowledge Graph"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Assistant"])

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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Research Literature Processing API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check for all services.
    Returns status of Supabase, Neo4j, ChromaDB, and memory.
    """
    mem = psutil.virtual_memory()
    
    return {
        "status": "ok",
        "supabase": check_supabase_health(),
        "neo4j": check_neo4j_health(),
        "chroma": check_chroma_health(),
        "memory_mb": mem.available // 1024 // 1024,
        "memory_percent": mem.percent
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
