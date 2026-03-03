"""
FastAPI Backend for Research Literature Processing Platform
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging

from src.api.routes import documents, search, graph, chat
from src.utils.config import load_config
from src.utils.logging_config import setup_logging

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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "neo4j": "checking...",
            "chromadb": "checking..."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
