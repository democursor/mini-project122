"""
Deduplication utilities for RAG system
"""
import logging
from typing import List, Dict, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate chunks based on chunk_id or text content.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Deduplicated list of chunks
    """
    seen_ids: Set[str] = set()
    seen_texts: Set[str] = set()
    deduplicated = []
    
    for chunk in chunks:
        # Try to get unique identifier
        chunk_id = chunk.get('id') or chunk.get('chunk_id')
        text = chunk.get('text', '')
        
        # Use chunk_id if available, otherwise use text hash
        if chunk_id:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduplicated.append(chunk)
        else:
            # Fallback to text-based deduplication
            text_hash = hash(text[:500])  # Use first 500 chars for hash
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                deduplicated.append(chunk)
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(deduplicated)} unique chunks")
    return deduplicated


def deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate sources by document_id, keeping the highest relevance score.
    
    Args:
        chunks: List of chunk dictionaries with metadata
        
    Returns:
        List of unique sources with aggregated information
    """
    # Group by document_id
    doc_groups = defaultdict(list)
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        doc_id = metadata.get('document_id') or chunk.get('document_id', 'unknown')
        doc_groups[doc_id].append(chunk)
    
    # Create deduplicated sources
    deduplicated_sources = []
    
    for doc_id, doc_chunks in doc_groups.items():
        # Get the chunk with highest score
        best_chunk = max(doc_chunks, key=lambda x: x.get('score', 0))
        
        # Aggregate information
        metadata = best_chunk.get('metadata', {})
        
        source = {
            'document_id': doc_id,
            'title': metadata.get('title', metadata.get('filename', 'Untitled')),
            'filename': metadata.get('filename', doc_id),
            'score': best_chunk.get('score', 0),
            'chunk_count': len(doc_chunks),  # How many chunks from this doc
            'text': best_chunk.get('text', ''),
            'metadata': metadata
        }
        
        deduplicated_sources.append(source)
    
    # Sort by score descending
    deduplicated_sources.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Deduplicated {len(chunks)} chunks to {len(deduplicated_sources)} unique sources")
    return deduplicated_sources


def merge_chunk_results(
    sub_query_results: Dict[str, List[Dict[str, Any]]],
    max_per_query: int = 3
) -> List[Dict[str, Any]]:
    """
    Merge results from multiple sub-queries with deduplication.
    
    Args:
        sub_query_results: Dict mapping sub-query to its retrieved chunks
        max_per_query: Maximum chunks to keep per sub-query
        
    Returns:
        Merged and deduplicated list of chunks
    """
    all_chunks = []
    
    # Collect top chunks from each sub-query
    for sub_query, chunks in sub_query_results.items():
        # Take top N chunks per sub-query
        top_chunks = chunks[:max_per_query]
        
        # Tag chunks with their source sub-query
        for chunk in top_chunks:
            chunk['source_sub_query'] = sub_query
            all_chunks.append(chunk)
    
    # Deduplicate across all sub-queries
    deduplicated = deduplicate_chunks(all_chunks)
    
    logger.info(f"Merged {len(all_chunks)} chunks from {len(sub_query_results)} sub-queries "
                f"to {len(deduplicated)} unique chunks")
    
    return deduplicated
