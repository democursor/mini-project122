"""
Agentic multi-step RAG pipeline with query decomposition and structured synthesis.
"""
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.rag.query_classifier import QueryClassifier, QueryIntent
from src.rag.query_decomposer import QueryDecomposer
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.utils.deduplication import deduplicate_chunks, deduplicate_sources, merge_chunk_results

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """
    Multi-step agentic RAG pipeline.
    
    Pipeline Steps:
    1. Query Intent Classification (SIMPLE/COMPLEX/COMPARATIVE)
    2. Query Decomposition (for COMPLEX queries)
    3. Parallel Retrieval with Deduplication
    4. Structured Synthesis
    5. Source Display with Deduplication
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        llm_client: LLMClient,
        max_workers: int = 3
    ):
        """
        Initialize agentic RAG pipeline.
        
        Args:
            retriever: RAG retriever instance
            llm_client: LLM client instance
            max_workers: Max parallel workers for retrieval
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.classifier = QueryClassifier()
        self.decomposer = QueryDecomposer(llm_client)
        self.max_workers = max_workers
        
        logger.info("AgenticRAGPipeline initialized")
    
    def process_query(
        self,
        query: str,
        top_k_per_query: int = 5,
        use_graph: bool = False
    ) -> Dict[str, Any]:
        """
        Process query through agentic pipeline.
        
        Args:
            query: User query
            top_k_per_query: Number of chunks per sub-query
            use_graph: Whether to use graph-enhanced retrieval
            
        Returns:
            Dict with structured response
        """
        logger.info(f"Processing query through agentic pipeline: {query[:100]}...")
        
        # Step 1: Classify query intent
        intent = self.classifier.classify(query)
        classification_meta = self.classifier.get_classification_metadata(query)
        
        logger.info(f"Query classified as: {intent.value}")
        
        # Step 2: Handle based on intent
        if intent == QueryIntent.SIMPLE:
            return self._process_simple_query(query, top_k_per_query, use_graph)
        elif intent == QueryIntent.COMPLEX:
            return self._process_complex_query(query, top_k_per_query, use_graph)
        elif intent == QueryIntent.COMPARATIVE:
            return self._process_comparative_query(query, top_k_per_query, use_graph)
        
        # Fallback
        return self._process_simple_query(query, top_k_per_query, use_graph)
    
    def _process_simple_query(
        self,
        query: str,
        top_k: int,
        use_graph: bool
    ) -> Dict[str, Any]:
        """Process simple query with standard RAG."""
        logger.info("Processing as SIMPLE query")
        
        # Single retrieval
        context = self.retriever.retrieve_context(query, top_k)
        
        if not context.retrieved_chunks:
            return {
                'intent': 'simple',
                'answer': "I couldn't find relevant information in the uploaded documents.",
                'sections': [],
                'sources': [],
                'metadata': {
                    'sub_queries': [],
                    'chunks_retrieved': 0
                }
            }
        
        # Deduplicate sources
        deduplicated_sources = deduplicate_sources(context.retrieved_chunks)
        
        # Generate answer (BUG FIX 1: Increased max_tokens to 4096 to prevent mid-sentence cutoff)
        prompt = self._build_simple_prompt(query, context.retrieved_chunks)
        answer = self.llm_client.generate_response(prompt, max_tokens=4096, temperature=0.3)
        
        return {
            'intent': 'simple',
            'answer': answer,
            'sections': [],  # No sections for simple queries
            'sources': deduplicated_sources,
            'metadata': {
                'sub_queries': [],
                'chunks_retrieved': len(context.retrieved_chunks),
                'unique_sources': len(deduplicated_sources)
            }
        }
    
    def _process_complex_query(
        self,
        query: str,
        top_k_per_query: int,
        use_graph: bool
    ) -> Dict[str, Any]:
        """Process complex query with decomposition and structured synthesis."""
        logger.info("Processing as COMPLEX query")
        
        # Step 2: Decompose query
        sub_queries = self.decomposer.decompose(query, min_subqueries=3, max_subqueries=6)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        
        # Step 3: Parallel retrieval
        sub_query_results = self._parallel_retrieval(sub_queries, top_k_per_query)
        
        # Merge and deduplicate
        all_chunks = merge_chunk_results(sub_query_results, max_per_query=3)
        
        if not all_chunks:
            return {
                'intent': 'complex',
                'answer': "I couldn't find relevant information in the uploaded documents.",
                'sections': [],
                'sources': [],
                'metadata': {
                    'sub_queries': sub_queries,
                    'chunks_retrieved': 0
                }
            }
        
        # Deduplicate sources for display
        deduplicated_sources = deduplicate_sources(all_chunks)
        
        # Step 4: Structured synthesis
        sections = self._synthesize_structured_response(query, sub_queries, all_chunks)
        
        # Build final answer from sections
        answer_parts = []
        for section in sections:
            answer_parts.append(f"## {section['title']}\n\n{section['content']}")
        
        answer = "\n\n".join(answer_parts)
        
        return {
            'intent': 'complex',
            'answer': answer,
            'sections': sections,
            'sources': deduplicated_sources,
            'metadata': {
                'sub_queries': sub_queries,
                'chunks_retrieved': len(all_chunks),
                'unique_sources': len(deduplicated_sources)
            }
        }
    
    def _process_comparative_query(
        self,
        query: str,
        top_k: int,
        use_graph: bool
    ) -> Dict[str, Any]:
        """Process comparative query."""
        logger.info("Processing as COMPARATIVE query")
        
        # For now, treat as complex query
        # TODO: Implement specialized comparative logic
        return self._process_complex_query(query, top_k, use_graph)
    
    def _parallel_retrieval(
        self,
        sub_queries: List[str],
        top_k: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks for multiple sub-queries in parallel.
        
        Args:
            sub_queries: List of sub-queries
            top_k: Number of chunks per query
            
        Returns:
            Dict mapping sub-query to retrieved chunks
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            future_to_query = {
                executor.submit(self.retriever.retrieve_context, sq, top_k): sq
                for sq in sub_queries
            }
            
            # Collect results
            for future in as_completed(future_to_query):
                sub_query = future_to_query[future]
                try:
                    context = future.result()
                    results[sub_query] = context.retrieved_chunks
                    logger.debug(f"Retrieved {len(context.retrieved_chunks)} chunks for: {sub_query[:50]}...")
                except Exception as e:
                    logger.error(f"Error retrieving for sub-query '{sub_query}': {e}")
                    results[sub_query] = []
        
        return results
    
    def _synthesize_structured_response(
        self,
        original_query: str,
        sub_queries: List[str],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Synthesize structured response with sections.
        
        Args:
            original_query: Original user query
            sub_queries: List of sub-queries
            chunks: Retrieved chunks
            
        Returns:
            List of sections with title and content
        """
        prompt = self._build_structured_synthesis_prompt(original_query, sub_queries, chunks)
        
        # BUG FIX 1: Increased max_tokens to 4096 to prevent mid-sentence cutoff
        response = self.llm_client.generate_response(
            prompt=prompt,
            max_tokens=4096,
            temperature=0.3
        )
        
        # Parse sections from response
        sections = self._parse_sections(response)
        
        return sections
    
    def _build_simple_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Build prompt for simple query with enhanced instructions."""
        context_parts = []
        for i, chunk in enumerate(chunks[:8], 1):
            text = chunk.get('text', '')
            context_parts.append(f"[{i}] {text}")
        
        context_text = "\n\n".join(context_parts)
        
        # BUG FIX 1 & 4: Enhanced prompt to prevent cutoffs and ensure complete answers
        return f"""You are a research assistant. Answer the question based on the provided context.

Context:
{context_text}

Question: {query}

IMPORTANT INSTRUCTIONS:
- Provide a clear, comprehensive answer based on the context
- If the context doesn't contain enough information, explicitly state what is missing
- DO NOT cut off mid-sentence - complete every sentence fully before ending your response
- Use all relevant information from the context
- Be thorough but concise

Answer:"""
    
    def _build_structured_synthesis_prompt(
        self,
        query: str,
        sub_queries: List[str],
        chunks: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for structured synthesis with enhanced instructions."""
        context_parts = []
        for i, chunk in enumerate(chunks[:15], 1):
            text = chunk.get('text', '')
            source_query = chunk.get('source_sub_query', 'general')
            context_parts.append(f"[{i}] (Related to: {source_query})\n{text}")
        
        context_text = "\n\n".join(context_parts)
        
        sub_queries_text = "\n".join([f"{i+1}. {sq}" for i, sq in enumerate(sub_queries)])
        
        # BUG FIX 4: Enhanced structured prompt for comprehensive answers
        return f"""You are a research assistant providing a comprehensive answer to a complex question.

Original Question: {query}

This question was broken down into these aspects:
{sub_queries_text}

Context from research papers:
{context_text}

IMPORTANT INSTRUCTIONS:
- Provide a comprehensive, well-structured answer with clear sections
- Format your response using these sections:

## Key Aspects Found
### 1. [First aspect]
[Detailed explanation]

### 2. [Second aspect]
[Detailed explanation]

(Continue for all aspects found in the context)

## What Was Not Found
[Mention any aspects that were not covered in the papers]

## Summary
[Brief summary of the main findings]

CRITICAL REQUIREMENTS:
- Use ALL the context provided - synthesize information from multiple sources
- Create 3-5 main sections covering different aspects
- Use descriptive section titles (## Title)
- Be comprehensive but concise
- Base your answer ONLY on the provided context
- DO NOT cut off mid-sentence - complete every sentence fully before ending your response
- If information is missing, explicitly state what was not found

Answer:"""
    
    def _parse_sections(self, response: str) -> List[Dict[str, str]]:
        """
        Parse sections from structured response.
        
        Args:
            response: LLM response with sections
            
        Returns:
            List of section dicts with title and content
        """
        sections = []
        lines = response.strip().split('\n')
        
        current_title = None
        current_content = []
        
        for line in lines:
            # Check if line is a section header (## Title)
            if line.strip().startswith('##'):
                # Save previous section
                if current_title:
                    sections.append({
                        'title': current_title,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_title = line.strip().replace('##', '').strip()
                current_content = []
            else:
                # Add to current section content
                if current_title:
                    current_content.append(line)
        
        # Save last section
        if current_title:
            sections.append({
                'title': current_title,
                'content': '\n'.join(current_content).strip()
            })
        
        # If no sections found, create a single section
        if not sections:
            sections.append({
                'title': 'Answer',
                'content': response.strip()
            })
        
        return sections
