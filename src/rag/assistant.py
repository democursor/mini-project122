"""Main RAG-based research assistant."""

import time
import logging
from typing import Dict, Any, List, Optional
from src.rag.models import RAGResponse, ConversationEntry
from src.rag.retriever import RAGRetriever
from src.rag.llm_client import LLMClient
from src.rag.prompt_template import RAGPromptTemplate
from src.rag.citation_extractor import CitationExtractor

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """Main RAG-based research assistant."""
    
    def __init__(self, retriever: RAGRetriever, llm_client: LLMClient):
        """
        Initialize research assistant.
        
        Args:
            retriever: RAG retriever instance
            llm_client: LLM client instance
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_template = RAGPromptTemplate()
        self.citation_extractor = CitationExtractor()
        self.conversation_history: List[ConversationEntry] = []
        logger.info("ResearchAssistant initialized")
    
    def ask_question(
        self, 
        question: str,
        conversation_history: list = None,
        top_k: int = 8,
        use_conversation_context: bool = False,
        use_graph: bool = False
    ) -> RAGResponse:
        """
        Answer research question using RAG pipeline.
        
        Pipeline:
        1. Retrieve relevant context
        2. Format prompt with context
        3. Generate LLM response
        4. Extract and validate citations
        5. Return structured response
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            use_conversation_context: Whether to include conversation history
            
        Returns:
            RAGResponse with answer and metadata
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Retrieve context with graph enhancement if available
            # Retrieve chunks — graph-enhanced if Neo4j available, else ChromaDB only
            if use_graph:
                try:
                    from src.graph.neo4j_client import Neo4jClient
                    from src.graph.searcher import GraphSearcher
                    neo4j_client = Neo4jClient()
                    graph_searcher = GraphSearcher(neo4j_client)
                    
                    # Get related concepts from graph to expand query
                    graph_concepts = []
                    try:
                        graph_concepts = graph_searcher.get_related_concepts(question, limit=5)
                    except Exception:
                        graph_concepts = []
                    
                    # Expand query with graph concepts for better retrieval
                    expanded_query = question
                    if graph_concepts:
                        concept_terms = " ".join([c.get('name', '') for c in graph_concepts])
                        expanded_query = f"{question} {concept_terms}"
                        logger.info(f"Query expanded with concepts: {concept_terms}")
                    
                    context = self.retriever.retrieve_context(expanded_query, top_k)
                    neo4j_client.close()
                except Exception as e:
                    logger.warning(f"Graph retrieval failed, falling back to ChromaDB: {e}")
                    context = self.retriever.retrieve_context(question, top_k)
            else:
                # ChromaDB only retrieval
                context = self.retriever.retrieve_context(question, top_k)
            
            # If no chunks found from either source
            if not context.retrieved_chunks:
                return RAGResponse(
                    answer="I couldn't find relevant information in the uploaded documents. Please try rephrasing your question.",
                    sources=[],
                    citations={'total_citations': 0, 'citation_accuracy': 0.0},
                    retrieval_stats={
                        "chunks_retrieved": 0,
                        "total_found": 0,
                        "retrieval_time": 0,
                        "context_length": 0
                    }
                )
            
            # Step 2: Format prompt
            if use_conversation_context and self.conversation_history:
                conversation_context = self.get_conversation_context()
                prompt = self.prompt_template.format_follow_up_prompt(
                    question, context.retrieved_chunks, conversation_context
                )
            else:
                prompt = self.prompt_template.format_research_prompt(
                    question, context.retrieved_chunks
                )
            
            # Step 3: Generate response
            response = self.llm_client.generate_response(prompt)
            
            # Step 4: Extract citations
            citations = self.citation_extractor.extract_citations(
                response, context.retrieved_chunks
            )
            
            # Step 5: Store in conversation history
            conversation_entry = ConversationEntry(
                question=question,
                response=response,
                context=context,
                citations=citations,
                timestamp=time.time()
            )
            self.conversation_history.append(conversation_entry)
            
            logger.info(f"Question answered successfully, citations={citations['total_citations']}, "
                       f"accuracy={citations['citation_accuracy']:.2%}")
            
            return RAGResponse(
                answer=response,
                sources=context.retrieved_chunks,
                citations=citations,
                retrieval_stats={
                    "chunks_retrieved": len(context.retrieved_chunks),
                    "total_found": context.total_chunks_found,
                    "retrieval_time": context.retrieval_time,
                    "context_length": context.context_length
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            raise
    
    def summarize_papers(self, paper_ids: Optional[List[str]] = None, top_k: int = 10) -> str:
        """
        Generate summary of papers.
        
        Args:
            paper_ids: Optional list of specific paper IDs to summarize
            top_k: Number of chunks to retrieve if paper_ids not specified
            
        Returns:
            Summary text
        """
        try:
            logger.info(f"Generating summary for {len(paper_ids) if paper_ids else 'all'} papers")
            
            # Retrieve context (either specific papers or general)
            if paper_ids:
                # TODO: Implement filtered retrieval by paper IDs
                context = self.retriever.retrieve_context("summarize all papers", top_k)
            else:
                context = self.retriever.retrieve_context("summarize all papers", top_k)
            
            # Format summarization prompt
            prompt = self.prompt_template.format_summarization_prompt(context.retrieved_chunks)
            
            # Generate summary
            summary = self.llm_client.generate_response(prompt, max_tokens=1500)
            
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True)
            raise
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """
        Get recent conversation context for follow-up questions.
        
        Args:
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            Formatted conversation context
        """
        recent_history = self.conversation_history[-max_turns:]
        
        context_parts = []
        for entry in recent_history:
            context_parts.append(f"Q: {entry.question}")
            context_parts.append(f"A: {entry.response}")
        
        return "\n\n".join(context_parts)
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_length(self) -> int:
        """Get number of conversation turns."""
        return len(self.conversation_history)
