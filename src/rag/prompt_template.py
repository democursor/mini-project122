"""Prompt templates for RAG system."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RAGPromptTemplate:
    """Manages prompt templates for RAG system."""
    
    RESEARCH_ASSISTANT_TEMPLATE = """You are an expert AI research assistant specializing in academic literature analysis. Your role is to provide accurate, insightful answers based on the provided research context.

INSTRUCTIONS:
1. Answer the user's question using the provided context as your primary source
2. Use your knowledge to interpret and explain concepts even if exact terms aren't in the context
3. Make intelligent connections between related concepts (e.g., "COVID-19" relates to "pandemic", "coronavirus", "SARS-CoV-2", "infectious disease")
4. Cite specific papers using the format [Paper Title, Authors] when referencing specific findings
5. If the context contains related information but not exact matches, explain the connection
6. If truly no relevant information exists, clearly state this limitation
7. Provide nuanced, analytical responses that synthesize multiple sources

CONTEXT FROM RESEARCH PAPERS:
{context}

USER QUESTION: {question}

ANSWER (be comprehensive and make intelligent connections):"""

    SUMMARIZATION_TEMPLATE = """Provide a comprehensive summary of the key findings and contributions from the following research papers:

PAPERS:
{context}

Create a structured summary covering:
1. Main contributions and findings
2. Methodological approaches
3. Key results and implications
4. Areas of agreement and disagreement

SUMMARY:"""

    FOLLOW_UP_TEMPLATE = """You are an expert AI research assistant. Continue the conversation based on the previous context and the new question.

PREVIOUS CONVERSATION:
{conversation_history}

NEW CONTEXT:
{context}

NEW QUESTION: {question}

ANSWER:"""
    
    def format_research_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format prompt for research questions.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        context = self._format_context(context_chunks)
        prompt = self.RESEARCH_ASSISTANT_TEMPLATE.format(
            context=context,
            question=question
        )
        logger.debug(f"Formatted research prompt, length={len(prompt)}")
        return prompt
    
    def format_summarization_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format prompt for summarization.
        
        Args:
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        context = self._format_context(context_chunks)
        prompt = self.SUMMARIZATION_TEMPLATE.format(context=context)
        logger.debug(f"Formatted summarization prompt, length={len(prompt)}")
        return prompt
    
    def format_follow_up_prompt(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]], 
        conversation_history: str
    ) -> str:
        """
        Format prompt for follow-up questions.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            conversation_history: Previous conversation context
            
        Returns:
            Formatted prompt string
        """
        context = self._format_context(context_chunks)
        prompt = self.FOLLOW_UP_TEMPLATE.format(
            conversation_history=conversation_history,
            context=context,
            question=question
        )
        logger.debug(f"Formatted follow-up prompt, length={len(prompt)}")
        return prompt
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            title = metadata.get("title", "Unknown Title")
            authors = metadata.get("authors", ["Unknown Authors"])
            year = metadata.get("year", "Unknown Year")
            
            # Format authors list
            if isinstance(authors, list):
                authors_str = ", ".join(authors)
            else:
                authors_str = str(authors)
            
            chunk_text = f"""[{i}] {title} ({authors_str}, {year})
{chunk["text"]}
"""
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)
