# Phase 5: RAG and AI Research Assistant

## Overview

Phase 5 implements the AI Research Assistant using Retrieval-Augmented Generation (RAG). This phase combines the semantic search capabilities from Phase 4 with large language models to create an intelligent conversational interface for research discovery and analysis.

**Learning Objectives:**
- Understand RAG (Retrieval-Augmented Generation) architecture
- Learn prompt engineering for research contexts
- Master LLM integration (OpenAI API and local models)
- Implement conversation management and context handling
- Build citation extraction and verification systems

**Key Concepts:**
- RAG pipeline: Retrieve → Augment → Generate
- Prompt engineering and template design
- Context window management
- Citation extraction and linking
- Conversation state management
- Multi-turn dialogue handling

---

## RAG Module Design

### 17.1 Retrieval Component

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RAGContext:
    """Context retrieved for RAG generation"""
    query: str
    retrieved_chunks: List[Dict]
    total_chunks_found: int
    retrieval_time: float
    context_length: int

class RAGRetriever:
    """Retrieval component for RAG system"""
    
    def __init__(self, search_engine: SemanticSearchEngine):
        self.search_engine = search_engine
        self.max_context_tokens = 3000  # Leave room for query and response
    
    def retrieve_context(self, query: str, top_k: int = 5) -> RAGContext:
        """
        Retrieve relevant context for RAG generation.
        
        Strategy:
        1. Perform semantic search
        2. Select diverse, high-quality chunks
        3. Ensure context fits within token limits
        4. Format for LLM consumption
        """
        import time
        start_time = time.time()
        
        # Retrieve more candidates for diversity
        candidates = self.search_engine.search(query, top_k * 2)
        
        # Select diverse, high-quality chunks
        selected_chunks = self._select_diverse_chunks(candidates, top_k)
        
        # Ensure context fits token limits
        context_chunks = self._fit_context_to_limits(selected_chunks)
        
        retrieval_time = time.time() - start_time
        context_length = sum(len(chunk["text"]) for chunk in context_chunks)
        
        return RAGContext(
            query=query,
            retrieved_chunks=context_chunks,
            total_chunks_found=len(candidates),
            retrieval_time=retrieval_time,
            context_length=context_length
        )
    
    def _select_diverse_chunks(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Select diverse chunks to avoid redundancy"""
        selected = []
        used_documents = set()
        
        for chunk in candidates:
            doc_id = chunk["metadata"]["document_id"]
            
            # Limit chunks per document for diversity
            if used_documents.count(doc_id) < 2:
                selected.append(chunk)
                used_documents.add(doc_id)
                
                if len(selected) >= top_k:
                    break
        
        return selected
    
    def _fit_context_to_limits(self, chunks: List[Dict]) -> List[Dict]:
        """Ensure context fits within token limits"""
        fitted_chunks = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk["text"].split()) * 1.3  # Rough token estimate
            
            if current_tokens + chunk_tokens <= self.max_context_tokens:
                fitted_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        return fitted_chunks

### 17.2 LLM Integration

class LLMClient:
    """Handles LLM API calls with fallback options"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            import openai
            return openai.OpenAI()
        elif self.provider == "ollama":
            # Local LLM via Ollama
            import requests
            return requests.Session()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using configured LLM"""
        if self.provider == "openai":
            return self._openai_generate(prompt, max_tokens)
        elif self.provider == "ollama":
            return self._ollama_generate(prompt, max_tokens)
    
    def _openai_generate(self, prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _ollama_generate(self, prompt: str, max_tokens: int) -> str:
        """Generate using local Ollama"""
        response = self.client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
        )
        return response.json()["response"]

### 17.3 Prompt Engineering

class RAGPromptTemplate:
    """Manages prompt templates for RAG system"""
    
    RESEARCH_ASSISTANT_TEMPLATE = """You are an expert AI research assistant specializing in academic literature analysis. Your role is to provide accurate, well-cited answers based on the provided research context.

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context
2. Cite specific papers using the format [Paper Title, Authors]
3. If information is not in the context, clearly state this limitation
4. Provide nuanced, analytical responses that synthesize multiple sources
5. Highlight conflicting viewpoints when they exist

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    SUMMARIZATION_TEMPLATE = """Provide a comprehensive summary of the key findings and contributions from the following research papers:

PAPERS:
{context}

Create a structured summary covering:
1. Main contributions and findings
2. Methodological approaches
3. Key results and implications
4. Areas of agreement and disagreement

SUMMARY:"""

    def format_research_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Format prompt for research questions"""
        context = self._format_context(context_chunks)
        return self.RESEARCH_ASSISTANT_TEMPLATE.format(
            context=context,
            question=question
        )
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            title = metadata.get("title", "Unknown Title")
            authors = metadata.get("authors", ["Unknown Authors"])
            year = metadata.get("year", "Unknown Year")
            
            chunk_text = f"""
[{i}] {title} ({", ".join(authors)}, {year})
{chunk["text"]}
"""
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)

### 17.4 Citation Extraction

class CitationExtractor:
    """Extracts and validates citations from LLM responses"""
    
    def __init__(self):
        self.citation_pattern = r'\[([^\]]+)\]'
    
    def extract_citations(self, response: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Extract citations from LLM response and validate against context.
        
        Returns:
            Dictionary with extracted citations and validation results
        """
        import re
        
        # Find all citations in response
        citations = re.findall(self.citation_pattern, response)
        
        # Validate citations against context
        validated_citations = []
        for citation in citations:
            validation = self._validate_citation(citation, context_chunks)
            validated_citations.append(validation)
        
        return {
            "total_citations": len(citations),
            "citations": validated_citations,
            "citation_accuracy": sum(1 for c in validated_citations if c["valid"]) / len(citations) if citations else 0
        }
    
    def _validate_citation(self, citation: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Validate a single citation against context"""
        for chunk in context_chunks:
            metadata = chunk["metadata"]
            title = metadata.get("title", "").lower()
            authors = [a.lower() for a in metadata.get("authors", [])]
            
            citation_lower = citation.lower()
            
            # Check if citation matches title or authors
            title_match = any(word in title for word in citation_lower.split() if len(word) > 3)
            author_match = any(author_name in citation_lower for author_name in authors)
            
            if title_match or author_match:
                return {
                    "citation": citation,
                    "valid": True,
                    "matched_paper": {
                        "title": metadata.get("title"),
                        "authors": metadata.get("authors"),
                        "document_id": metadata.get("document_id")
                    }
                }
        
        return {
            "citation": citation,
            "valid": False,
            "matched_paper": None
        }

### 17.5 Complete RAG System

class ResearchAssistant:
    """Main RAG-based research assistant"""
    
    def __init__(self, retriever: RAGRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_template = RAGPromptTemplate()
        self.citation_extractor = CitationExtractor()
        self.conversation_history = []
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Answer research question using RAG pipeline.
        
        Pipeline:
        1. Retrieve relevant context
        2. Format prompt with context
        3. Generate LLM response
        4. Extract and validate citations
        5. Return structured response
        """
        # Step 1: Retrieve context
        context = self.retriever.retrieve_context(question)
        
        # Step 2: Format prompt
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
        conversation_entry = {
            "question": question,
            "response": response,
            "context": context,
            "citations": citations,
            "timestamp": time.time()
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            "answer": response,
            "sources": context.retrieved_chunks,
            "citations": citations,
            "retrieval_stats": {
                "chunks_retrieved": len(context.retrieved_chunks),
                "total_found": context.total_chunks_found,
                "retrieval_time": context.retrieval_time
            }
        }
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """Get recent conversation context for follow-up questions"""
        recent_history = self.conversation_history[-max_turns:]
        
        context_parts = []
        for entry in recent_history:
            context_parts.append(f"Q: {entry['question']}")
            context_parts.append(f"A: {entry['response']}")
        
        return "\n\n".join(context_parts)
---

## Learning Outcomes

### Skills Learned in Phase 5

**1. RAG Architecture**
- Retrieval-Augmented Generation pipeline design
- Context retrieval and selection strategies
- Integration of search and generation components

**2. LLM Integration**
- OpenAI API and local LLM usage
- Prompt engineering for research contexts
- Context window management and optimization

**3. Conversation Management**
- Multi-turn dialogue handling
- Context preservation across conversations
- Follow-up question processing

**4. Citation Systems**
- Citation extraction from generated text
- Validation against source documents
- Academic citation formatting

**5. Production Considerations**
- Error handling for LLM failures
- Cost optimization for API usage
- Response quality monitoring

---

## Success Criteria

Phase 5 is successful when:

✅ **RAG Pipeline**
- Retrieval provides relevant context
- LLM generates accurate, grounded responses
- Citations are extracted and validated

✅ **Conversation Quality**
- Responses are coherent and helpful
- Citations link to actual source documents
- Follow-up questions work correctly

✅ **System Integration**
- RAG system integrates with search engine
- Multiple LLM providers are supported
- Error handling prevents system crashes

---

**Phase 5 demonstrates advanced AI system integration including RAG, prompt engineering, and conversational AI - highly valued skills for AI product development roles.**