import logging
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.ingestion import PDFValidator, PDFStorage, PDFUploader
from src.parsing import PDFParser
from src.chunking import SemanticChunker, ChunkingConfig
from src.extraction import ConceptExtractor
from src.graph import KnowledgeGraphBuilder
from src.vector import EmbeddingGenerator, EmbeddingConfig, VectorStore
from src.vector.models import ChunkEmbedding

logger = logging.getLogger(__name__)


class ProcessingState(TypedDict):
    document_id: str
    file_path: Optional[Path]
    status: str
    error_message: Optional[str]
    parsed_data: Optional[dict]
    chunks: Optional[list]
    concepts: Optional[list]
    graph_built: Optional[bool]
    embeddings_stored: Optional[bool]


class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.validator = PDFValidator(config.get('storage.max_file_size_mb', 50))
        self.storage = PDFStorage(config.get('storage.pdf_directory', './data/pdfs'))
        self.uploader = PDFUploader(self.validator, self.storage)
        self.parser = PDFParser()
        self.chunker = SemanticChunker(ChunkingConfig())
        self.extractor = ConceptExtractor()
        
        # Initialize graph builder if Neo4j config exists
        self.graph_builder = None
        if config.get('neo4j'):
            try:
                self.graph_builder = KnowledgeGraphBuilder(
                    uri=config.get('neo4j.uri', 'bolt://localhost:7687'),
                    user=config.get('neo4j.user', 'neo4j'),
                    password=config.get('neo4j.password', 'password'),
                    database=config.get('neo4j.database', 'neo4j')
                )
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {e}")
        
        # Initialize vector store and embedding generator
        self.embedding_generator = EmbeddingGenerator(EmbeddingConfig())
        self.vector_store = VectorStore(config.get('vector.persist_directory', './data/chroma'))
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        workflow = StateGraph(ProcessingState)
        
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("chunk", self._chunk_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("build_graph", self._build_graph_node)
        workflow.add_node("store_vectors", self._store_vectors_node)
        workflow.add_node("complete", self._complete_node)
        workflow.add_node("error", self._error_node)
        
        workflow.set_entry_point("parse")
        
        workflow.add_conditional_edges(
            "parse",
            self._check_parse_result,
            {
                "success": "chunk",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "chunk",
            self._check_chunk_result,
            {
                "success": "extract",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "extract",
            self._check_extract_result,
            {
                "success": "build_graph",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "build_graph",
            self._check_graph_result,
            {
                "success": "store_vectors",
                "skip": "store_vectors",
                "error": "error"
            }
        )
        
        workflow.add_conditional_edges(
            "store_vectors",
            self._check_vector_result,
            {
                "success": "complete",
                "error": "error"
            }
        )
        
        workflow.add_edge("complete", END)
        workflow.add_edge("error", END)
        
        return workflow.compile()
    
    def _parse_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            file_path = self.storage.retrieve(doc_id)
            
            parsed = self.parser.parse(file_path, doc_id)
            
            state["parsed_data"] = parsed.to_dict()
            state["status"] = "parsed"
            logger.info(f"Parsed document: {doc_id}")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Parse error: {str(e)}"
            logger.error(f"Parse error: {e}")
        
        return state
    
    def _chunk_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            parsed_data = state["parsed_data"]
            
            all_chunks = []
            for section in parsed_data.get('sections', []):
                chunks = self.chunker.chunk_document(
                    doc_id,
                    section['content'],
                    section['heading']
                )
                all_chunks.extend([c.to_dict() for c in chunks])
            
            state["chunks"] = all_chunks
            state["status"] = "chunked"
            logger.info(f"Created {len(all_chunks)} chunks")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Chunking error: {str(e)}"
            logger.error(f"Chunking error: {e}")
        
        return state
    
    def _extract_node(self, state: ProcessingState) -> ProcessingState:
        try:
            doc_id = state["document_id"]
            chunks = state["chunks"]
            
            all_concepts = []
            for chunk in chunks[:5]:  # Limit to first 5 chunks for demo
                result = self.extractor.extract(
                    chunk['chunk_id'],
                    doc_id,
                    chunk['text']
                )
                all_concepts.append(result.to_dict())
            
            state["concepts"] = all_concepts
            state["status"] = "extracted"
            logger.info(f"Extracted concepts from {len(all_concepts)} chunks")
            
        except Exception as e:
            state["status"] = "error"
            state["error_message"] = f"Extraction error: {str(e)}"
            logger.error(f"Extraction error: {e}")
        
        return state
    
    def _build_graph_node(self, state: ProcessingState) -> ProcessingState:
        """Build knowledge graph from extracted data"""
        try:
            if not self.graph_builder:
                state["graph_built"] = False
                state["status"] = "graph_skipped"
                logger.info("Graph builder not available, skipping graph construction")
                return state
            
            # Build complete data structure
            graph_data = {
                'document_id': state['document_id'],
                'parsed_data': state['parsed_data'],
                'concepts': state['concepts']
            }
            
            # Build graph
            success = self.graph_builder.build_from_parsed_data(graph_data)
            
            if success:
                state["graph_built"] = True
                state["status"] = "graph_built"
                logger.info(f"Built knowledge graph for {state['document_id']}")
            else:
                state["graph_built"] = False
                state["status"] = "graph_failed"
                logger.warning("Graph construction failed")
            
        except Exception as e:
            state["graph_built"] = False
            state["status"] = "graph_error"
            state["error_message"] = f"Graph error: {str(e)}"
            logger.error(f"Graph construction error: {e}")
        
        return state
    
    def _complete_node(self, state: ProcessingState) -> ProcessingState:
        state["status"] = "complete"
        logger.info(f"Processing complete: {state['document_id']}")
        return state
    
    def _error_node(self, state: ProcessingState) -> ProcessingState:
        logger.error(f"Processing failed: {state.get('error_message')}")
        return state
    
    def _check_parse_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "parsed" else "error"
    
    def _check_chunk_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "chunked" else "error"
    
    def _check_extract_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "extracted" else "error"
    
    def _check_graph_result(self, state: ProcessingState) -> str:
        if state["status"] == "graph_built":
            return "success"
        elif state["status"] == "graph_skipped":
            return "skip"
        else:
            return "error"
    
    def _store_vectors_node(self, state: ProcessingState) -> ProcessingState:
        """Store chunk embeddings in vector database"""
        try:
            chunks = state["chunks"]
            doc_id = state["document_id"]
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            print(f"  Generating embeddings for {len(chunks)} chunks...")
            
            # Create ChunkEmbedding objects
            chunk_embeddings = []
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batch (this is the slow part)
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk['chunk_id'],
                    document_id=doc_id,
                    text=chunk['text'],
                    embedding=embedding,
                    embedding_model=self.embedding_generator.config.model_name,
                    section_heading=chunk.get('section_heading')
                )
                chunk_embeddings.append(chunk_embedding)
            
            # Store in vector database
            logger.info(f"Storing {len(chunk_embeddings)} embeddings in ChromaDB...")
            print(f"  Storing embeddings in vector database...")
            self.vector_store.add_embeddings(chunk_embeddings)
            
            state["embeddings_stored"] = True
            state["status"] = "vectors_stored"
            logger.info(f"Stored {len(chunk_embeddings)} embeddings for {doc_id}")
            
        except Exception as e:
            state["embeddings_stored"] = False
            state["status"] = "vector_error"
            state["error_message"] = f"Vector storage error: {str(e)}"
            logger.error(f"Vector storage error: {e}")
        
        return state
    
    def _check_vector_result(self, state: ProcessingState) -> str:
        return "success" if state["status"] == "vectors_stored" else "error"
    
    def process_document(self, document_id: str) -> ProcessingState:
        initial_state = ProcessingState(
            document_id=document_id,
            file_path=None,
            status="uploaded",
            error_message=None,
            parsed_data=None,
            chunks=None,
            concepts=None,
            graph_built=None,
            embeddings_stored=None
        )
        
        result = self.workflow.invoke(initial_state)
        return result
