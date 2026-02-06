# Phase 7: Data Persistence and Testing

## Overview

Phase 7 implements comprehensive data persistence strategies and testing frameworks to ensure data durability and system correctness. This phase covers database design, backup strategies, and both unit and property-based testing.

## Key Components

### 22.1 SQLite Schema Design

```sql
-- Complete database schema for research platform

-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT, -- JSON array
    year INTEGER,
    venue TEXT,
    abstract TEXT,
    doi TEXT,
    file_path TEXT NOT NULL,
    status TEXT DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    section_heading TEXT,
    start_sentence INTEGER,
    end_sentence INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Concepts table
CREATE TABLE concepts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    type TEXT, -- METHOD, DATASET, METRIC, etc.
    frequency INTEGER DEFAULT 1,
    first_seen DATE,
    last_seen DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document-Concept relationships
CREATE TABLE document_concepts (
    document_id TEXT,
    concept_id TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL,
    extraction_method TEXT,
    PRIMARY KEY (document_id, concept_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_documents_year ON documents(year);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_concepts_normalized ON concepts(normalized_name);
CREATE INDEX idx_doc_concepts_document ON document_concepts(document_id);
CREATE INDEX idx_doc_concepts_concept ON document_concepts(concept_id);
```

### 22.2 Data Persistence Manager

```python
import sqlite3
from typing import List, Dict, Any, Optional
import json

class DataPersistenceManager:
    """Manages all data persistence operations"""
    
    def __init__(self, db_path: str = "./data/research_platform.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Execute schema creation
            with open("schema.sql", "r") as f:
                conn.executescript(f.read())
    
    def store_document(self, document: ParsedDocument) -> bool:
        """Store parsed document in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, title, authors, year, venue, abstract, doi, file_path, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.document_id,
                    document.metadata.title,
                    json.dumps(document.metadata.authors),
                    document.metadata.year,
                    document.metadata.venue,
                    document.metadata.abstract,
                    document.metadata.doi,
                    document.file_path,
                    "processed"
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store document {document.document_id}: {e}")
            return False

### 22.3 Backup Strategy

class BackupManager:
    """Manages database backups and recovery"""
    
    def __init__(self, db_path: str, backup_dir: str = "./backups"):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self) -> str:
        """Create incremental backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}.db"
        
        # Copy database file
        shutil.copy2(self.db_path, backup_path)
        
        # Compress backup
        compressed_path = f"{backup_path}.gz"
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove uncompressed backup
        backup_path.unlink()
        
        return compressed_path

## Testing Strategy

### 23.1 Property-Based Testing

```python
from hypothesis import given, strategies as st
import hypothesis

class TestChunkingProperties:
    """Property-based tests for chunking functionality"""
    
    @given(st.text(min_size=100, max_size=10000))
    def test_chunking_preserves_content(self, text):
        """Property: Chunking should preserve all original content"""
        chunker = SemanticChunker()
        chunks = chunker.chunk(text)
        
        # Reconstruct text from chunks
        reconstructed = " ".join(chunk.text for chunk in chunks)
        
        # Content should be preserved (allowing for whitespace normalization)
        assert self._normalize_text(text) == self._normalize_text(reconstructed)
    
    @given(st.text(min_size=100, max_size=5000))
    def test_chunk_size_constraints(self, text):
        """Property: All chunks should meet size constraints"""
        chunker = SemanticChunker(min_tokens=50, max_tokens=200)
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            token_count = len(chunk.text.split())
            assert 50 <= token_count <= 200, f"Chunk size {token_count} violates constraints"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return " ".join(text.split())

### 23.2 Unit Testing

class TestConceptExtraction:
    """Unit tests for concept extraction"""
    
    def test_ner_extraction(self):
        """Test named entity recognition"""
        extractor = ConceptExtractor()
        text = "BERT is a transformer model developed by Google."
        
        entities = extractor.extract_entities(text)
        
        # Should extract BERT as METHOD and Google as ORG
        method_entities = [e for e in entities if e.label == "METHOD"]
        org_entities = [e for e in entities if e.label == "ORG"]
        
        assert len(method_entities) >= 1
        assert len(org_entities) >= 1
        assert any("bert" in e.text.lower() for e in method_entities)
        assert any("google" in e.text.lower() for e in org_entities)
    
    def test_keyphrase_extraction(self):
        """Test keyphrase extraction"""
        extractor = KeyphraseExtractor()
        text = "Deep learning models using neural networks have achieved state-of-the-art results."
        
        keyphrases = extractor.extract_keyphrases(text, top_k=5)
        
        assert len(keyphrases) <= 5
        assert all(kp.score > 0 for kp in keyphrases)
        assert any("deep learning" in kp.phrase.lower() for kp in keyphrases)

### 23.3 Integration Testing

class TestEndToEndPipeline:
    """Integration tests for complete pipeline"""
    
    def test_document_processing_pipeline(self):
        """Test complete document processing"""
        # Upload test document
        test_pdf = "test_data/sample_paper.pdf"
        
        # Process through pipeline
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_document(test_pdf)
        
        # Verify all stages completed
        assert result["status"] == "completed"
        assert "chunks" in result
        assert "concepts" in result
        assert len(result["chunks"]) > 0
        assert len(result["concepts"]) > 0
    
    def test_search_integration(self):
        """Test search functionality"""
        search_engine = SemanticSearchEngine()
        
        # Perform search
        results = search_engine.search("transformer neural networks")
        
        # Verify results
        assert len(results) > 0
        assert all("similarity_score" in r for r in results)
        assert all(r["similarity_score"] > 0 for r in results)

## Learning Outcomes

### Skills Learned in Phase 7

**1. Database Design**
- Relational schema design for complex data
- Index optimization for query performance
- Foreign key constraints and referential integrity

**2. Data Persistence**
- Transaction management and ACID properties
- Backup and recovery strategies
- Data migration and versioning

**3. Testing Methodologies**
- Property-based testing with Hypothesis
- Unit testing best practices
- Integration testing strategies

**4. Quality Assurance**
- Test coverage measurement
- Continuous integration setup
- Error detection and prevention

## Success Criteria

Phase 7 is successful when:

✅ **Data Persistence**
- All data is stored reliably in SQLite
- Referential integrity is maintained
- Backups can be created and restored

✅ **Testing Coverage**
- Unit tests cover core functionality
- Property-based tests validate invariants
- Integration tests verify end-to-end workflows

✅ **Data Quality**
- No data loss during operations
- Consistent data across system restarts
- Backup and recovery procedures work

---

**Phase 7 demonstrates database design, testing methodologies, and quality assurance - fundamental skills for building reliable data systems.**