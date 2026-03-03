# Phase 3: Knowledge Graph Construction - Starting Point

## Project Status

✅ **Phase 1 Complete**: PDF Ingestion and Parsing
- PDF validation, storage, and text extraction working
- Documents stored in `data/pdfs/YYYY/MM/`
- Parsed data saved to `data/parsed/`

✅ **Phase 2 Complete**: Semantic Chunking and Concept Extraction
- Semantic chunking with sentence-transformers
- Named entity recognition with SpaCy
- Keyphrase extraction with KeyBERT
- Results stored in JSON files with chunks and concepts

## Phase 3 Goal

Build a **Knowledge Graph** using Neo4j to represent relationships between:
- Papers (documents)
- Concepts (entities and keyphrases)
- Authors
- Venues

## What to Implement

### 1. Graph Database Setup
- Install Neo4j (Desktop or Docker)
- Configure connection
- Create database schema

### 2. Create Module Structure
```
src/graph/
├── __init__.py
├── builder.py      # Creates nodes and relationships
├── models.py       # Graph data models
├── queries.py      # Query patterns
└── analytics.py    # Trend analysis (optional)
```

### 3. Node Types to Create

**Paper Nodes**
```python
{
  "id": "doc_abc123",
  "title": "Paper Title",
  "abstract": "Abstract text",
  "year": 2024,
  "page_count": 15
}
```

**Concept Nodes**
```python
{
  "id": "concept_transformer",
  "name": "transformer",
  "normalized_name": "transformer",
  "type": "METHOD",  # From entity.label
  "frequency": 5     # Count of papers mentioning it
}
```

### 4. Relationships to Create

- `(Paper)-[:MENTIONS]->(Concept)` - Paper mentions a concept
- `(Concept)-[:RELATED_TO]->(Concept)` - Concepts co-occur in papers

### 5. Key Functions Needed

```python
class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        # Connect to Neo4j
        
    def create_paper_node(self, document):
        # Create paper node from ParsedDocument
        
    def create_concept_nodes(self, extraction_results):
        # Create concept nodes from entities/keyphrases
        
    def create_relationships(self, document_id, extraction_results):
        # Link papers to concepts
        
    def find_related_papers(self, paper_id):
        # Query: Find papers with shared concepts
```

## Data Sources

**Input Data Location:**
- Parsed documents: `data/parsed/*.json`
- Each JSON contains:
  - `parsed_data`: Document metadata
  - `chunks`: Semantic chunks
  - `concepts`: Extracted entities and keyphrases

**Example JSON Structure:**
```json
{
  "document_id": "doc_abc123",
  "parsed_data": {
    "metadata": {
      "title": "...",
      "authors": ["..."],
      "year": 2024
    }
  },
  "chunks": [...],
  "concepts": [
    {
      "chunk_id": "...",
      "entities": [
        {
          "text": "transformer",
          "label": "METHOD",
          "confidence": 0.95
        }
      ],
      "keyphrases": [
        {
          "phrase": "attention mechanism",
          "score": 0.85
        }
      ]
    }
  ]
}
```

## Implementation Steps

### Step 1: Install Neo4j
```bash
# Option 1: Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# Option 2: Neo4j Desktop
# Download from https://neo4j.com/download/
```

### Step 2: Install Python Driver
```bash
pip install neo4j
```

### Step 3: Create Graph Builder
- Read parsed JSON files
- Extract paper metadata
- Extract concepts (entities + keyphrases)
- Create nodes in Neo4j
- Create relationships

### Step 4: Test Queries
```cypher
// Find all papers
MATCH (p:Paper) RETURN p LIMIT 10

// Find all concepts
MATCH (c:Concept) RETURN c.name, c.frequency ORDER BY c.frequency DESC

// Find papers mentioning "transformer"
MATCH (p:Paper)-[:MENTIONS]->(c:Concept {name: "transformer"})
RETURN p.title

// Find related papers
MATCH (p1:Paper)-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Paper)
WHERE p1.id = "doc_abc123"
RETURN p2.title, count(c) as shared_concepts
ORDER BY shared_concepts DESC
```

## Success Criteria

✅ Neo4j database running
✅ Paper nodes created from parsed documents
✅ Concept nodes created from entities/keyphrases
✅ MENTIONS relationships created
✅ Can query related papers
✅ Can find popular concepts
✅ Graph visualizable in Neo4j Browser

## Configuration

Add to `config/default.yaml`:
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  database: "research"
```

## Testing

Create `test_phase3.py`:
```python
# Test Neo4j connection
# Test node creation
# Test relationship creation
# Test queries
```

## Next Steps After Phase 3

- Phase 4: Vector Storage & Semantic Search
- Phase 5: RAG & AI Assistant
- Phase 6: Orchestration & Error Handling

---

## Important Notes

- Keep implementation MINIMAL - focus on core functionality
- Use existing parsed data from Phase 1 & 2
- Don't rebuild parsing/chunking - just read JSON files
- Start with simple queries, add complexity later
- Test with small dataset first (2-3 documents)

## Quick Start Command

```bash
# 1. Start Neo4j
docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# 2. Install driver
pip install neo4j

# 3. Run graph builder
python -m src.graph.builder

# 4. Open Neo4j Browser
# http://localhost:7474
```

---

**Ready to implement Phase 3!** 🚀
