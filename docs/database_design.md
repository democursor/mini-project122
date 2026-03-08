# Database Design Documentation

## Overview

The system uses two specialized databases to handle different aspects of data storage and retrieval:

1. **Neo4j** - Graph database for storing structured relationships
2. **ChromaDB** - Vector database for semantic search

This dual-database approach leverages the strengths of each technology to provide both structured querying and semantic search capabilities.

## Neo4j Knowledge Graph Design

### Graph Schema

The knowledge graph represents research papers, authors, concepts, and their relationships.

#### Node Types

**1. Paper Node**
```cypher
(:Paper {
  document_id: String,      // Unique identifier
  title: String,            // Paper title
  year: Integer,            // Publication year
  abstract: String,         // Paper abstract
  page_count: Integer,      // Number of pages
  upload_date: DateTime,    // When uploaded
  filename: String          // Original filename
})
```

**2. Author Node**
```cypher
(:Author {
  name: String,             // Author name (unique)
  paper_count: Integer      // Number of papers authored
})
```

**3. Concept Node**
```cypher
(:Concept {
  name: String,             // Concept name (unique)
  type: String,             // Type: keyphrase, topic, method
  mention_count: Integer    // Total mentions across papers
})
```

**4. Entity Node**
```cypher
(:Entity {
  text: String,             // Entity text
  label: String,            // NER label (PERSON, ORG, GPE, etc.)
  chunk_id: String,         // Source chunk
  document_id: String       // Source document
})
```

#### Relationship Types

**1. AUTHORED_BY**
```cypher
(Paper)-[:AUTHORED_BY]->(Author)
```
Connects papers to their authors.

**2. MENTIONS**
```cypher
(Paper)-[:MENTIONS {
  count: Integer,           // Number of mentions
  relevance: Float          // Relevance score
}]->(Concept)
```
Connects papers to concepts they discuss.

**3. CONTAINS**
```cypher
(Paper)-[:CONTAINS]->(Entity)
```
Connects papers to extracted entities.

**4. RELATED_TO**
```cypher
(Concept)-[:RELATED_TO {
  strength: Float           // Relationship strength
}]->(Concept)
```
Connects related concepts (co-occurrence based).

**5. CITES** (Future)
```cypher
(Paper)-[:CITES]->(Paper)
```
Citation relationships between papers.

### Graph Visualization

```
    ┌─────────┐
    │ Author  │
    │  John   │
    └────┬────┘
         │ AUTHORED_BY
         ▼
    ┌─────────┐         ┌──────────┐
    │  Paper  │─MENTIONS→│ Concept  │
    │ Paper A │         │Deep Learn│
    └────┬────┘         └────┬─────┘
         │                   │
         │ CONTAINS          │ RELATED_TO
         ▼                   ▼
    ┌─────────┐         ┌──────────┐
    │ Entity  │         │ Concept  │
    │ Neural  │         │   AI     │
    └─────────┘         └──────────┘
```

### Indexes

```cypher
// Primary indexes for fast lookups
CREATE INDEX paper_id FOR (p:Paper) ON (p.document_id);
CREATE INDEX author_name FOR (a:Author) ON (a.name);
CREATE INDEX concept_name FOR (c:Concept) ON (c.name);

// Composite indexes for complex queries
CREATE INDEX paper_year FOR (p:Paper) ON (p.year);
CREATE INDEX concept_type FOR (c:Concept) ON (c.type);
```

### Common Cypher Queries

**1. Find all papers by an author**
```cypher
MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author {name: $author_name})
RETURN p
ORDER BY p.year DESC
```

**2. Find papers mentioning a concept**
```cypher
MATCH (p:Paper)-[m:MENTIONS]->(c:Concept {name: $concept_name})
RETURN p, m.count as mentions
ORDER BY m.count DESC
LIMIT 10
```

**3. Find related papers (via shared concepts)**
```cypher
MATCH (p1:Paper {document_id: $doc_id})-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Paper)
WHERE p1 <> p2
WITH p2, COUNT(c) as shared_concepts
RETURN p2
ORDER BY shared_concepts DESC
LIMIT 10
```

**4. Find related concepts**
```cypher
MATCH (c1:Concept {name: $concept_name})-[:RELATED_TO]-(c2:Concept)
RETURN c2
ORDER BY c2.mention_count DESC
LIMIT 20
```

**5. Author collaboration network**
```cypher
MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
WHERE a1 <> a2
RETURN a1, a2, COUNT(p) as collaborations
ORDER BY collaborations DESC
```

**6. Most mentioned concepts**
```cypher
MATCH (c:Concept)
RETURN c.name, c.mention_count
ORDER BY c.mention_count DESC
LIMIT 50
```

**7. Papers by year distribution**
```cypher
MATCH (p:Paper)
WHERE p.year IS NOT NULL
RETURN p.year, COUNT(p) as paper_count
ORDER BY p.year DESC
```

**8. Delete paper and all relationships**
```cypher
MATCH (p:Paper {document_id: $doc_id})
DETACH DELETE p
```

### Graph Statistics

**Get database statistics**:
```cypher
// Count nodes by type
MATCH (n)
RETURN labels(n) as type, COUNT(n) as count

// Count relationships by type
MATCH ()-[r]->()
RETURN type(r) as relationship, COUNT(r) as count

// Database size
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store sizes")
YIELD attributes
RETURN attributes
```

## ChromaDB Vector Database Design

### Collection Schema

ChromaDB stores document chunks as vector embeddings with associated metadata.

**Collection Name**: `research_papers`

**Data Structure**:
```python
{
  "ids": ["doc_abc123_chunk_001", ...],
  "embeddings": [[0.123, -0.456, ...], ...],  # 384-dimensional vectors
  "documents": ["Chunk text content...", ...],
  "metadatas": [
    {
      "document_id": "doc_abc123",
      "chunk_id": "doc_abc123_chunk_001",
      "section_heading": "Introduction",
      "token_count": 487,
      "position": 0,
      "embedding_model": "all-MiniLM-L6-v2"
    },
    ...
  ]
}
```

### Embedding Model

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Specifications**:
- Dimensions: 384
- Max sequence length: 512 tokens
- Similarity metric: Cosine similarity
- Model size: ~90MB
- Performance: ~2000 sentences/second (CPU)

**Why this model?**:
- Lightweight and fast
- Good balance of speed and quality
- Suitable for semantic search
- Pre-trained on diverse text
- No GPU required

### Index Structure

ChromaDB uses **HNSW** (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search.

**HNSW Parameters**:
```python
{
  "hnsw:space": "cosine",           # Similarity metric
  "hnsw:construction_ef": 200,      # Construction time accuracy
  "hnsw:search_ef": 100,            # Search time accuracy
  "hnsw:M": 16                      # Number of connections per node
}
```

**Performance Characteristics**:
- Search time: O(log N)
- Memory usage: ~4KB per vector
- Recall@10: >95%
- Suitable for millions of vectors

### Storage Structure

```
data/chroma/
├── chroma.sqlite3                  # Metadata database
│   ├── Collections table
│   ├── Embeddings table
│   └── Metadata table
│
└── [collection-uuid]/              # Vector data
    ├── data_level0.bin             # Base layer vectors
    ├── header.bin                  # Index header
    ├── length.bin                  # Vector lengths
    └── link_lists.bin              # HNSW graph structure
```

### Query Operations

**1. Semantic Search**
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10,
    where={"document_id": {"$eq": "doc_abc123"}},  # Optional filter
    include=["documents", "metadatas", "distances"]
)
```

**2. Find Similar Documents**
```python
# Get embedding for a document chunk
source_embedding = collection.get(
    ids=["doc_abc123_chunk_001"],
    include=["embeddings"]
)

# Find similar chunks
results = collection.query(
    query_embeddings=source_embedding,
    n_results=10,
    where={"document_id": {"$ne": "doc_abc123"}}  # Exclude source doc
)
```

**3. Filter by Metadata**
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10,
    where={
        "$and": [
            {"section_heading": {"$eq": "Introduction"}},
            {"token_count": {"$gte": 200}}
        ]
    }
)
```

**4. Get All Chunks for a Document**
```python
chunks = collection.get(
    where={"document_id": {"$eq": "doc_abc123"}},
    include=["documents", "metadatas"]
)
```

**5. Delete Document Chunks**
```python
collection.delete(
    where={"document_id": {"$eq": "doc_abc123"}}
)
```

**6. Count Total Chunks**
```python
count = collection.count()
```

### Metadata Filtering

ChromaDB supports rich metadata filtering:

**Operators**:
- `$eq`: Equal
- `$ne`: Not equal
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$lt`: Less than
- `$lte`: Less than or equal
- `$in`: In list
- `$nin`: Not in list

**Logical Operators**:
- `$and`: Logical AND
- `$or`: Logical OR

**Example Complex Filter**:
```python
where={
    "$and": [
        {
            "$or": [
                {"section_heading": {"$eq": "Introduction"}},
                {"section_heading": {"$eq": "Conclusion"}}
            ]
        },
        {"token_count": {"$gte": 200}},
        {"document_id": {"$in": ["doc_1", "doc_2", "doc_3"]}}
    ]
}
```

## Database Comparison

| Feature | Neo4j | ChromaDB |
|---------|-------|----------|
| **Type** | Graph Database | Vector Database |
| **Primary Use** | Structured relationships | Semantic similarity |
| **Query Language** | Cypher | Python API |
| **Strengths** | Complex relationships, graph traversal | Fast similarity search |
| **Data Model** | Nodes and edges | Vectors and metadata |
| **Scalability** | Millions of nodes | Millions of vectors |
| **Query Speed** | Fast for graph queries | Fast for similarity search |
| **Storage** | ~1KB per node | ~4KB per vector |

## Data Consistency

### Cross-Database Consistency

Both databases store references to the same documents using `document_id` as the common identifier.

**Consistency Rules**:
1. Every document in Neo4j should have chunks in ChromaDB
2. Every chunk in ChromaDB should reference a valid document
3. Deletion must be coordinated across both databases

**Deletion Workflow**:
```python
# 1. Delete from ChromaDB
vector_store.delete_document_chunks(document_id)

# 2. Delete from Neo4j
graph_builder.delete_paper_node(document_id)

# 3. Delete from file system
storage.delete(document_id)

# 4. Update metadata
metadata.remove(document_id)
```

### Consistency Checks

**Check for orphaned chunks** (in ChromaDB but not Neo4j):
```python
# Get all document IDs from ChromaDB
chroma_docs = set(collection.get()["metadatas"]["document_id"])

# Get all document IDs from Neo4j
neo4j_docs = set(graph.run("MATCH (p:Paper) RETURN p.document_id"))

# Find orphans
orphaned = chroma_docs - neo4j_docs
```

**Check for missing embeddings** (in Neo4j but not ChromaDB):
```python
missing = neo4j_docs - chroma_docs
```

## Backup and Recovery

### Neo4j Backup

```bash
# Dump database
neo4j-admin dump --database=neo4j --to=/backup/neo4j-backup.dump

# Restore database
neo4j-admin load --from=/backup/neo4j-backup.dump --database=neo4j --force
```

### ChromaDB Backup

```bash
# Backup entire directory
cp -r data/chroma /backup/chroma-backup

# Restore
cp -r /backup/chroma-backup data/chroma
```

### Metadata Backup

```bash
# Backup metadata JSON
cp data/documents_metadata.json /backup/metadata-backup.json
```

## Performance Tuning

### Neo4j Optimization

**Memory Configuration** (`neo4j.conf`):
```
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
```

**Query Optimization**:
- Use indexes for frequently queried properties
- Limit result sets with `LIMIT`
- Use `PROFILE` to analyze query performance
- Avoid Cartesian products

### ChromaDB Optimization

**Batch Operations**:
```python
# Batch insert (faster than individual inserts)
collection.add(
    ids=chunk_ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=metadata_list
)
```

**Query Optimization**:
- Use metadata filters to reduce search space
- Adjust `n_results` based on needs
- Consider using `where_document` for text filtering

## Monitoring

### Neo4j Monitoring

```cypher
// Active queries
CALL dbms.listQueries()

// Database size
CALL apoc.meta.stats()

// Slow queries
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Transactions")
```

### ChromaDB Monitoring

```python
# Collection statistics
collection.count()
collection.get(limit=1)  # Check if accessible

# Disk usage
import os
chroma_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk('data/chroma')
    for filename in filenames
)
```

## Future Enhancements

### Neo4j
- Citation network analysis
- Author collaboration metrics
- Temporal analysis (paper trends over time)
- Full-text search integration
- Graph algorithms (PageRank, community detection)

### ChromaDB
- Multi-modal embeddings (text + images)
- Hybrid search (keyword + semantic)
- Query expansion
- Relevance feedback
- Custom distance metrics
