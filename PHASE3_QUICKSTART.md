# Phase 3 Quick Start Guide

## Prerequisites
- Phase 1 & 2 completed (parsed data in `data/parsed/`)
- Neo4j installed and running

## Setup (5 minutes)

### 1. Start Neo4j
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

### 2. Install Dependencies
```bash
pip install neo4j
```

### 3. Build Graph
```bash
python build_graph.py
```

## Verify Installation

### Open Neo4j Browser
http://localhost:7474

**Login:** neo4j / password

### Run Test Query
```cypher
MATCH (p:Paper)-[:MENTIONS]->(c:Concept)
RETURN p.title, c.name
LIMIT 10
```

## Quick Commands

### Build graph from existing data
```bash
python build_graph.py
```

### Run tests
```bash
python test_phase3.py
```

### Process new PDF (full pipeline)
```bash
python main.py
```

## Useful Cypher Queries

### Top concepts
```cypher
MATCH (c:Concept)
RETURN c.name, c.frequency
ORDER BY c.frequency DESC
LIMIT 20
```

### Find related papers
```cypher
MATCH (p1:Paper)-[:MENTIONS]->(c)<-[:MENTIONS]-(p2:Paper)
WHERE p1.id = "doc_abc123"
RETURN p2.title, count(c) as shared
ORDER BY shared DESC
```

### Visualize network
```cypher
MATCH (p:Paper)-[:MENTIONS]->(c:Concept)
WHERE c.frequency >= 3
RETURN p, c
LIMIT 50
```

## Troubleshooting

**Neo4j not connecting?**
```bash
docker ps  # Check if running
docker logs neo4j  # Check logs
```

**No data in graph?**
```bash
python build_graph.py  # Rebuild from parsed data
```

**Clear graph and start over?**
```cypher
MATCH (n) DETACH DELETE n
```

## Next Steps
- Explore graph in Neo4j Browser
- Try different Cypher queries
- Visualize paper-concept networks
- Ready for Phase 4!
