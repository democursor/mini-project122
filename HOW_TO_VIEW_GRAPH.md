# How to View Your Knowledge Graph

## 🌐 Method 1: Neo4j Browser (BEST - Visual & Interactive)

### Step 1: Open Browser
Open your web browser and go to:
```
http://localhost:7474
```

### Step 2: Login
- **Connect URL:** `neo4j://127.0.0.1:7687`
- **Database:** `miniproject`
- **Username:** `neo4j`
- **Password:** `Yadav2480@`

Click **"Connect"**

### Step 3: Run Queries

Copy and paste these queries into the query box at the top:

#### 🎨 **Visualize Everything:**
```cypher
MATCH (n) RETURN n LIMIT 100
```
*Shows all nodes and relationships as a graph visualization*

#### 📊 **Paper-Concept Network:**
```cypher
MATCH (p:Paper)-[:MENTIONS]->(c:Concept)
WHERE c.frequency >= 3
RETURN p, c
LIMIT 50
```
*Shows papers connected to their top concepts*

#### 🏆 **Top Concepts:**
```cypher
MATCH (c:Concept)
RETURN c.name AS Concept, 
       c.type AS Type, 
       c.frequency AS Mentions
ORDER BY c.frequency DESC
LIMIT 20
```
*Table view of most mentioned concepts*

#### 🔍 **Find COVID-19 Research:**
```cypher
MATCH (p:Paper)-[r:MENTIONS]->(c:Concept {name: "COVID-19"})
RETURN p.title AS Paper, 
       r.frequency AS Mentions
ORDER BY r.frequency DESC
```
*Papers that mention COVID-19*

#### 🔗 **Related Papers:**
```cypher
MATCH (p1:Paper)-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Paper)
WHERE p1 <> p2
WITH p1, p2, count(c) as shared
WHERE shared >= 5
RETURN p1.title AS Paper1, 
       p2.title AS Paper2, 
       shared AS SharedConcepts
ORDER BY shared DESC
LIMIT 10
```
*Papers that share many concepts*

#### 🕸️ **Concept Relationships:**
```cypher
MATCH (c1:Concept)-[r:RELATED_TO]-(c2:Concept)
WHERE r.strength > 0.5
RETURN c1, r, c2
LIMIT 50
```
*Concepts that frequently appear together*

---

## 💻 Method 2: Command Line Viewer

Run this in your terminal:
```bash
python view_graph.py
```

Shows:
- Graph statistics
- All papers
- Top concepts
- Related papers
- Papers by concept

---

## 🖥️ Method 3: Neo4j Desktop

1. Open **Neo4j Desktop** application
2. Find your database: `miniproject`
3. Click **"Open"** button (or the play icon)
4. Neo4j Browser opens automatically
5. Run the queries above

---

## 📈 What You'll See

### Your Current Graph:
- **3 Papers** (NBER Working Papers)
- **52 Concepts** (COVID-19, NBER, percentages, etc.)
- **127 Mentions** (paper-concept links)
- **1,326 Relationships** (concept-concept connections)

### Top Concepts:
1. COVID-19 (12 mentions)
2. NBER (7 mentions)
3. Various statistics (41%, 30%, etc.)

---

## 🎨 Visualization Tips

### In Neo4j Browser:

**Change Layout:**
- Click nodes to expand connections
- Drag nodes to rearrange
- Double-click to see properties

**Color Coding:**
- Different colors = different node types
- Papers = one color
- Concepts = another color

**Zoom & Pan:**
- Scroll to zoom
- Click and drag background to pan
- Click node to highlight connections

**View Properties:**
- Click any node
- See properties panel on right
- Shows: name, type, frequency, etc.

---

## 🔍 Useful Queries for Your Data

### Find Papers About Specific Topics:
```cypher
MATCH (p:Paper)-[:MENTIONS]->(c:Concept)
WHERE c.name CONTAINS "COVID"
RETURN p.title, collect(c.name) as concepts
```

### Most Connected Concepts:
```cypher
MATCH (c:Concept)-[r:RELATED_TO]-()
RETURN c.name, count(r) as connections
ORDER BY connections DESC
LIMIT 10
```

### Papers with Most Concepts:
```cypher
MATCH (p:Paper)-[:MENTIONS]->(c:Concept)
RETURN p.title, count(c) as concept_count
ORDER BY concept_count DESC
```

### Concept Co-occurrence:
```cypher
MATCH (c1:Concept {name: "COVID-19"})-[:RELATED_TO]-(c2:Concept)
RETURN c2.name, c2.frequency
ORDER BY c2.frequency DESC
LIMIT 10
```

---

## 📸 Screenshots Guide

### What You'll See:

1. **Graph View** - Nodes and edges visualization
   - Circles = Nodes (Papers, Concepts)
   - Lines = Relationships (MENTIONS, RELATED_TO)

2. **Table View** - Data in rows and columns
   - Click "Table" icon to switch
   - Good for statistics

3. **Text View** - Raw query results
   - Click "Text" icon
   - Shows JSON-like output

---

## 🚀 Quick Start

**Fastest way to see your graph:**

1. Open: http://localhost:7474
2. Login: neo4j / Yadav2480@
3. Run: `MATCH (n) RETURN n LIMIT 100`
4. Click the graph visualization
5. Explore!

---

## 💡 Pro Tips

- **Save Queries:** Click the star icon to save favorites
- **Export Data:** Click download icon for CSV/JSON
- **Full Screen:** Click expand icon for better view
- **Help:** Type `:help` in query box for Neo4j guide

---

## 🎯 Your Graph is Ready!

Open http://localhost:7474 now and explore your knowledge graph!

All your research papers, concepts, and relationships are visualized and queryable.
