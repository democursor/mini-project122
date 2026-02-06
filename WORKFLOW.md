# Workflow & Data Flow: Autonomous Research Literature Intelligence & Discovery Platform

## Overview

This document explains the end-to-end data flow, execution lifecycle, state transitions, and error handling in the platform. Understanding workflow is crucial for debugging, optimization, and system design interviews.

---

## Table of Contents

1. [End-to-End Data Flow](#end-to-end-data-flow)
2. [Document Processing Lifecycle](#document-processing-lifecycle)
3. [State Management](#state-management)
4. [Error and Failure Flow](#error-and-failure-flow)
5. [Query Workflows](#query-workflows)
6. [Data Transformations](#data-transformations)
7. [Interview Preparation](#interview-preparation)

---

## End-to-End Data Flow

### High-Level Flow

```
📄 User uploads PDF
  ↓
🔍 Validate & Store
  ↓
📖 Parse text & metadata
  ↓
✂️ Chunk semantically
  ↓
🧠 Extract concepts
  ↓
[Parallel Processing]
  ├─> 🕸️ Build knowledge graph
  └─> 🔢 Generate embeddings
  ↓
✅ Mark complete
  ↓
🔍 Enable search & chat
```

### Detailed Flow with Data Transformations

```
Stage 1: Upload
  Input:  PDF file (binary)
  Output: document_id, file_path, metadata_record
  
Stage 2: Parse
  Input:  file_path
  Output: {text, metadata, sections}
  
Stage 3: Chunk
  Input:  {text, sections}
  Output: [{chunk_text, position, token_count}]
  
Stage 4: Extract Concepts
  Input:  [{chunk_text}]
  Output: [{entities, keyphrases, relationships}]
  
Stage 5a: Build Graph
  Input:  {metadata, concepts}
  Output: Graph nodes & edges
  
Stage 5b: Generate Embeddings
  Input:  [{chunk_text}]
  Output: [{embedding_vector, chunk_id}]
  
Stage 6: Complete
  Input:  document_id
  Output: status = "complete"
```

---

## Document Processing Lifecycle

### State Diagram

```

                    ┌─────────┐
                    │ UPLOADED│
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ PARSING │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │CHUNKING │
                    └────┬────┘
                         │
                    ┌────▼────────┐
                    │ EXTRACTING  │
                    └────┬────────┘
                         │
                ┌────────┴────────┐
                │                 │
         ┌──────▼──────┐   ┌─────▼──────┐
         │BUILDING_GRAPH│   │EMBEDDING   │
         └──────┬──────┘   └─────┬──────┘
                │                 │
                └────────┬────────┘
                         │
                    ┌────▼────┐
                    │COMPLETE │
                    └─────────┘
                    
                    [Any stage can transition to FAILED]
                         │
                    ┌────▼────┐
                    │ FAILED  │
                    └─────────┘
```

### Detailed Step-by-Step Execution

#### Step 1: Upload & Validation

**Trigger**: User uploads PDF file

**Operations**:
1. Receive file from user
2. Validate file format (check magic bytes for PDF signature)
3. Validate file size (< 50MB)
4. Generate unique document_id (UUID)
5. Store PDF in `data/pdfs/{document_id}.pdf`
6. Create metadata record in SQLite
7. Set status = "UPLOADED"
8. Trigger LangGraph workflow
9. Return document_id to user

**Data Created**:
```sql
INSERT INTO documents (id, filename, upload_date, status)
VALUES ('doc_123', 'attention_paper.pdf', '2024-01-15', 'UPLOADED');
```

**Error Conditions**:
- Invalid format → Reject, return error
- File too large → Reject, return error
- Storage failure → Retry 3x, then fail

**Duration**: < 1 second

---

#### Step 2: Parsing

**Trigger**: LangGraph workflow starts

**Operations**:
1. Update status = "PARSING"
2. Load PDF from file storage
3. Extract text page by page using PyMuPDF
4. Identify document structure:
   - Title (first large text, often bold)
   - Authors (below title, specific format)
   - Abstract (section labeled "Abstract")
   - Sections (headings + content)
   - References (section labeled "References")
5. Extract metadata (title, authors, year, venue)
6. Create structured JSON output
7. Store parsed data in state
8. Update SQLite with metadata

**Data Created**:
```json
{
  "document_id": "doc_123",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer", "..."],
    "year": 2017,
    "venue": "NeurIPS",
    "abstract": "The dominant sequence transduction models..."
  },
  "sections": [
    {
      "heading": "Introduction",
      "text": "Recurrent neural networks...",
      "position": 0,
      "page": 1
    },
    {
      "heading": "Background",
      "text": "The goal of reducing...",
      "position": 1,
      "page": 2
    }
  ],
  "total_pages": 15,
  "word_count": 7234
}
```

**Error Conditions**:
- Corrupted PDF → Log error, mark FAILED
- Extraction failure → Retry with different method
- Missing metadata → Continue with partial data

**Duration**: 10-30 seconds (depends on PDF size)

---

#### Step 3: Semantic Chunking

**Trigger**: Parsing completes successfully

**Operations**:
1. Update status = "CHUNKING"
2. Load parsed document from state
3. Split text into sentences
4. Generate sentence embeddings using Sentence-BERT
5. Compute cosine similarity between consecutive sentences
6. Identify boundaries where similarity drops below threshold (e.g., 0.7)
7. Respect section boundaries as hard boundaries
8. Create chunks ensuring 100-500 tokens each
9. Store chunks in SQLite with references to document

**Algorithm Details**:
```python
def chunk_document(sections: List[Section]) -> List[Chunk]:
    chunks = []
    for section in sections:
        sentences = split_into_sentences(section.text)
        embeddings = model.encode(sentences)
        
 
       chunks in state
   - Stores chunks in SQLite
   - Updates status = "chunked"

5. **Concept Extractor identifies entities**
   - Reads chunks from state
   - Runs SpaCy NER on each chunk
   - Runs KeyBERT for keyphrases
   - Stores concepts in state
   - Updates status = "extracting"

6. **Parallel processing: Graph + Embeddings**
   - **Graph Builder**:
     - Creates Paper node in Neo4j
     - Creates Concept nodes
     - Creates MENTIONS relationships
     - Updates status = "graph_built"
   
   - **Vector Store**:
     - Generates embeddings for each chunk
     - Stores in ChromaDB with metadata
     - Updates status = "embeddings_generated"

7. **Workflow completes**
   - Both parallel tasks finish
   - Update status = "COMPLETE"
   - Notify user
   - Document now searchable

**Total Duration**: 2-5 minutes (depending on paper length and hardware)

---

## Data Transformations

### Transformation 1: PDF → Parsed JSON

**Input:**
```
Binary PDF file (attention_is_all_you_need.pdf)
```

**Process:**
- PyMuPDF extracts text page by page
- Regex patterns identify title, authors, sections
- Structure is preserved using position information

**Output:**
```json
{
  "document_id": "doc_123",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "..."],
    "year": 2017,
    "venue": "NeurIPS",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
  },
  "sections": [
    {
      "heading": "Introduction",
      "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular...",
      "position": 0,
      "page": 1,
      "word_count": 342
    },
    {
      "heading": "Background",
      "text": "The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU...",
      "position": 1,
      "page": 2,
      "word_count": 287
    }
  ],
  "total_pages": 15,
  "total_words": 7234
}
```

**Value Added:**
- Unstructured PDF → Structured JSON
- Metadata extracted and normalized
- Document structure preserved
- Ready for downstream processing

---

### Transformation 2: Parsed JSON → Semantic Chunks

**Input:**
```json
{
  "sections": [
    {
      "heading": "Introduction",
      "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures. Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."
    }
  ]
}
```

**Process:**
1. Split into sentences
2. Generate embeddings for each sentence
3. Compute cosine similarity between consecutive sentences
4. Identify boundaries where similarity drops
5. Create chunks respecting boundaries and size constraints

**Output:**
```json
[
  {
    "chunk_id": "chunk_001",
    "document_id": "doc_123",
    "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.",
    "position": 0,
    "section": "Introduction",
    "token_count": 52,
    "embedding_id": "emb_001"
  },
  {
    "chunk_id": "chunk_002",
    "document_id": "doc_123",
    "text": "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network.",
    "position": 1,
    "section": "Introduction",
    "token_count": 54,
    "embedding_id": "emb_002"
  },
  {
    "chunk_id": "chunk_003",
    "document_id": "doc_123",
    "text": "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",
    "position": 2,
    "section": "Introduction",
    "token_count": 32,
    "embedding_id": "emb_003"
  }
]
```

**Value Added:**
- Long text → Semantically coherent chunks
- Each chunk is a complete thought
- Optimal size for retrieval (not too small, not too large)
- Maintains context and meaning

---

### Transformation 3: Chunks → Concepts and Entities

**Input:**
```json
{
  "chunk_id": "chunk_001",
  "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation."
}
```

**Process:**
1. SpaCy NER identifies entities
2. KeyBERT extracts keyphrases
3. Normalize concept names
4. Assign confidence scores

**Output:**
```json
{
  "chunk_id": "chunk_001",
  "entities": [
    {
      "text": "recurrent neural networks",
      "type": "METHOD",
      "confidence": 0.96,
      "start": 0,
      "end": 25
    },
    {
      "text": "long short-term memory",
      "type": "METHOD",
      "confidence": 0.94,
      "start": 27,
      "end": 49
    },
    {
      "text": "gated recurrent neural networks",
      "type": "METHOD",
      "confidence": 0.95,
      "start": 54,
      "end": 85
    },
    {
      "text": "sequence modeling",
      "type": "TASK",
      "confidence": 0.89,
      "start": 156,
      "end": 173
    },
    {
      "text": "machine translation",
      "type": "TASK",
      "confidence": 0.92,
      "start": 234,
      "end": 253
    }
  ],
  "keyphrases": [
    {
      "phrase": "recurrent neural networks",
      "score": 0.87
    },
    {
      "phrase": "sequence modeling",
      "score": 0.82
    },
    {
      "phrase": "state of the art approaches",
      "score": 0.78
    }
  ],
  "normalized_concepts": [
    "rnn",
    "lstm",
    "gru",
    "sequence_modeling",
    "machine_translation"
  ]
}
```

**Value Added:**
- Raw text → Structured concepts
- Entities identified and typed
- Important phrases extracted
- Concepts normalized for consistency

---

### Transformation 4: Concepts → Knowledge Graph

**Input:**
```json
{
  "document_id": "doc_123",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": ["Vaswani et al."],
    "year": 2017
  },
  "concepts": [
    {"name": "transformer", "type": "METHOD"},
    {"name": "attention mechanism", "type": "METHOD"},
    {"name": "sequence modeling", "type": "TASK"}
  ]
}
```

**Process:**
1. Create Paper node
2. Create or link to Concept nodes
3. Create MENTIONS relationships
4. Compute co-occurrence for RELATED_TO edges

**Output (Cypher):**
```cypher
// Create Paper node
CREATE (p:Paper {
  id: 'doc_123',
  title: 'Attention Is All You Need',
  authors: ['Vaswani et al.'],
  year: 2017
})

// Create Concept nodes (or MERGE if exists)
MERGE (c1:Concept {name: 'transformer', type: 'METHOD'})
MERGE (c2:Concept {name: 'attention mechanism', type: 'METHOD'})
MERGE (c3:Concept {name: 'sequence modeling', type: 'TASK'})

// Create MENTIONS relationships
CREATE (p)-[:MENTIONS {frequency: 47, confidence: 0.95}]->(c1)
CREATE (p)-[:MENTIONS {frequency: 89, confidence: 0.97}]->(c2)
CREATE (p)-[:MENTIONS {frequency: 23, confidence: 0.89}]->(c3)

// Create RELATED_TO relationships (co-occurrence)
MERGE (c1)-[:RELATED_TO {weight: 0.85, papers: ['doc_123']}]-(c2)
MERGE (c1)-[:RELATED_TO {weight: 0.72, papers: ['doc_123']}]-(c3)
```

**Value Added:**
- Flat concepts → Connected graph
- Relationships between concepts
- Queryable structure
- Enables graph traversal and discovery

---

### Transformation 5: Chunks → Embeddings

**Input:**
```json
{
  "chunk_id": "chunk_001",
  "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation."
}
```

**Process:**
1. Tokenize text
2. Pass through Sentence-BERT model
3. Extract embedding vector (384 or 768 dimensions)
4. Normalize vector (L2 normalization)

**Output:**
```json
{
  "chunk_id": "chunk_001",
  "embedding": [
    0.0234, -0.0456, 0.0789, 0.0123, -0.0345, 0.0567, ..., 0.0234
  ],
  "dimensions": 384,
  "model": "all-MiniLM-L6-v2",
  "normalized": true
}
```

**Stored in ChromaDB:**
```python
collection.add(
    ids=["chunk_001"],
    embeddings=[[0.0234, -0.0456, ..., 0.0234]],
    metadatas=[{
        "document_id": "doc_123",
        "position": 0,
        "section": "Introduction",
        "token_count": 52
    }],
    documents=["Recurrent neural networks, long short-term memory..."]
)
```

**Value Added:**
- Text → Dense vector representation
- Semantic meaning captured in numbers
- Enables similarity search
- Fast approximate nearest neighbor queries

---

## Query Workflows

### Workflow 1: Semantic Search

**User Action:** Search for "attention mechanisms in computer vision"

**Step-by-Step Execution:**

1. **Query Embedding Generation**
   ```python
   query = "attention mechanisms in computer vision"
   query_embedding = model.encode(query)  # [0.0123, -0.0234, ..., 0.0456]
   ```

2. **Vector Search**
   ```python
   results = collection.query(
       query_embeddings=[query_embedding],
       n_results=50
   )
   ```

3. **Result Aggregation**
   ```python
   # Group chunks by document
   documents = {}
   for chunk in results:
       doc_id = chunk['metadata']['document_id']
       if doc_id not in documents:
           documents[doc_id] = {
               'chunks': [],
               'total_similarity': 0
           }
       documents[doc_id]['chunks'].append(chunk)
       documents[doc_id]['total_similarity'] += chunk['similarity']
   ```

4. **Ranking**
   ```python
   # Rank documents by total similarity
   ranked_docs = sorted(
       documents.items(),
       key=lambda x: x[1]['total_similarity'],
       reverse=True
   )[:10]
   ```

5. **Result Formatting**
   ```json
   {
     "query": "attention mechanisms in computer vision",
     "results": [
       {
         "document_id": "doc_456",
         "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
         "authors": ["Dosovitskiy et al."],
         "year": 2020,
         "relevance_score": 0.89,
         "relevant_chunks": [
           {
             "text": "We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks...",
             "similarity": 0.91
           },
           {
             "text": "The Transformer architecture has become the de-facto standard for natural language processing tasks. However, in computer vision, convolutional architectures remain dominant...",
             "similarity": 0.87
           }
         ]
       }
     ],
     "total_results": 10,
     "search_time_ms": 234
   }
   ```

**Duration:** < 2 seconds

---

### Workflow 2: Graph Query

**User Action:** Find papers related to "transformer" through 2 hops

**Cypher Query:**
```cypher
MATCH (p1:Paper)-[:MENTIONS]->(c1:Concept {name: "transformer"})
      -[:RELATED_TO*1..2]-(c2:Concept)
      <-[:MENTIONS]-(p2:Paper)
WHERE p1 <> p2
RETURN p2.title, p2.authors, p2.year, 
       COUNT(DISTINCT c2) as shared_concepts,
       COLLECT(DISTINCT c2.name) as concepts
ORDER BY shared_concepts DESC
LIMIT 10
```

**Result:**
```json
{
  "query": "Papers related to 'transformer' (2 hops)",
  "results": [
    {
      "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      "authors": ["Devlin et al."],
      "year": 2018,
      "shared_concepts": 8,
      "concepts": [
        "transformer",
        "attention mechanism",
        "pre-training",
        "language model",
        "bidirectional",
        "masked language modeling",
        "next sentence prediction",
        "fine-tuning"
      ]
    },
    {
      "title": "Vision Transformer (ViT)",
      "authors": ["Dosovitskiy et al."],
      "year": 2020,
      "shared_concepts": 6,
      "concepts": [
        "transformer",
        "attention mechanism",
        "image patches",
        "computer vision",
        "self-attention",
        "positional encoding"
      ]
    }
  ]
}
```

**Duration:** < 500ms

---

### Workflow 3: AI Research Assistant (RAG)

**User Action:** Ask "What are the main advantages of transformers over RNNs?"

**Step-by-Step Execution:**

1. **Retrieve Context**
   ```python
   # Convert question to embedding
   question_embedding = model.encode(
       "What are the main advantages of transformers over RNNs?"
   )
   
   # Retrieve top 5 relevant chunks
   context_chunks = collection.query(
       query_embeddings=[question_embedding],
       n_results=5
   )
   ```

2. **Construct Prompt**
   ```python
   prompt = f"""
   You are a research assistant helping a researcher understand academic papers.
   Answer the question based ONLY on the provided context from research papers.
   Cite papers using [Paper Title, Authors, Year] format.
   
   Context:
   
   [1] "Transformers eliminate recurrence, enabling parallel processing of sequences. 
        This allows for significantly more efficient training on modern hardware."
        (Attention Is All You Need, Vaswani et al., 2017)
   
   [2] "Unlike RNNs which process tokens sequentially, transformers process all tokens 
        simultaneously through self-attention, capturing long-range dependencies more effectively."
        (Attention Is All You Need, Vaswani et al., 2017)
   
   [3] "The self-attention mechanism in transformers allows the model to weigh the importance 
        of different parts of the input when processing each token, without the vanishing 
        gradient problems that plague RNNs."
        (BERT: Pre-training of Deep Bidirectional Transformers, Devlin et al., 2018)
   
   [4] "Transformers scale better to longer sequences than RNNs because they don't suffer 
        from the sequential bottleneck."
        (Attention Is All You Need, Vaswani et al., 2017)
   
   [5] "The parallel nature of transformers enables training on much larger datasets 
        compared to RNNs, leading to better performance."
        (Language Models are Few-Shot Learners, Brown et al., 2020)
   
   Question: {question}
   
   Answer:
   """
   ```

3. **Generate Answer**
   ```python
   response = openai.ChatCompletion.create(
       model="gpt-4",
       messages=[
           {"role": "system", "content": "You are a helpful research assistant."},
           {"role": "user", "content": prompt}
       ],
       temperature=0.3
   )
   
   answer = response.choices[0].message.content
   ```

4. **Format Response**
   ```json
   {
     "question": "What are the main advantages of transformers over RNNs?",
     "answer": "Transformers offer several key advantages over RNNs:\n\n1. **Parallel Processing**: Transformers eliminate recurrence and process all tokens simultaneously, enabling much more efficient training on modern hardware [Attention Is All You Need, Vaswani et al., 2017]. This is in contrast to RNNs which must process tokens sequentially.\n\n2. **Long-Range Dependencies**: The self-attention mechanism allows transformers to capture long-range dependencies more effectively than RNNs, without suffering from vanishing gradient problems [BERT, Devlin et al., 2018].\n\n3. **Scalability**: Transformers scale better to longer sequences because they don't have the sequential bottleneck that limits RNNs [Attention Is All You Need, Vaswani et al., 2017].\n\n4. **Training Efficiency**: The parallel nature enables training on much larger datasets, leading to better performance [Language Models are Few-Shot Learners, Brown et al., 2020].\n\nThese advantages have made transformers the dominant architecture in NLP and increasingly in computer vision as well.",
     "sources": [
       {
         "title": "Attention Is All You Need",
         "authors": ["Vaswani et al."],
         "year": 2017,
         "relevance": 0.94
       },
       {
         "title": "BERT: Pre-training of Deep Bidirectional Transformers",
         "authors": ["Devlin et al."],
         "year": 2018,
         "relevance": 0.87
       },
       {
         "title": "Language Models are Few-Shot Learners",
         "authors": ["Brown et al."],
         "year": 2020,
         "relevance": 0.82
       }
     ],
     "confidence": "high"
   }
   ```

**Duration:** 3-5 seconds (depending on LLM)

---

## Error and Failure Flow

### Error Scenario 1: PDF Parsing Failure

**Trigger:** Corrupted or malformed PDF

**Flow:**
```
1. Parser attempts to extract text
   ↓
2. PyMuPDF raises exception
   ↓
3. Orchestrator catches exception
   ↓
4. Log error with details
   ↓
5. Retry with alternative extraction method (attempt 1/3)
   ↓
6. If still fails, retry again (attempt 2/3)
   ↓
7. If still fails, retry once more (attempt 3/3)
   ↓
8. If all retries exhausted:
   - Mark document status = "FAILED"
   - Store error message in database
   - Notify user with specific error
   ↓
9. User can:
   - Re-upload a different version
   - Mark as "cannot process"
   - Request manual review
```

**Error Message to User:**
```
"Unable to parse 'paper_name.pdf'. The file may be corrupted or use an unsupported PDF format. 
Please try:
1. Re-downloading the PDF from the source
2. Converting to a standard PDF format
3. Checking if the file opens correctly in a PDF reader

Error details: [PyMuPDF error: Invalid PDF structure at byte 12345]"
```

---

### Error Scenario 2: Concept Extraction Failure

**Trigger:** Model fails to load or process text

**Flow:**
```
1. Concept Extractor attempts to load SpaCy model
   ↓
2. Model fails to load (missing, corrupted, or OOM)
   ↓
3. Orchestrator catches exception
   ↓
4. Log error with details
   ↓
5. Attempt graceful degradation:
   - Skip NER, use only KeyBERT
   - Or skip concept extraction entirely
   ↓
6. Continue processing with partial data
   ↓
7. Mark document with warning flag
   ↓
8. Notify user:
   "Document processed successfully, but concept extraction was limited. 
    Search and graph features may be less accurate for this document."
```

**Why graceful degradation:**
- Better to have partial data than complete failure
- User can still search by text similarity
- Can retry concept extraction later when model is fixed

---

### Error Scenario 3: Database Connection Failure

**Trigger:** Neo4j or ChromaDB unavailable

**Flow:**
```
1. Graph Builder attempts to connect to Neo4j
   ↓
2. Connection fails (database down, network issue)
   ↓
3. Circuit breaker detects failure
   ↓
4. Circuit breaker opens (stops trying)
   ↓
5. Queue operation for retry
   ↓
6. Continue processing other steps
   ↓
7. Background process retries connection every 30 seconds
   ↓
8. When connection restored:
   - Circuit breaker closes
   - Process queued operations
   - Resume normal operation
```

**Why circuit breaker:**
- Prevents wasting resources on failing operations
- Gives database time to recover
- Fails fast instead of hanging

---

### Error Scenario 4: LLM API Failure

**Trigger:** OpenAI API rate limit or timeout

**Flow:**
```
1. User asks question
   ↓
2. RAG system retrieves context
   ↓
3. Attempts to call OpenAI API
   ↓
4. API returns 429 (rate limit) or times out
   ↓
5. Retry with exponential backoff:
   - Wait 1 second, retry
   - Wait 2 seconds, retry
   - Wait 4 seconds, retry
   ↓
6. If all retries fail:
   - Return retrieved context without LLM generation
   - Notify user: "AI assistant temporarily unavailable. 
                   Here are relevant excerpts from papers..."
```

**Fallback strategy:**
- Show retrieved chunks even without LLM
- User still gets value (relevant excerpts)
- Can try again later

---

## State Management

### Document State Schema

```python
class DocumentState:
    # Identity
    document_id: str
    filename: str
    upload_timestamp: datetime
    
    # Processing status
    status: str  # "uploaded", "parsing", "chunking", "extracting", 
                 # "building_graph", "embedding", "complete", "failed"
    current_step: str
    progress_percentage: int  # 0-100
    
    # Error handling
    retry_count: int
    max_retries: int = 3
    error_message: Optional[str]
    error_timestamp: Optional[datetime]
    
    # Processing data
    parsed_data: Optional[dict]
    chunks: Optional[List[dict]]
    concepts: Optional[List[dict]]
    
    # Metadata
    processing_start_time: datetime
    processing_end_time: Optional[datetime]
    processing_duration_seconds: Optional[float]
```

### State Transitions

```
UPLOADED → PARSING → CHUNKING → EXTRACTING → [BUILDING_GRAPH, EMBEDDING] → COMPLETE
                                                                          ↓
                                                                       FAILED
```

**Valid Transitions:**
- UPLOADED → PARSING
- PARSING → CHUNKING (on success)
- PARSING → FAILED (on failure after retries)
- CHUNKING → EXTRACTING
- EXTRACTING → BUILDING_GRAPH + EMBEDDING (parallel)
- BUILDING_GRAPH + EMBEDDING → COMPLETE (when both finish)
- Any state → FAILED (on unrecoverable error)

**Invalid Transitions:**
- COMPLETE → any other state (terminal state)
- FAILED → any state except UPLOADED (must re-upload)

---

## Interview Preparation

### Key Talking Points

1. **End-to-End Data Flow**
   - "The system transforms unstructured PDFs into a queryable knowledge graph and vector store through a multi-stage pipeline"
   - "Each stage adds value: parsing extracts structure, chunking preserves meaning, extraction identifies concepts, storage enables search"

2. **State Management**
   - "We use LangGraph to track document progress through the pipeline, with explicit state transitions and error handling"
   - "Each document has independent state, allowing concurrent processing without interference"

3. **Error Handling**
   - "We implement retry with exponential backoff for transient failures"
   - "Circuit breakers prevent cascading failures when external services are down"
   - "Graceful degradation ensures partial functionality even when some components fail"

4. **Data Transformations**
   - "Each transformation is reversible and traceable - we can always go back to the source"
   - "Transformations are idempotent - running them multiple times produces the same result"

5. **Scalability**
   - "The pipeline is designed for horizontal scaling - we can add more workers to process documents in parallel"
   - "Stateless components make it easy to distribute processing across multiple machines"

### Sample Interview Questions

**Q: Walk me through what happens when a user uploads a PDF.**

**A:** "When a user uploads a PDF, the system goes through several stages:

1. **Validation**: We check the file format and size to ensure it's a valid PDF under 50MB
2. **Storage**: We store the PDF in the file system and create a metadata record in SQLite
3. **Orchestration**: LangGraph starts a workflow to process the document
4. **Parsing**: PyMuPDF extracts text, metadata, and structure
5. **Chunking**: We use Sentence-BERT to identify semantic boundaries and create coherent chunks
6. **Extraction**: SpaCy and KeyBERT identify concepts and entities
7. **Parallel Storage**: We simultaneously build the knowledge graph in Neo4j and generate embeddings for ChromaDB
8. **Completion**: The document is marked complete and becomes searchable

The entire process takes 2-5 minutes and is fully asynchronous, so the user can continue working while it processes."

**Q: How do you handle failures in the pipeline?**

**A:** "We have multiple layers of error handling:

1. **Retry Logic**: Transient failures (network issues, temporary resource constraints) are retried up to 3 times with exponential backoff
2. **Circuit Breakers**: For external services like databases or APIs, we use circuit breakers to fail fast and prevent resource exhaustion
3. **Graceful Degradation**: If a non-critical component fails (like concept extraction), we continue processing with partial data rather than failing completely
4. **State Tracking**: Every document has explicit state, so we know exactly where it failed and can resume or retry from that point
5. **User Notification**: We provide clear, actionable error messages that help users understand what went wrong and what they can do

For example, if concept extraction fails but parsing and chunking succeed, we still store the chunks and enable text-based search, just with a warning that concept-based features may be limited."

**Q: How would you optimize this pipeline for better performance?**

**A:** "Several optimization strategies:

1. **Batch Processing**: Process multiple documents in parallel (currently limited to 10 concurrent)
2. **Caching**: Cache embeddings for common queries and frequently accessed chunks
3. **Async I/O**: Use async/await for database operations to avoid blocking
4. **GPU Acceleration**: Use GPU for embedding generation (10-50x faster than CPU)
5. **Incremental Updates**: Only reprocess changed sections instead of entire documents
6. **Database Optimization**: Add indexes on frequently queried fields, use connection pooling
7. **Lazy Loading**: Don't load full document text into memory, stream it
8. **Compression**: Compress embeddings for storage (quantization)

The specific optimizations depend on the bottleneck - we'd profile first to identify where time is spent."

**Q: How does this design scale to millions of documents?**

**A:** "The current design handles thousands of documents well, but for millions we'd need:

1. **Distributed Processing**: Replace LangGraph with a distributed task queue (Celery + Redis) to process across multiple machines
2. **Database Scaling**: 
   - SQLite → PostgreSQL (supports concurrent writes)
   - ChromaDB → Milvus or Weaviate (distributed vector search with sharding)
   - Neo4j → Neo4j cluster (distributed graph database)
3. **Storage**: Move from local files to S3 or similar object storage
4. **Caching Layer**: Add Redis for frequently accessed data
5. **Load Balancing**: Multiple API servers behind a load balancer
6. **Monitoring**: Prometheus + Grafana for observability
7. **Async Everything**: All operations become async with message queues

The key is that the architecture is already modular, so we can swap components without rewriting the entire system."

---

**This workflow documentation provides a complete understanding of how data flows through the system, how errors are handled, and how the system behaves in production - essential knowledge for system design interviews.**
