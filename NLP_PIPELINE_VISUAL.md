# NLP Pipeline - Visual Explanation

## Complete NLP Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: PDF DOCUMENT                      │
│              "Attention Is All You Need.pdf"                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 1: TEXT EXTRACTION                    │
│                                                              │
│  Library: PyMuPDF                                           │
│  NLP: Document parsing, structure preservation              │
│                                                              │
│  Output: "The dominant sequence transduction models..."     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2A: SEMANTIC CHUNKING                     │
│                                                              │
│  Model: Sentence Transformers (all-MiniLM-L6-v2)           │
│  NLP Techniques:                                            │
│    1. Sentence Segmentation                                 │
│    2. Embedding Generation (384-dim vectors)                │
│    3. Cosine Similarity Calculation                         │
│    4. Boundary Detection                                    │
│                                                              │
│  Example:                                                   │
│  Sentence 1: "Transformers are powerful"                    │
│     ↓ Encode                                                │
│  Vector 1: [0.23, -0.45, 0.67, ..., 0.12]  (384 dims)     │
│                                                              │
│  Sentence 2: "They use attention mechanisms"                │
│     ↓ Encode                                                │
│  Vector 2: [0.21, -0.43, 0.69, ..., 0.15]  (384 dims)     │
│                                                              │
│     ↓ Calculate Similarity                                  │
│  Similarity: 0.87 (HIGH - same topic, keep together)        │
│                                                              │
│  Sentence 3: "The weather is nice today"                    │
│     ↓ Encode                                                │
│  Vector 3: [-0.12, 0.34, -0.23, ..., 0.45] (384 dims)     │
│                                                              │
│     ↓ Calculate Similarity                                  │
│  Similarity: 0.23 (LOW - different topic, split here!)      │
│                                                              │
│  Output: Semantically coherent chunks                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         PHASE 2B: NAMED ENTITY RECOGNITION (NER)             │
│                                                              │
│  Model: spaCy (en_core_web_sm)                              │
│  NLP Techniques:                                            │
│    1. Tokenization                                          │
│    2. Part-of-Speech Tagging                                │
│    3. Dependency Parsing                                    │
│    4. Entity Classification                                 │
│                                                              │
│  Example Input:                                             │
│  "Google developed BERT at Stanford in 2018"                │
│                                                              │
│  NLP Processing:                                            │
│  ┌────────┬──────┬─────────┬──────────┐                    │
│  │ Token  │ POS  │ Dep     │ Entity   │                    │
│  ├────────┼──────┼─────────┼──────────┤                    │
│  │ Google │ NOUN │ nsubj   │ ORG      │ ← Organization     │
│  │ developed│VERB│ ROOT    │ -        │                    │
│  │ BERT   │ NOUN │ dobj    │ PRODUCT  │ ← Product          │
│  │ at     │ ADP  │ prep    │ -        │                    │
│  │ Stanford│NOUN │ pobj    │ ORG      │ ← Organization     │
│  │ in     │ ADP  │ prep    │ -        │                    │
│  │ 2018   │ NUM  │ pobj    │ DATE     │ ← Date             │
│  └────────┴──────┴─────────┴──────────┘                    │
│                                                              │
│  Output Entities:                                           │
│    - "Google" (ORG, confidence: 0.95)                       │
│    - "BERT" (PRODUCT, confidence: 0.89)                     │
│    - "Stanford" (ORG, confidence: 0.92)                     │
│    - "2018" (DATE, confidence: 0.99)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          PHASE 2C: KEYPHRASE EXTRACTION (KeyBERT)            │
│                                                              │
│  Model: KeyBERT + Sentence Transformers                     │
│  NLP Techniques:                                            │
│    1. N-gram Generation (1-3 words)                         │
│    2. BERT Embeddings                                       │
│    3. Semantic Similarity Ranking                           │
│    4. Maximal Marginal Relevance (diversity)                │
│                                                              │
│  Example Input:                                             │
│  "Neural networks use backpropagation for training.         │
│   Deep learning models achieve state-of-the-art results."   │
│                                                              │
│  Step 1: Generate Candidates                                │
│    - "neural networks"                                      │
│    - "backpropagation"                                      │
│    - "deep learning"                                        │
│    - "state-of-the-art"                                     │
│    - "neural networks use"                                  │
│    - "deep learning models"                                 │
│    - ... (many more)                                        │
│                                                              │
│  Step 2: Embed Document & Candidates                        │
│    Doc Vector: [0.45, -0.23, 0.67, ..., 0.12]             │
│    "neural networks": [0.43, -0.21, 0.65, ..., 0.14]      │
│    "backpropagation": [0.41, -0.19, 0.63, ..., 0.16]      │
│    "deep learning": [0.44, -0.22, 0.66, ..., 0.13]        │
│                                                              │
│  Step 3: Calculate Similarity                               │
│    "neural networks": 0.92 ← HIGH relevance                │
│    "deep learning": 0.89 ← HIGH relevance                  │
│    "backpropagation": 0.85 ← HIGH relevance                │
│    "state-of-the-art": 0.78 ← MEDIUM relevance             │
│                                                              │
│  Step 4: Apply MMR (diversity)                              │
│    Remove similar phrases, keep diverse ones                │
│                                                              │
│  Output Keyphrases:                                         │
│    1. "neural networks" (score: 0.92)                       │
│    2. "deep learning" (score: 0.89)                         │
│    3. "backpropagation" (score: 0.85)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            PHASE 3: KNOWLEDGE GRAPH CONSTRUCTION             │
│                                                              │
│  NLP Techniques:                                            │
│    1. Concept Normalization (lowercasing, deduplication)    │
│    2. Co-occurrence Analysis                                │
│    3. Relationship Inference                                │
│                                                              │
│  Example:                                                   │
│                                                              │
│  Paper 1 mentions: ["neural network", "deep learning"]      │
│  Paper 2 mentions: ["neural network", "backpropagation"]    │
│  Paper 3 mentions: ["deep learning", "CNN"]                 │
│                                                              │
│  Co-occurrence Matrix:                                      │
│                neural_net  deep_learn  backprop  CNN        │
│  neural_net        -           2          1       0         │
│  deep_learn        2           -          0       1         │
│  backprop          1           0          -       0         │
│  CNN               0           1          0       -         │
│                                                              │
│  Relationships Created:                                     │
│    neural_net ←→ deep_learn (strength: 0.8)                │
│    neural_net ←→ backprop (strength: 0.5)                  │
│    deep_learn ←→ CNN (strength: 0.5)                       │
│                                                              │
│  Graph Structure:                                           │
│                                                              │
│         [Paper 1]                                           │
│            ↓                                                │
│      MENTIONS (freq: 5)                                     │
│            ↓                                                │
│    [neural network] ←─RELATED_TO─→ [deep learning]         │
│            ↓                              ↓                 │
│      RELATED_TO                      RELATED_TO             │
│            ↓                              ↓                 │
│    [backpropagation]                    [CNN]               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                              │
│                                                              │
│  Structured Knowledge:                                      │
│    ✓ Papers with metadata                                  │
│    ✓ Semantic chunks (context-aware)                       │
│    ✓ Extracted entities (NER)                              │
│    ✓ Key concepts (KeyBERT)                                │
│    ✓ Concept relationships (co-occurrence)                 │
│    ✓ Knowledge graph (optional)                            │
│                                                              │
│  Ready for:                                                 │
│    → Semantic search                                        │
│    → Question answering                                     │
│    → Research recommendations                               │
│    → Trend analysis                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## NLP Models in Detail

### 1. Sentence Transformers Architecture

```
Input: "Neural networks are powerful"
   ↓
Tokenization: ["neural", "networks", "are", "powerful"]
   ↓
Token Embeddings: [768-dim vectors for each token]
   ↓
Transformer Layers (12 layers):
   - Self-Attention: Tokens attend to each other
   - Feed-Forward: Non-linear transformations
   ↓
Pooling: Mean pooling across tokens
   ↓
Output: Single 384-dim sentence vector
   ↓
Use: Semantic similarity, clustering, search
```

### 2. spaCy NER Pipeline

```
Input: "Google developed BERT"
   ↓
Tokenizer: ["Google", "developed", "BERT"]
   ↓
Tagger: [NOUN, VERB, NOUN]
   ↓
Parser: Dependency tree
   ↓
NER: Statistical model predicts entity labels
   - "Google" → ORG (probability: 0.95)
   - "BERT" → PRODUCT (probability: 0.89)
   ↓
Output: Labeled entities with confidence
```

### 3. KeyBERT Process

```
Input Document
   ↓
Extract N-grams (1-3 words)
   ↓
Embed Document: BERT → 384-dim vector
   ↓
Embed Each Candidate: BERT → 384-dim vectors
   ↓
Calculate Cosine Similarity:
   similarity = (doc_vec · candidate_vec) / (||doc_vec|| × ||candidate_vec||)
   ↓
Rank by Similarity
   ↓
Apply MMR for Diversity:
   - Keep high similarity to document
   - Reduce similarity between selected phrases
   ↓
Output: Top N diverse, relevant keyphrases
```

---

## Mathematical Concepts

### Cosine Similarity
```
Given two vectors A and B:

similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of A
- Result: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)

Example:
A = [0.5, 0.3, 0.8]  (sentence 1 embedding)
B = [0.4, 0.4, 0.7]  (sentence 2 embedding)

A · B = (0.5×0.4) + (0.3×0.4) + (0.8×0.7) = 0.84
||A|| = √(0.5² + 0.3² + 0.8²) = 1.02
||B|| = √(0.4² + 0.4² + 0.7²) = 0.92

similarity = 0.84 / (1.02 × 0.92) = 0.895 (HIGH similarity!)
```

### Embedding Space
```
High-dimensional space where similar meanings are close:

    "dog" ●
           ↘
            ● "puppy"
           ↗
    "cat" ●

    (far away)
    
    "car" ●
```

---

## Real Example from Your Data

From `test_without_neo4j.py` output:

```
Input Text: "NBER Working Paper on economic analysis..."

↓ NER Processing ↓

Entities Extracted:
  - "NBER" (ORG) ← Organization recognized
  - "AZ 85287" (PRODUCT) ← Product code
  - "1" (CARDINAL) ← Number

↓ KeyBERT Processing ↓

Keyphrases Extracted:
  - "box 879801 tempe" (score: 0.78) ← High relevance
  - "box 879801" (score: 0.72)
  - "tempe az 85287" (score: 0.67)
  - "az 85287 nber" (score: 0.55)

↓ Graph Construction ↓

Relationships:
  Paper ─MENTIONS→ "NBER"
  Paper ─MENTIONS→ "box 879801 tempe"
  "NBER" ←RELATED_TO→ "box 879801 tempe" (co-occur)
```

---

## Why This is Advanced NLP

**Traditional Approach:**
- Keyword matching: "Find papers with 'neural network'"
- Misses: "deep learning", "artificial neural nets", "ANNs"

**Your Approach (Semantic):**
- Embedding-based: Understands "neural network" ≈ "deep learning"
- Finds related concepts even with different words
- Context-aware: "bank" (financial) vs "bank" (river)

---

**This is production-grade NLP using state-of-the-art models!**
