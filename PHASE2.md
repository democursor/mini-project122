# Phase 2: Semantic Chunking and Concept Extraction

## Overview

Phase 2 transforms the structured text from Phase 1 into semantically meaningful chunks and extracts key concepts, entities, and relationships. This phase bridges the gap between raw text and knowledge representation, enabling intelligent search and discovery.

**Learning Objectives:**
- Understand semantic similarity and embeddings
- Learn chunking strategies for optimal retrieval
- Master named entity recognition (NER) for scientific text
- Explore keyphrase extraction and concept normalization
- Apply deep learning models for text understanding

**Key Concepts:**
- Sentence embeddings and semantic similarity
- Semantic boundary detection
- Named Entity Recognition (NER)
- Keyphrase extraction with KeyBERT
- Concept normalization and relationship extraction

---

## Table of Contents

1. [Semantic Chunking Module](#semantic-chunking-module)
2. [Concept Extraction Module](#concept-extraction-module)
3. [Deep Learning Models](#deep-learning-models)
4. [Learning Outcomes](#learning-outcomes)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Success Criteria](#success-criteria)

---

## Semantic Chunking Module

### Purpose

The Semantic Chunking Module intelligently segments documents into coherent chunks that preserve meaning and context. Unlike fixed-size chunking, semantic chunking creates boundaries at natural topic transitions.

**Why Semantic Chunking Matters:**
- **Better Retrieval:** Chunks contain complete thoughts, not fragments
- **Preserved Context:** Related sentences stay together
- **Optimal Size:** Balances context with granularity
- **Improved Search:** More relevant results for user queries

### 9.1 Chunking Algorithm

#### The Challenge with Fixed-Size Chunking

**Problem with Fixed-Size (e.g., every 500 tokens):**
```
"Transformers have revolutionized natural language processing. The attention mechanism allows models to focus on relevant parts of the input sequence. [CHUNK BREAK] This selective attention is computed using queries, keys, and values. The self-attention mechanism enables..."
```

**Issues:**
- Breaks mid-concept (attention mechanism split)
- Loses context between chunks
- Arbitrary boundaries ignore semantic structure

#### Semantic Chunking Solution

**Approach:** Use sentence embeddings to detect topic transitions

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class SemanticChunker:
    """Intelligently chunks text based on semantic similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with sentence embedding model.
        
        Model choice:
        - all-MiniLM-L6-v2: Fast, 384 dimensions, good quality
        - all-mpnet-base-v2: Slower, 768 dimensions, best quality
        - sentence-t5-base: Good for scientific text
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.7  # Configurable
    
    def chunk(self, text: str, min_tokens: int = 100, max_tokens: int = 500) -> List[Chunk]:
        """
        Create semantic chunks from text.
        
        Algorithm:
        1. Split text into sentences
        2. Compute embeddings for each sentence
        3. Calculate similarity between consecutive sentences
        4. Create boundaries where similarity drops below threshold
        5. Ensure chunks meet size constraints
        """
        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return [Chunk(text=text, start_idx=0, end_idx=len(text))]
        
        # Step 2: Compute embeddings
        embeddings = self.model.encode(sentences)
        
        # Step 3: Find semantic boundaries
        boundaries = self._find_boundaries(sentences, embeddings)
        
        # Step 4: Create chunks respecting size constraints
        chunks = self._create_chunks(sentences, boundaries, min_tokens, max_tokens)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or spaCy"""
        import nltk
        nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_boundaries(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
        """
        Find semantic boundaries between sentences.
        
        Method:
        1. Calculate cosine similarity between consecutive sentences
        2. Find positions where similarity drops significantly
        3. Mark as potential chunk boundaries
        """
        boundaries = [0]  # Always start with first sentence
        
        for i in range(len(embeddings) - 1):
            # Calculate similarity between consecutive sentences
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            
            # If similarity drops below threshold, create boundary
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        
        boundaries.append(len(sentences))  # Always end with last sentence
        return boundaries
    
    def _create_chunks(self, sentences: List[str], boundaries: List[int], 
                      min_tokens: int, max_tokens: int) -> List[Chunk]:
        """
        Create chunks from boundaries, respecting size constraints.
        
        Strategy:
        1. Combine sentences between boundaries
        2. If chunk too small, merge with next
        3. If chunk too large, split at sentence boundaries
        """
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Count tokens (approximate)
            token_count = len(chunk_text.split())
            
            # Handle size constraints
            if token_count < min_tokens and i < len(boundaries) - 2:
                # Merge with next chunk
                continue
            elif token_count > max_tokens:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk_sentences, max_tokens)
                chunks.extend(sub_chunks)
            else:
                # Perfect size
                chunk = Chunk(
                    text=chunk_text,
                    sentences=chunk_sentences,
                    token_count=token_count,
                    start_sentence=start_idx,
                    end_sentence=end_idx
                )
                chunks.append(chunk)
        
        return chunks
```

---

#### Advanced Boundary Detection

**Problem:** Simple threshold doesn't work for all text types

**Solution:** Adaptive threshold based on local context

```python
def _find_adaptive_boundaries(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
    """
    Find boundaries using adaptive threshold.
    
    Method:
    1. Calculate all pairwise similarities
    2. Use local minimum detection
    3. Adjust threshold based on text characteristics
    """
    similarities = []
    
    # Calculate similarities between consecutive sentences
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    # Find local minima (topic transitions)
    boundaries = [0]
    
    for i in range(1, len(similarities) - 1):
        # Check if this is a local minimum
        if (similarities[i] < similarities[i-1] and 
            similarities[i] < similarities[i+1] and
            similarities[i] < np.mean(similarities) - np.std(similarities)):
            boundaries.append(i + 1)
    
    boundaries.append(len(sentences))
    return boundaries
```

---

### 9.2 Chunk Size Constraints

#### Token Counting

```python
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken (OpenAI's tokenizer).
    
    Why accurate token counting matters:
    - LLM context windows are measured in tokens
    - Embedding models have token limits
    - Consistent chunk sizes improve retrieval
    """
    try:
        import tiktoken
        
        # Get tokenizer for specific model
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
        
    except ImportError:
        # Fallback: approximate with word count
        return len(text.split()) * 1.3  # Rough approximation
```

---

#### Size Balancing Strategy

```python
class ChunkSizeBalancer:
    """Balances chunk sizes to meet constraints"""
    
    def __init__(self, min_tokens: int = 100, max_tokens: int = 500):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    def balance_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Balance chunk sizes by merging small chunks and splitting large ones.
        
        Strategy:
        1. Merge consecutive small chunks
        2. Split large chunks at sentence boundaries
        3. Ensure all chunks meet size requirements
        """
        balanced_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if current_chunk.token_count < self.min_tokens:
                # Try to merge with next chunk
                merged_chunk = self._merge_with_next(chunks, i)
                if merged_chunk:
                    balanced_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk (already merged)
                else:
                    balanced_chunks.append(current_chunk)
                    i += 1
                    
            elif current_chunk.token_count > self.max_tokens:
                # Split large chunk
                split_chunks = self._split_chunk(current_chunk)
                balanced_chunks.extend(split_chunks)
                i += 1
                
            else:
                # Perfect size
                balanced_chunks.append(current_chunk)
                i += 1
        
        return balanced_chunks
    
    def _merge_with_next(self, chunks: List[Chunk], index: int) -> Optional[Chunk]:
        """Merge current chunk with next chunk if possible"""
        if index + 1 >= len(chunks):
            return None
        
        current = chunks[index]
        next_chunk = chunks[index + 1]
        
        merged_text = current.text + " " + next_chunk.text
        merged_tokens = count_tokens(merged_text)
        
        if merged_tokens <= self.max_tokens:
            return Chunk(
                text=merged_text,
                token_count=merged_tokens,
                sentences=current.sentences + next_chunk.sentences
            )
        
        return None
    
    def _split_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split large chunk at sentence boundaries"""
        if len(chunk.sentences) <= 1:
            # Can't split single sentence
            return [chunk]
        
        # Split roughly in half
        mid_point = len(chunk.sentences) // 2
        
        first_half = chunk.sentences[:mid_point]
        second_half = chunk.sentences[mid_point:]
        
        chunks = []
        
        # Create first chunk
        first_text = ' '.join(first_half)
        chunks.append(Chunk(
            text=first_text,
            sentences=first_half,
            token_count=count_tokens(first_text)
        ))
        
        # Create second chunk
        second_text = ' '.join(second_half)
        chunks.append(Chunk(
            text=second_text,
            sentences=second_half,
            token_count=count_tokens(second_text)
        ))
        
        return chunks
```

---

### 9.3 Section Boundary Handling

#### Respecting Document Structure

**Principle:** Never split across major section boundaries

```python
def chunk_with_sections(self, document: ParsedDocument) -> List[Chunk]:
    """
    Chunk document while respecting section boundaries.
    
    Strategy:
    1. Process each section independently
    2. Create chunks within sections
    3. Never merge chunks across sections
    """
    all_chunks = []
    
    for section in document.sections:
        # Chunk this section
        section_chunks = self.chunk(section.content)
        
        # Add section metadata to chunks
        for chunk in section_chunks:
            chunk.section_heading = section.heading
            chunk.section_id = section.id
        
        all_chunks.extend(section_chunks)
    
    return all_chunks
```

---

#### Handling Special Sections

```python
def handle_special_sections(self, section: Section) -> List[Chunk]:
    """
    Handle special sections with custom chunking rules.
    
    Special cases:
    - Abstract: Usually one chunk
    - References: Skip or minimal chunking
    - Figures/Tables: Extract captions only
    """
    section_type = self._classify_section(section.heading)
    
    if section_type == "abstract":
        # Abstract is usually one coherent chunk
        return [Chunk(
            text=section.content,
            section_heading=section.heading,
            chunk_type="abstract"
        )]
    
    elif section_type == "references":
        # Skip references or create one chunk
        return [Chunk(
            text=section.content,
            section_heading=section.heading,
            chunk_type="references"
        )]
    
    elif section_type == "figures":
        # Extract figure captions only
        captions = self._extract_captions(section.content)
        return [Chunk(
            text=caption,
            section_heading=section.heading,
            chunk_type="figure_caption"
        ) for caption in captions]
    
    else:
        # Regular semantic chunking
        return self.chunk(section.content)
```

---

### 9.4 Chunk Metadata and References

#### Chunk Data Model

```python
from dataclasses import dataclass
from typing import List, Optional
import uuid

@dataclass
class Chunk:
    """Represents a semantic chunk of text"""
    
    # Core content
    text: str
    sentences: List[str]
    token_count: int
    
    # Positioning
    start_sentence: int
    end_sentence: int
    start_char: int
    end_char: int
    
    # Document context
    document_id: str
    section_heading: Optional[str] = None
    section_id: Optional[str] = None
    page_numbers: List[int] = None
    
    # Metadata
    chunk_id: str = None
    chunk_type: str = "content"  # content, abstract, references, etc.
    language: str = "en"
    
    # Processing metadata
    created_at: str = None
    embedding_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = f"chunk_{uuid.uuid4()}"
        
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'text': self.text,
            'token_count': self.token_count,
            'section_heading': self.section_heading,
            'start_sentence': self.start_sentence,
            'end_sentence': self.end_sentence,
            'page_numbers': self.page_numbers,
            'chunk_type': self.chunk_type,
            'created_at': self.created_at
        }
```

---

#### Reference Tracking

```python
class ChunkReferenceManager:
    """Manages references between chunks and source documents"""
    
    def create_chunk_references(self, chunks: List[Chunk], 
                              document: ParsedDocument) -> List[ChunkReference]:
        """
        Create reference objects linking chunks to source document.
        
        References enable:
        - Tracing chunks back to original document
        - Finding all chunks from a document
        - Updating chunks when document changes
        """
        references = []
        
        for chunk in chunks:
            ref = ChunkReference(
                chunk_id=chunk.chunk_id,
                document_id=document.document_id,
                document_title=document.metadata.title,
                section_heading=chunk.section_heading,
                start_position=chunk.start_char,
                end_position=chunk.end_char,
                page_numbers=chunk.page_numbers
            )
            references.append(ref)
        
        return references
    
    def find_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Find all chunks belonging to a document"""
        # Query database for chunks with matching document_id
        pass
    
    def find_source_document(self, chunk_id: str) -> ParsedDocument:
        """Find the source document for a chunk"""
        # Query database for document referenced by chunk
        pass
```

---

## Concept Extraction Module

### Purpose

The Concept Extraction Module identifies key concepts, entities, and relationships within document chunks. This module transforms unstructured text into structured knowledge that can be stored in knowledge graphs and used for intelligent search.

**Why Concept Extraction Matters:**
- **Knowledge Discovery:** Automatically identify important concepts
- **Relationship Mapping:** Find connections between ideas
- **Search Enhancement:** Enable concept-based search
- **Knowledge Graphs:** Populate graph databases with entities and relationships

### 10.1 Named Entity Recognition (NER) Approach

#### Scientific NER with SpaCy

**Challenge:** General NER models miss scientific entities

**Solution:** Use domain-specific models trained on scientific literature

```python
import spacy
from spacy import displacy
from typing import List, Dict, Set

class ScientificNER:
    """Named Entity Recognition for scientific literature"""
    
    def __init__(self, model_name: str = "en_core_sci_md"):
        """
        Initialize with scientific NER model.
        
        Models:
        - en_core_sci_md: General scientific model
        - en_ner_bc5cdr_md: Biomedical (chemicals, diseases)
        - en_ner_bionlp13cg_md: Biology (genes, proteins)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Install with:")
            print(f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_name}-0.5.1.tar.gz")
            # Fallback to general model
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Entity types in scientific models:
        - PERSON: Researchers, authors
        - ORG: Institutions, companies
        - GPE: Geographic locations
        - CHEMICAL: Chemical compounds
        - DISEASE: Medical conditions
        - GENE: Gene names
        - PROTEIN: Protein names
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=self._calculate_confidence(ent)
            )
            entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, ent) -> float:
        """
        Calculate confidence score for entity.
        
        Factors:
        - Entity length (longer = more confident)
        - Capitalization pattern
        - Context words
        """
        # Simple heuristic - can be improved with model scores
        base_confidence = 0.5
        
        # Longer entities are more confident
        if len(ent.text) > 10:
            base_confidence += 0.2
        
        # Proper capitalization increases confidence
        if ent.text[0].isupper():
            base_confidence += 0.1
        
        # Known patterns
        if ent.label_ in ["PERSON", "ORG"] and " " in ent.text:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
```

---

#### Custom Entity Types for Research Papers

```python
class ResearchPaperNER(ScientificNER):
    """Extended NER for research paper concepts"""
    
    def __init__(self):
        super().__init__()
        
        # Add custom patterns for research concepts
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add patterns for research-specific entities"""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Method patterns
        method_patterns = [
            [{"LOWER": {"IN": ["cnn", "rnn", "lstm", "gru", "transformer"]}},
            [{"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "support"}, {"LOWER": "vector"}, {"LOWER": "machine"}],
            [{"LOWER": "random"}, {"LOWER": "forest"}],
        ]
        
        self.matcher.add("METHOD", method_patterns)
        
        # Dataset patterns
        dataset_patterns = [
            [{"UPPER": {"REGEX": r"^[A-Z]+\d*$"}}],  # MNIST, CIFAR10
            [{"LOWER": {"IN": ["imagenet", "coco", "squad", "glue"]}},
        ]
        
        self.matcher.add("DATASET", dataset_patterns)
        
        # Metric patterns
        metric_patterns = [
            [{"LOWER": {"IN": ["accuracy", "precision", "recall", "f1"]}},
            [{"LOWER": "f1"}, {"LOWER": "score"}],
            [{"LOWER": {"IN": ["bleu", "rouge", "meteor"]}},
            [{"LOWER": "mean"}, {"LOWER": "squared"}, {"LOWER": "error"}],
        ]
        
        self.matcher.add("METRIC", metric_patterns)
    
    def extract_research_entities(self, text: str) -> List[Entity]:
        """Extract both standard and research-specific entities"""
        # Get standard entities
        entities = self.extract_entities(text)
        
        # Add custom entities
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            entity = Entity(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                confidence=0.8  # High confidence for pattern matches
            )
            entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping highest confidence"""
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        deduplicated = []
        used_spans = set()
        
        for entity in entities:
            span = (entity.start_char, entity.end_char)
            
            # Check for overlap with existing entities
            overlaps = any(
                (span[0] < used[1] and span[1] > used[0])
                for used in used_spans
            )
            
            if not overlaps:
                deduplicated.append(entity)
                used_spans.add(span)
        
        return deduplicated
```

---

### 10.2 Keyphrase Extraction with KeyBERT

#### Understanding KeyBERT

**What is KeyBERT?**
- Uses BERT embeddings to find keyphrases most similar to document
- Extracts phrases that best represent document content
- Better than TF-IDF for semantic understanding

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

class KeyphraseExtractor:
    """Extract keyphrases using KeyBERT"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize KeyBERT with sentence transformer.
        
        Model choices:
        - all-MiniLM-L6-v2: Fast, good quality
        - all-mpnet-base-v2: Best quality, slower
        - sentence-t5-base: Good for scientific text
        """
        sentence_model = SentenceTransformer(model_name)
        self.kw_model = KeyBERT(model=sentence_model)
    
    def extract_keyphrases(self, text: str, top_k: int = 10) -> List[Keyphrase]:
        """
        Extract top keyphrases from text.
        
        Args:
            text: Input text
            top_k: Number of keyphrases to extract
        
        Returns:
            List of keyphrases with scores
        """
        # Extract keyphrases with scores
        keyphrases = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
            stop_words='english',
            top_k=top_k,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.5
        )
        
        # Convert to Keyphrase objects
        result = []
        for phrase, score in keyphrases:
            kp = Keyphrase(
                phrase=phrase,
                score=score,
                ngram_length=len(phrase.split())
            )
            result.append(kp)
        
        return result
    
    def extract_with_candidates(self, text: str, candidates: List[str]) -> List[Keyphrase]:
        """
        Extract keyphrases from predefined candidates.
        
        Useful when you have domain-specific terms to check.
        """
        keyphrases = self.kw_model.extract_keywords(
            text,
            candidates=candidates,
            top_k=len(candidates)
        )
        
        return [Keyphrase(phrase=phrase, score=score) 
                for phrase, score in keyphrases if score > 0.3]
```

---

#### Advanced Keyphrase Extraction

```python
class AdvancedKeyphraseExtractor(KeyphraseExtractor):
    """Advanced keyphrase extraction with domain knowledge"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        
        # Load domain-specific candidates
        self.ml_terms = self._load_ml_terms()
        self.research_terms = self._load_research_terms()
    
    def extract_scientific_keyphrases(self, text: str) -> List[Keyphrase]:
        """Extract keyphrases optimized for scientific text"""
        
        # 1. Extract general keyphrases
        general_kp = self.extract_keyphrases(text, top_k=15)
        
        # 2. Extract from ML/AI candidates
        ml_candidates = [term for term in self.ml_terms if term.lower() in text.lower()]
        ml_kp = self.extract_with_candidates(text, ml_candidates)
        
        # 3. Extract from research methodology candidates
        research_candidates = [term for term in self.research_terms if term.lower() in text.lower()]
        research_kp = self.extract_with_candidates(text, research_candidates)
        
        # 4. Combine and deduplicate
        all_keyphrases = general_kp + ml_kp + research_kp
        return self._deduplicate_keyphrases(all_keyphrases)
    
    def _load_ml_terms(self) -> List[str]:
        """Load machine learning terminology"""
        return [
            "neural network", "deep learning", "machine learning",
            "convolutional neural network", "recurrent neural network",
            "transformer", "attention mechanism", "self-attention",
            "gradient descent", "backpropagation", "overfitting",
            "regularization", "dropout", "batch normalization",
            "transfer learning", "fine-tuning", "pre-training"
        ]
    
    def _load_research_terms(self) -> List[str]:
        """Load research methodology terms"""
        return [
            "experimental design", "statistical significance",
            "cross-validation", "hyperparameter tuning",
            "baseline model", "state-of-the-art", "benchmark",
            "ablation study", "error analysis", "qualitative analysis"
        ]
    
    def _deduplicate_keyphrases(self, keyphrases: List[Keyphrase]) -> List[Keyphrase]:
        """Remove duplicate keyphrases, keeping highest scores"""
        seen = {}
        
        for kp in keyphrases:
            phrase_lower = kp.phrase.lower()
            if phrase_lower not in seen or kp.score > seen[phrase_lower].score:
                seen[phrase_lower] = kp
        
        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)
```

---

### 10.3 Concept Normalization

#### The Normalization Challenge

**Problem:** Same concepts expressed differently
- "BERT" vs "Bidirectional Encoder Representations from Transformers"
- "CNN" vs "Convolutional Neural Network" vs "ConvNet"
- "SVM" vs "Support Vector Machine"

**Solution:** Normalize to canonical forms

```python
class ConceptNormalizer:
    """Normalizes concepts to canonical forms"""
    
    def __init__(self):
        self.normalization_rules = self._load_normalization_rules()
        self.abbreviation_map = self._load_abbreviations()
    
    def normalize_concept(self, concept: str) -> str:
        """
        Normalize concept to canonical form.
        
        Steps:
        1. Convert to lowercase
        2. Expand abbreviations
        3. Apply normalization rules
        4. Remove extra whitespace
        """
        # Step 1: Lowercase
        normalized = concept.lower().strip()
        
        # Step 2: Expand abbreviations
        normalized = self._expand_abbreviations(normalized)
        
        # Step 3: Apply rules
        normalized = self._apply_rules(normalized)
        
        # Step 4: Clean whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        for abbrev, full_form in self.abbreviation_map.items():
            # Match whole words only
            import re
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_rules(self, text: str) -> str:
        """Apply normalization rules"""
        for pattern, replacement in self.normalization_rules:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load abbreviation mappings"""
        return {
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "lstm": "long short-term memory",
            "gru": "gated recurrent unit",
            "svm": "support vector machine",
            "rf": "random forest",
            "knn": "k-nearest neighbors",
            "pca": "principal component analysis",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "dl": "deep learning"
        }
    
    def _load_normalization_rules(self) -> List[Tuple[str, str]]:
        """Load normalization rules as (pattern, replacement) pairs"""
        return [
            (r"convnet", "convolutional neural network"),
            (r"conv net", "convolutional neural network"),
            (r"neural net", "neural network"),
            (r"deep net", "deep neural network"),
            (r"feedforward", "feed-forward"),
            (r"multi-layer perceptron", "multilayer perceptron"),
            (r"mlp", "multilayer perceptron")
        ]
```

---

### 10.4 Relationship Extraction

#### Co-occurrence Based Relationships

```python
class RelationshipExtractor:
    """Extract relationships between concepts"""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize with co-occurrence window size.
        
        Args:
            window_size: Number of tokens to consider for co-occurrence
        """
        self.window_size = window_size
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships based on co-occurrence.
        
        Method:
        1. Find entities that appear within window_size tokens
        2. Calculate co-occurrence strength
        3. Infer relationship types based on context
        """
        relationships = []
        
        # Tokenize text
        tokens = text.split()
        
        # Find entity positions in tokens
        entity_positions = self._find_entity_positions(tokens, entities)
        
        # Find co-occurring entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                
                # Check if entities co-occur within window
                cooccurrence = self._check_cooccurrence(
                    entity_positions[i], entity_positions[j]
                )
                
                if cooccurrence:
                    relationship = Relationship(
                        subject=entity1.text,
                        object=entity2.text,
                        predicate=self._infer_relationship_type(entity1, entity2, text),
                        strength=cooccurrence['strength'],
                        context=cooccurrence['context']
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _check_cooccurrence(self, pos1: List[int], pos2: List[int]) -> Optional[Dict]:
        """Check if two entities co-occur within window"""
        min_distance = float('inf')
        best_context = ""
        
        for p1 in pos1:
            for p2 in pos2:
                distance = abs(p1 - p2)
                if distance <= self.window_size and distance < min_distance:
                    min_distance = distance
                    # Extract context around co-occurrence
                    start = max(0, min(p1, p2) - 10)
                    end = min(len(tokens), max(p1, p2) + 10)
                    best_context = ' '.join(tokens[start:end])
        
        if min_distance <= self.window_size:
            # Strength inversely related to distance
            strength = 1.0 - (min_distance / self.window_size)
            return {
                'strength': strength,
                'context': best_context,
                'distance': min_distance
            }
        
        return None
    
    def _infer_relationship_type(self, entity1: Entity, entity2: Entity, text: str) -> str:
        """
        Infer relationship type based on entity types and context.
        
        Heuristics:
        - PERSON + METHOD = "uses" or "proposes"
        - METHOD + DATASET = "evaluated_on"
        - METHOD + METRIC = "measured_by"
        - PERSON + ORG = "affiliated_with"
        """
        type1, type2 = entity1.label, entity2.label
        
        # Person-Method relationships
        if type1 == "PERSON" and type2 == "METHOD":
            if "propose" in text or "introduce" in text:
                return "proposes"
            else:
                return "uses"
        
        # Method-Dataset relationships
        elif type1 == "METHOD" and type2 == "DATASET":
            return "evaluated_on"
        
        # Method-Metric relationships
        elif type1 == "METHOD" and type2 == "METRIC":
            return "measured_by"
        
        # Person-Organization relationships
        elif type1 == "PERSON" and type2 == "ORG":
            return "affiliated_with"
        
        # Default relationship
        else:
            return "related_to"
```

---

### 10.5 Extraction Output Format

#### Structured Output Schema

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
import json

@dataclass
class Entity:
    """Named entity with metadata"""
    text: str
    label: str  # PERSON, ORG, METHOD, DATASET, etc.
    start_char: int
    end_char: int
    confidence: float
    normalized_form: Optional[str] = None

@dataclass
class Keyphrase:
    """Extracted keyphrase with score"""
    phrase: str
    score: float
    ngram_length: int
    category: Optional[str] = None  # method, dataset, metric, etc.

@dataclass
class Relationship:
    """Relationship between entities"""
    subject: str
    predicate: str  # uses, proposes, evaluated_on, etc.
    object: str
    strength: float
    context: str
    relationship_type: str = "co_occurrence"

@dataclass
class ConceptExtractionResult:
    """Complete concept extraction output"""
    chunk_id: str
    document_id: str
    
    # Extracted concepts
    entities: List[Entity]
    keyphrases: List[Keyphrase]
    relationships: List[Relationship]
    
    # Metadata
    extraction_date: str
    model_versions: Dict[str, str]
    processing_time: float
    
    def to_json(self) -> str:
        """Convert to JSON for storage"""
        return json.dumps(asdict(self), indent=2, default=str)
    
    def get_all_concepts(self) -> List[str]:
        """Get all unique concepts (entities + keyphrases)"""
        concepts = set()
        
        # Add entities
        for entity in self.entities:
            concepts.add(entity.normalized_form or entity.text)
        
        # Add keyphrases
        for kp in self.keyphrases:
            concepts.add(kp.phrase)
        
        return sorted(list(concepts))
```

---

#### Example Output

```json
{
  "chunk_id": "chunk_abc123",
  "document_id": "doc_def456",
  "entities": [
    {
      "text": "BERT",
      "label": "METHOD",
      "start_char": 45,
      "end_char": 49,
      "confidence": 0.95,
      "normalized_form": "bidirectional encoder representations from transformers"
    },
    {
      "text": "ImageNet",
      "label": "DATASET",
      "start_char": 120,
      "end_char": 128,
      "confidence": 0.98,
      "normalized_form": "imagenet"
    }
  ],
  "keyphrases": [
    {
      "phrase": "transformer architecture",
      "score": 0.87,
      "ngram_length": 2,
      "category": "method"
    },
    {
      "phrase": "attention mechanism",
      "score": 0.82,
      "ngram_length": 2,
      "category": "method"
    }
  ],
  "relationships": [
    {
      "subject": "BERT",
      "predicate": "uses",
      "object": "attention mechanism",
      "strength": 0.9,
      "context": "BERT uses multi-head attention mechanism to...",
      "relationship_type": "co_occurrence"
    }
  ],
  "extraction_date": "2024-01-15T10:30:00Z",
  "model_versions": {
    "ner_model": "en_core_sci_md-0.5.1",
    "keyphrase_model": "all-MiniLM-L6-v2",
    "sentence_model": "all-MiniLM-L6-v2"
  },
  "processing_time": 2.3
}
```
---

## Deep Learning Models

### Model Selection and Rationale

#### Sentence Embeddings: all-MiniLM-L6-v2

**Why this model:**
- **Speed:** Fast inference (important for chunking many documents)
- **Quality:** Good semantic understanding for general text
- **Size:** 384 dimensions (manageable memory usage)
- **Compatibility:** Works well with ChromaDB and other vector stores

**Alternatives considered:**
- `all-mpnet-base-v2`: Better quality but slower (768 dimensions)
- `sentence-t5-base`: Good for scientific text but larger
- `all-distilroberta-v1`: Fast but lower quality

**Trade-offs:**
- ✅ Fast inference, reasonable quality, manageable size
- ❌ Not specifically trained on scientific text
- ❌ Lower quality than larger models

---

#### Scientific NER: en_core_sci_md

**Why this model:**
- **Domain-specific:** Trained on scientific literature
- **Entity types:** Recognizes scientific entities (chemicals, diseases, etc.)
- **Accuracy:** Better than general models for scientific text
- **Integration:** Works with spaCy ecosystem

**Installation:**
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

**Alternatives:**
- `en_core_web_sm`: General model, misses scientific entities
- `en_ner_bc5cdr_md`: Biomedical specific (chemicals, diseases)
- `en_ner_bionlp13cg_md`: Biology specific (genes, proteins)

---

#### KeyBERT for Keyphrase Extraction

**Why KeyBERT:**
- **Semantic understanding:** Uses BERT embeddings
- **Quality:** Better than TF-IDF for semantic keyphrases
- **Flexibility:** Can use different embedding models
- **Diversity:** MMR algorithm ensures diverse keyphrases

**How it works:**
1. Compute document embedding
2. Compute candidate phrase embeddings
3. Find phrases most similar to document
4. Use MMR to ensure diversity

---

### Model Performance Considerations

#### GPU vs CPU Inference

```python
class ModelManager:
    """Manages model loading and inference optimization"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._select_device(device)
        self.models = {}
    
    def _select_device(self, device: str) -> str:
        """Select optimal device for inference"""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_embedding_model(self, model_name: str):
        """Load embedding model with device optimization"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_name, device=self.device)
        
        # Optimize for inference
        if self.device == "cuda":
            model.half()  # Use FP16 for faster inference
        
        self.models['embedding'] = model
        return model
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for efficiency"""
        model = self.models['embedding']
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
```

---

#### Memory Management

```python
class MemoryEfficientProcessor:
    """Process documents with memory constraints"""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.current_memory = 0
    
    def process_chunks_in_batches(self, chunks: List[Chunk]) -> List[ConceptExtractionResult]:
        """Process chunks in memory-efficient batches"""
        results = []
        batch = []
        
        for chunk in chunks:
            # Estimate memory usage
            chunk_memory = len(chunk.text) * 0.001  # Rough estimate
            
            if self.current_memory + chunk_memory > self.max_memory_mb:
                # Process current batch
                if batch:
                    batch_results = self._process_batch(batch)
                    results.extend(batch_results)
                    batch = []
                    self.current_memory = 0
            
            batch.append(chunk)
            self.current_memory += chunk_memory
        
        # Process final batch
        if batch:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
```

---

## Learning Outcomes

### Skills Learned in Phase 2

**1. Semantic Understanding**
- How embeddings capture semantic meaning
- Cosine similarity for measuring text similarity
- Semantic boundary detection algorithms

**2. Deep Learning for NLP**
- Using pre-trained transformer models
- Sentence-BERT for embeddings
- Domain-specific models (scientific NER)

**3. Information Extraction**
- Named Entity Recognition (NER)
- Keyphrase extraction with KeyBERT
- Relationship extraction from co-occurrence

**4. Text Normalization**
- Concept normalization strategies
- Handling abbreviations and synonyms
- Canonical form mapping

**5. Model Integration**
- Loading and using multiple models
- Batch processing for efficiency
- Memory management for large datasets

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: "Model not found" error**
- **Cause:** Scientific models not installed
- **Solution:** Install scispacy and download models:
  ```bash
  pip install scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
  ```

**Issue 2: "CUDA out of memory"**
- **Cause:** GPU memory exhausted
- **Solutions:** 
  - Reduce batch size
  - Use CPU instead of GPU
  - Use smaller models (MiniLM instead of MPNet)

**Issue 3: "Poor chunking quality"**
- **Cause:** Similarity threshold too high/low
- **Solutions:**
  - Adjust similarity threshold (try 0.6-0.8)
  - Use different embedding model
  - Check text preprocessing

**Issue 4: "Missing scientific entities"**
- **Cause:** Using general NER model
- **Solutions:**
  - Use scientific NER model (en_core_sci_md)
  - Add custom patterns for domain terms
  - Combine multiple NER models

**Issue 5: "Slow processing"**
- **Cause:** Large models, CPU inference
- **Solutions:**
  - Use GPU if available
  - Reduce batch sizes
  - Use smaller/faster models
  - Process in parallel

---

## Success Criteria

Phase 2 is successful when:

✅ **Semantic Chunking**
- Documents are chunked at semantic boundaries
- Chunks maintain coherent topics
- Chunk sizes meet token constraints (100-500 tokens)

✅ **Entity Recognition**
- Scientific entities are identified accurately
- Confidence scores are reasonable (>0.7 for high-confidence)
- Custom patterns catch domain-specific terms

✅ **Keyphrase Extraction**
- Relevant keyphrases are extracted
- Keyphrases represent document content well
- Diversity is maintained (no redundant phrases)

✅ **Concept Normalization**
- Abbreviations are expanded correctly
- Synonyms are mapped to canonical forms
- Normalization rules work consistently

✅ **Relationship Extraction**
- Co-occurring entities are identified
- Relationship types are inferred reasonably
- Relationship strengths reflect proximity

✅ **Output Quality**
- Structured JSON output is generated
- All required fields are populated
- Processing is efficient and scalable

---

## Next Steps

After completing Phase 2, you'll have:
- Semantically chunked documents
- Extracted concepts and entities
- Identified relationships between concepts
- Structured data ready for knowledge graphs

**Phase 3** will build on this foundation by:
- Constructing knowledge graphs from extracted concepts
- Creating nodes and relationships in Neo4j
- Enabling graph-based queries and discovery
- Aggregating concept information across documents

---

**Phase 2 demonstrates advanced NLP skills including semantic understanding, entity recognition, and concept extraction - core competencies for AI/ML engineering roles.**
