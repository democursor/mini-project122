# Phase 2 Setup Instructions

## Install Dependencies

```cmd
pip install sentence-transformers scikit-learn nltk spacy keybert
```

## Download Required Models

### 1. SpaCy Model
```cmd
python -m spacy download en_core_web_sm
```

### 2. NLTK Data (automatic on first run)
The punkt tokenizer will download automatically when needed.

## Optional: Scientific NER Model

For better entity recognition in scientific papers:

```cmd
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

## Test Installation

```cmd
python test_phase2.py
```

Expected output:
```
Testing Phase 2 Implementation...
--------------------------------------------------
✓ All Phase 2 imports successful
✓ Chunker initialized (threshold: 0.7)
✓ Extractor initialized
--------------------------------------------------
Phase 2 implementation is ready!
```

## Run Phase 1 & 2

```cmd
python main.py
```

## What Phase 2 Does

1. **Semantic Chunking**
   - Splits documents into meaningful chunks
   - Uses sentence embeddings to detect topic boundaries
   - Maintains chunks between 100-500 tokens

2. **Concept Extraction**
   - Named Entity Recognition (NER) for people, organizations, methods
   - Keyphrase extraction using KeyBERT
   - Confidence scores for each extraction

## Output

Results are saved to `data/parsed/<document_id>.json` with:
- Parsed text and metadata (Phase 1)
- Semantic chunks (Phase 2)
- Extracted entities and keyphrases (Phase 2)

## Troubleshooting

**Issue**: "No module named 'sentence_transformers'"
```cmd
pip install sentence-transformers
```

**Issue**: "No module named 'keybert'"
```cmd
pip install keybert
```

**Issue**: "Can't find model 'en_core_web_sm'"
```cmd
python -m spacy download en_core_web_sm
```

**Issue**: Models downloading slowly
- First run downloads ~100MB of models
- Subsequent runs use cached models
- Be patient on first execution

## Performance Notes

- **First run**: Slower (downloading models)
- **Subsequent runs**: Faster (cached models)
- **CPU**: Works fine, ~10-30 seconds per document
- **GPU**: Much faster if available (~2-5 seconds)

---

**Phase 2 is ready! Test with your PDF files.**
