"""Test Phase 2 Implementation"""
import sys
from pathlib import Path

print("Testing Phase 2 Implementation...")
print("-" * 50)

# Test imports
try:
    from src.chunking import SemanticChunker, Chunk, ChunkingConfig
    from src.extraction import ConceptExtractor, Entity, Keyphrase
    print("✓ All Phase 2 imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test chunking
try:
    config = ChunkingConfig()
    chunker = SemanticChunker(config)
    print(f"✓ Chunker initialized (threshold: {config.similarity_threshold})")
except Exception as e:
    print(f"✗ Chunker initialization error: {e}")
    sys.exit(1)

# Test extraction
try:
    extractor = ConceptExtractor()
    print("✓ Extractor initialized")
except Exception as e:
    print(f"✗ Extractor initialization error: {e}")
    sys.exit(1)

print("-" * 50)
print("Phase 2 implementation is ready!")
print("\nNote: Models will be downloaded on first use:")
print("  - sentence-transformers/all-MiniLM-L6-v2 (~80MB)")
print("  - spacy en_core_web_sm (~12MB)")
print("\nRun 'python main.py' to test with a PDF")
