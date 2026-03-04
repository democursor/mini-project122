"""
Clear all data from the system - Neo4j, ChromaDB, PDFs, and metadata
"""
from src.utils.config import Config
from src.graph.builder import KnowledgeGraphBuilder
from src.vector.store import VectorStore
from pathlib import Path
import shutil
import json

print("=" * 60)
print("CLEARING ALL DATA")
print("=" * 60)

config = Config()

# 1. Clear Neo4j Knowledge Graph
print("\n1. Clearing Neo4j Knowledge Graph...")
try:
    builder = KnowledgeGraphBuilder(
        uri=config.get('neo4j.uri'),
        user=config.get('neo4j.user'),
        password=config.get('neo4j.password'),
        database=config.get('neo4j.database')
    )
    builder.clear_graph()
    builder.close()
    print("   ✓ Neo4j graph cleared")
except Exception as e:
    print(f"   ✗ Error clearing Neo4j: {e}")

# 2. Clear ChromaDB Vector Store
print("\n2. Clearing ChromaDB Vector Store...")
try:
    chroma_dir = Path(config.get('vector.persist_directory', './data/chroma'))
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print(f"   ✓ Deleted: {chroma_dir}")
    else:
        print(f"   - Directory doesn't exist: {chroma_dir}")
except Exception as e:
    print(f"   ✗ Error clearing ChromaDB: {e}")

# 3. Clear PDF Storage
print("\n3. Clearing PDF Storage...")
try:
    pdf_dir = Path(config.get('storage.pdf_directory', './data/pdfs'))
    if pdf_dir.exists():
        for item in pdf_dir.rglob('*'):
            if item.is_file():
                item.unlink()
                print(f"   ✓ Deleted: {item.name}")
        print(f"   ✓ All PDFs cleared from {pdf_dir}")
    else:
        print(f"   - Directory doesn't exist: {pdf_dir}")
except Exception as e:
    print(f"   ✗ Error clearing PDFs: {e}")

# 4. Clear Parsed Data
print("\n4. Clearing Parsed Data...")
try:
    parsed_dir = Path('./data/parsed')
    if parsed_dir.exists():
        for item in parsed_dir.rglob('*.json'):
            item.unlink()
            print(f"   ✓ Deleted: {item.name}")
        print(f"   ✓ All parsed data cleared")
    else:
        print(f"   - Directory doesn't exist: {parsed_dir}")
except Exception as e:
    print(f"   ✗ Error clearing parsed data: {e}")

# 5. Clear Document Metadata
print("\n5. Clearing Document Metadata...")
try:
    metadata_file = Path('./data/documents_metadata.json')
    if metadata_file.exists():
        metadata_file.write_text('{}')
        print(f"   ✓ Metadata cleared: {metadata_file}")
    else:
        print(f"   - File doesn't exist: {metadata_file}")
except Exception as e:
    print(f"   ✗ Error clearing metadata: {e}")

print("\n" + "=" * 60)
print("DATA CLEANUP COMPLETE")
print("=" * 60)
print("\nAll data has been cleared. You can now upload new documents.")
print("The system will start fresh with:")
print("  - Empty knowledge graph")
print("  - Empty vector database")
print("  - No stored PDFs")
print("  - No document metadata")
