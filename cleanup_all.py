"""Clean all data - PDFs, parsed files, vectors, and Neo4j graph"""
import shutil
from pathlib import Path
from src.utils.config import Config
from src.graph.builder import KnowledgeGraphBuilder
import chromadb

def cleanup_all():
    print("=" * 60)
    print("CLEANING ALL DATA")
    print("=" * 60)
    
    # 1. Clean PDFs
    print("\n1. Cleaning PDFs...")
    pdf_dir = Path("data/pdfs")
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)
        print(f"   ✓ Deleted {pdf_dir}")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    print("   ✓ PDFs cleaned")
    
    # 2. Clean parsed JSON files
    print("\n2. Cleaning parsed JSON files...")
    parsed_dir = Path("data/parsed")
    if parsed_dir.exists():
        for file in parsed_dir.glob("*.json"):
            file.unlink()
            print(f"   ✓ Deleted {file.name}")
    else:
        parsed_dir.mkdir(parents=True, exist_ok=True)
    print("   ✓ Parsed files cleaned")
    
    # 3. Clean ChromaDB vectors
    print("\n3. Cleaning ChromaDB vectors...")
    chroma_dir = Path("data/chroma")
    if chroma_dir.exists():
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()
            for collection in collections:
                client.delete_collection(collection.name)
                print(f"   ✓ Deleted collection: {collection.name}")
        except Exception as e:
            print(f"   ⚠ Error cleaning ChromaDB: {e}")
            # Force delete directory
            shutil.rmtree(chroma_dir)
            chroma_dir.mkdir(parents=True, exist_ok=True)
    print("   ✓ ChromaDB cleaned")
    
    # 4. Clean Neo4j graph
    print("\n4. Cleaning Neo4j graph...")
    try:
        config = Config().config
        if config.get('neo4j'):
            graph_builder = KnowledgeGraphBuilder(
                uri=config['neo4j']['uri'],
                user=config['neo4j']['user'],
                password=config['neo4j']['password'],
                database=config['neo4j']['database']
            )
            graph_builder.clear_graph()
            graph_builder.close()
            print("   ✓ Neo4j graph cleared")
        else:
            print("   ⚠ Neo4j not configured")
    except Exception as e:
        print(f"   ⚠ Error cleaning Neo4j: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ALL DATA CLEANED - FRESH START READY")
    print("=" * 60)

if __name__ == "__main__":
    cleanup_all()
