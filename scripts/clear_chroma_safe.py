"""
Safely clear ChromaDB by deleting and recreating the collection
"""
import chromadb
from pathlib import Path

print("=" * 60)
print("CLEARING CHROMADB SAFELY")
print("=" * 60)

try:
    # Connect to ChromaDB
    chroma_dir = Path("./data/chroma")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Get all collections
    collections = client.list_collections()
    print(f"\nFound {len(collections)} collection(s)")
    
    # Delete each collection
    for collection in collections:
        print(f"Deleting collection: {collection.name}")
        client.delete_collection(collection.name)
        print(f"✓ Deleted: {collection.name}")
    
    print("\n" + "=" * 60)
    print("CHROMADB CLEARED SUCCESSFULLY")
    print("=" * 60)
    print("\nAll vector embeddings have been removed.")
    print("You can now upload new documents.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nIf the error persists, try:")
    print("1. Stop the backend server")
    print("2. Run: python clear_all_data.py")
    print("3. Restart the backend server")
