from src.vector.store import VectorStore

store = VectorStore('./data/chroma')
print(f'Collection name: {store.collection.name}')
print(f'Total chunks: {store.collection.count()}')

if store.collection.count() > 0:
    results = store.collection.get(limit=5)
    print(f'\nSample document IDs: {results["ids"][:5]}')
    print(f'Sample metadatas: {results["metadatas"][:5]}')
