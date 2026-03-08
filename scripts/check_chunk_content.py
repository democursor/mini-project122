from src.vector.store import VectorStore

store = VectorStore('./data/chroma')
results = store.collection.get(limit=2, include=['documents', 'metadatas'])

print(f'Total chunks: {store.collection.count()}\n')

for i, (doc_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
    print(f'--- Chunk {i+1} ---')
    print(f'ID: {doc_id}')
    print(f'Document ID: {metadata.get("document_id")}')
    print(f'Section: {metadata.get("section_heading")}')
    print(f'Text preview: {document[:200]}...')
    print()
