from src.vector.search import SemanticSearchEngine, QueryProcessor
from src.vector.embedder import EmbeddingGenerator, EmbeddingConfig
from src.vector.store import VectorStore

# Initialize components
embedding_config = EmbeddingConfig(model_name='all-MiniLM-L6-v2', device='cpu')
embedding_generator = EmbeddingGenerator(embedding_config)
vector_store = VectorStore('./data/chroma')
query_processor = QueryProcessor(embedding_generator)
search_engine = SemanticSearchEngine(vector_store, query_processor)

# Test search
query = "what is cancer"
results = search_engine.search(query, top_k=3)

print(f'Query: {query}')
print(f'Found {len(results)} results\n')

for i, result in enumerate(results):
    print(f'--- Result {i+1} ---')
    print(f'Score: {result.get("score", "N/A")}')
    print(f'Document ID: {result["metadata"].get("document_id")}')
    print(f'Section: {result["metadata"].get("section_heading")}')
    print(f'Text: {result["text"][:300]}...')
    print()
