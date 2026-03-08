"""Check Neo4j graph statistics"""
from src.graph import GraphQueryEngine
from src.utils.config import Config
from neo4j import GraphDatabase

config = Config()

driver = GraphDatabase.driver(
    config.get('neo4j.uri'),
    auth=(config.get('neo4j.user'), config.get('neo4j.password'))
)

engine = GraphQueryEngine(driver, config.get('neo4j.database'))
stats = engine.get_graph_statistics()

print("=" * 60)
print("Knowledge Graph Statistics")
print("=" * 60)
print(f"Papers: {stats['papers']}")
print(f"Concepts: {stats['concepts']}")
print(f"Mentions: {stats['mentions']}")
print(f"Relationships: {stats['relationships']}")

print("\nTop 10 Concepts:")
concepts = engine.get_all_concepts(limit=10)
for i, concept in enumerate(concepts, 1):
    print(f"  {i}. {concept['name']} ({concept['type']}): {concept['frequency']} mentions")

driver.close()
print("\n✓ Graph is accessible!")
print("Open Neo4j Browser: http://localhost:7474")
